"""The event-based main loop of Blocks."""
import sqlite3
from abc import ABCMeta
from collections import defaultdict, MutableMapping, Mapping
from numbers import Integral
from operator import itemgetter
from uuid import uuid4

from six import add_metaclass
from six.moves import map


@add_metaclass(ABCMeta)
class _TrainingLog(object):
    """Base class for training log.

    A training log stores the training timeline, statistics and other
    auxiliary information. Information is stored as a nested dictionary,
    ``log[time][key]``. An entry without stored data will return an empty
    dictionary (like ``defaultdict(dict)``).

    In addition to the set of records displaying training dynamics, a
    training log has a :attr:`status` attribute, which is a dictionary with
    data that is not bound to a particular time.

    Parameters
    ----------
    uuid : :class:`uuid.UUID`, optional
        The UUID of this log. For persistent log backends, passing the UUID
        will result in an old log being loaded.

    Attributes
    ----------
    status : dict
        A dictionary with data representing the current state of training.
        By default it contains ``iterations_done`` and ``epochs_done``.

    Notes
    -----
    For analysis of the logs, it can be useful to convert the log to a
    Pandas_ data frame:

    .. code:: python

       df = DataFrame.from_dict(log, orient='index')

    .. _Pandas: http://pandas.pydata.org

    """
    def __init__(self, uuid=None):
        if uuid is None:
            uuid = uuid4()
        self.uuid = uuid
        self.status = {
            'iterations_done': 0,
            'epochs_done': 0,
            'resumed_from': None
        }

    def resume(self):
        """Resume a log by setting a new random UUID."""
        old_uuid = self.uuid
        self.uuid = uuid4()
        self.status['resumed_from'] = old_uuid

    def _check_time(self, time):
        if not isinstance(time, Integral) or time < 0:
            raise ValueError("time must be a non-negative integer")

    @property
    def current_row(self):
        return self[self.status['iterations_done']]

    @property
    def previous_row(self):
        return self[self.status['iterations_done'] - 1]


class SQLiteStatus(MutableMapping):
    def __init__(self, log):
        self.log = log
        self.conn = log.conn

    def __getitem__(self, key):
        with self.conn:
            value = self.conn.execute(
                "SELECT value FROM status WHERE uuid = ? AND key = ?",
                (self.log.b_uuid, key)
            ).fetchone()
            if value is None:
                raise KeyError(key)
            else:
                return value[0]

    def __setitem__(self, key, value):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO status VALUES (?, ?, ?)",
                (self.log.b_uuid, key, value)
            )

    def __delitem__(self, key):
        with self.conn:
            self.conn.execute(
                "DELETE FROM status WHERE uuid = ? AND key = ?",
                (self.log.b_uuid, key)
            )

    def __len__(self):
        with self.conn:
            return self.conn.execute("SELECT COUNT(*) FROM status "
                                     "WHERE uuid = ?",
                                     (self.log.b_uuid,)).fetchone()[0]

    def __iter__(self):
        with self.conn:
            return map(itemgetter(0), self.conn.execute(
                "SELECT key FROM status WHERE uuid = ?", (self.log.b_uuid,)
            ))


class SQLiteEntry(MutableMapping):
    def __init__(self, log, time):
        self.log = log
        self.conn = log.conn
        self.time = time

    def __getitem__(self, key):
        with self.conn:
            value = self.conn.execute(
                "SELECT value FROM entries WHERE uuid = ? AND time = ? "
                "AND key = ?", (self.log.b_uuid, self.time, key)
            ).fetchone()
            if value is None:
                raise KeyError(key)
            else:
                return value[0]

    def __setitem__(self, key, value):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?)",
                (self.log.b_uuid, self.time, key, value)
            )

    def __delitem__(self, key):
        with self.conn:
            self.conn.execute(
                "DELETE FROM entries WHERE uuid = ? AND time =? AND key = ?",
                (self.log.b_uuid, self.time, key)
            )

    def __len__(self):
        with self.conn:
            return self.conn.execute("SELECT COUNT(*) FROM entries WHERE "
                                     "uuid = ? AND time = ?",
                                     (self.log.b_uuid,
                                      self.time)).fetchone()[0]

    def __iter__(self):
        with self.conn:
            return map(itemgetter(0), self.conn.execute(
                "SELECT key FROM entries WHERE uuid = ? AND time = ?",
                (self.log.b_uuid, self.time)
            ))


class SQLiteLog(_TrainingLog, Mapping):
    """Training log using SQLite as a backend.

    Parameters
    ----------
    database : str
        The database (file) to connect to. Can also be `:memory:`. See
        :func:`sqlite3.connect` for details.

    Notes
    -----
    .. todo::

       Currently this log ignores previous logs in case of resumption.

    """
    def __init__(self, database):
        self.conn = sqlite3.connect(database)
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS entries (
                                   uuid BLOB NOT NULL,
                                   time INT NOT NULL,
                                   "key" TEXT NOT NULL,
                                   value,
                                   PRIMARY KEY(uuid, time, "key")
                                 );""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS status (
                                   uuid BLOB NOT NULL,
                                   "key" text NOT NULL,
                                   value,
                                   PRIMARY KEY(uuid, "key")
                                 );""")
        self.status = SQLiteStatus(self)
        super(SQLiteLog, self).__init__()

    @property
    def b_uuid(self):
        return sqlite3.Binary(self.uuid.bytes)

    def __getitem__(self, time):
        self._check_time(time)
        return SQLiteEntry(self, time)

    def __iter__(self):
        with self.conn:
            return map(itemgetter(0), self.conn.execute(
                "SELECT DISTINCT time FROM entries WHERE uuid = ? "
                "ORDER BY time ASC",
                (self.b_uuid,)
            ))

    def __len__(self):
        with self.conn:
            return self.conn.execute("SELECT COUNT(DISTINCT time) "
                                     "FROM entries WHERE uuid = ?",
                                     (self.b_uuid,)).fetchone()[0]


class TrainingLog(defaultdict, _TrainingLog):
    """Training log using a `defaultdict` as backend."""
    def __init__(self):
        defaultdict.__init__(self, dict)
        _TrainingLog.__init__(self)

    def __reduce__(self):
        constructor, args, _, _, items = super(TrainingLog, self).__reduce__()
        return constructor, (), self.__dict__, _, items

    def __getitem__(self, time):
        self._check_time(time)
        return super(TrainingLog, self).__getitem__(time)

    def __setitem__(self, time, value):
        self._check_time(time)
        return super(TrainingLog, self).__setitem__(time, value)


BACKENDS = {
    'python': TrainingLog,
    'sqlite': SQLiteLog
}
