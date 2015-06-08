"""The event-based main loop of Blocks."""
import sqlite3
from abc import ABCMeta
from collections import defaultdict, MutableMapping, Mapping
from itertools import repeat
from numbers import Integral
from operator import itemgetter
from uuid import uuid4, UUID

from six import add_metaclass
from six.moves import map

from blocks.config import config


def _sub_string(replacements):
    """Return a certain number of ? to be substitutede by `replacements`."""
    return '({})'.format(', '.join(repeat('?', len(replacements))))


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
        self.status.update({
            'iterations_done': 0,
            'epochs_done': 0,
            'resumed_from': None
        })

    @property
    def b_uuid(self):
        return sqlite3.Binary(self.uuid.bytes)

    def resume(self):
        """Resume a log by setting a new random UUID."""
        old_uuid = self.b_uuid
        old_status = dict(self.status)
        self.uuid = uuid4()
        self.status.update(old_status)
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

    @property
    def conn(self):
        return self.log.conn

    def __getitem__(self, key):
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
        return self.conn.execute("SELECT COUNT(*) FROM status WHERE uuid = ?",
                                 (self.log.b_uuid,)).fetchone()[0]

    def __iter__(self):
        return map(itemgetter(0), self.conn.execute(
            "SELECT key FROM status WHERE uuid = ?", (self.log.b_uuid,)
        ))


class SQLiteEntry(MutableMapping):
    def __init__(self, log, time):
        self.log = log
        self.time = time

    @property
    def conn(self):
        return self.log.conn

    def __getitem__(self, key):
        for ancestor_b_uuid in self.log.ancestors:
            value = self.conn.execute(
                "SELECT value FROM entries WHERE uuid = ? AND time = ? "
                "AND key = ?", (ancestor_b_uuid, self.time, key)
            ).fetchone()
            if value is None:
                continue
            else:
                return value[0]
        raise KeyError(key)

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
        return self.conn.execute(
            "SELECT COUNT(*) FROM entries WHERE uuid IN {} "
            "AND time = ?".format(_sub_string(self.log.ancestors)),
            tuple(self.log.ancestors) + (self.time,)
        ).fetchone()[0]

    def __iter__(self):
        return map(itemgetter(0), self.conn.execute(
            "SELECT key FROM entries WHERE uuid = IN {} "
            "AND time = ?".format(_sub_string(self.log.ancestors)),
            tuple(self.log.ancestors) + (self.time,)
        ))


class SQLiteLog(_TrainingLog, Mapping):
    """Training log using SQLite as a backend.

    Parameters
    ----------
    database : str, optional
        The database (file) to connect to. Can also be `:memory:`. See
        :func:`sqlite3.connect` for details. Uses `config.sqlite_database`
        by default.

    Notes
    -----
    .. todo::

       Currently this log ignores previous logs in case of resumption.

    """
    def __init__(self, database=None):
        if database is None:
            database = config.sqlite_database
        self.database = database
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['conn']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = sqlite3.connect(self.database)

    @property
    def ancestors(self):
        """A list of ancestral logs, including this one."""
        if hasattr(self, '_ancestors') and self.b_uuid in self._ancestors:
            return [sqlite3.Binary(a.bytes) for a in self._ancestors]
        ancestors = [self.b_uuid]
        while True:
            parent = self.conn.execute(
                "SELECT value FROM status WHERE uuid = ? AND "
                "key = 'resumed_from'", (ancestors[-1],)
            ).fetchone()
            if parent is None or parent[0] is None:
                break
            ancestors.append(parent[0])
        self._ancestors = [UUID(bytes=a) for a in ancestors]
        return [sqlite3.Binary(a.bytes) for a in self._ancestors]

    def __getitem__(self, time):
        self._check_time(time)
        return SQLiteEntry(self, time)

    def __iter__(self):
        return map(itemgetter(0), self.conn.execute(
            "SELECT DISTINCT time FROM entries WHERE uuid IN {} "
            "ORDER BY time ASC".format(_sub_string(self.ancestors)),
            tuple(self.ancestors)
        ))

    def __len__(self):
        return self.conn.execute(
            "SELECT COUNT(DISTINCT time) FROM entries "
            "WHERE uuid IN {}".format(_sub_string(self.ancestors)),
            tuple(self.ancestors)
        ).fetchone()[0]


class TrainingLog(defaultdict, _TrainingLog):
    """Training log using a `defaultdict` as backend."""
    def __init__(self):
        defaultdict.__init__(self, dict)
        self.status = {}
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
