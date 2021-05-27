import contextlib
import uuid

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import sqlalchemy.future
import sqlalchemy.orm

from neuroAPI.utils import config as _config  # noqa


class DatabaseHandler(object):
    _engine = None
    _active_session = None

    @property
    def engine(self) -> sqlalchemy.future.Engine:

        if isinstance(self._engine, sqlalchemy.future.Engine):
            return self._engine

        _driver = _config.get('DATABASE', 'DB_DRIVER')

        if _driver == 'sqlite':
            engine = create_engine(f'{_driver}:///{_config.get("DATABASE", "DB_HOST")}')
        elif _driver == 'postgresql':
            engine = create_engine('{}://{}:{}@{}/{}'.format(_driver,
                                                             _config.get("DATABASE", "DB_USER"),
                                                             _config.get("DATABASE", "DB_PASS"),
                                                             _config.get("DATABASE", "DB_HOST"),
                                                             _config.get("DATABASE", "DB_NAME")),
                                   connect_args={"application_name": "geology-neuro-module"}, poolclass=NullPool)
        else:
            raise NotImplemented

        self._engine = engine
        return self._engine

    @contextlib.contextmanager
    def get_session(self) -> sqlalchemy.orm.Session:
        if self._active_session:
            raise NotImplementedError  # TODO: think about it
            # return self.active_session

        try:
            self._active_session = sqlalchemy.orm.Session(self.engine, future=True)
            yield self._active_session
        except Exception as e:
            if self._active_session:
                self._active_session.rollback()
            raise e
        else:
            self._active_session.commit()
        finally:
            if self._active_session:
                self._active_session.close()
            self._active_session = None

    @property
    def active_session(self):
        return self._active_session

    def get_object_by_id(self, cls: sqlalchemy.orm.declarative_base(), idx: uuid.UUID,
                         session: sqlalchemy.orm.Session = None):

        # assert issubclass(cls, sqlalchemy.orm.declarative_base())  # FIXME
        assert isinstance(idx, uuid.UUID)

        if not session:
            session = self.active_session

        if not session:
            raise NotImplementedError  # TODO: Implement

        q_result = session.execute(sqlalchemy.select(cls)
                                   .where(cls.id == idx), None, None).fetchone()

        if not q_result:
            return None
        else:
            return q_result[0]
