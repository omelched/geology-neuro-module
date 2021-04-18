from sqlalchemy import create_engine
import sqlalchemy.future
import sqlalchemy.orm

from neuroAPI.utils import config as _config  # noqa


# TODO: re-code with sessionmaker() https://docs.sqlalchemy.org/en/14/orm/session_basics.html

class DatabaseHandler(object):
    _engine = None

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
                                                             _config.get("DATABASE", "DB_NAME")))
        else:
            raise NotImplemented

        self._engine = engine
        return self._engine

    def get_session(self) -> sqlalchemy.orm.Session:

        return sqlalchemy.orm.Session(self.engine)


database_handler = DatabaseHandler()
