import configparser

from neuroAPI.utils.log_handling import logger, handle_exception


class LoggedBaseException(BaseException):
    logger = logger

    def __init__(self, message: str = None):
        super().__init__(message)

        self.logger.error(str(self))

    def __str__(self):
        return str(self.__class__).split('\'')[1]


class Config(configparser.ConfigParser):

    def __init__(self):
        super().__init__()

        self.read('pyCONFIG.cfg')


config = Config()
