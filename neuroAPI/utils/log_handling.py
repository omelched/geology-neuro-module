import errno
import logging
import os
import sys
import traceback

from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")

LOG_FILE = "logs/geologyAPI.log"


def _get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def _get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


logger = logging.getLogger('Logger')
logger.setLevel(logging.DEBUG)

if not os.path.exists(os.path.dirname(LOG_FILE)):
    try:
        os.makedirs(os.path.dirname(LOG_FILE))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
if logger.handlers:
    logger.handlers = []

logger.addHandler(_get_file_handler())
logger.addHandler(_get_console_handler())
logger.propagate = False


def handle_exception(_logger: logging.Logger, e: Exception):
    _logger.error(e)
    _logger.error(traceback.format_exc())
