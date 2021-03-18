from flask import Flask
from neuroAPI.utils import config, logger

server = Flask(__name__)
server.config.from_object(config)
logger.info('OK')

from neuroAPI.application.api_handler import api  # noqa
