from mongoengine import connect
import traceback

from neuroAPI.utils import handle_exception, logger, config
from neuroAPI.database.models import Deposit

try:
    connect(config['DEFAULT']['MONGODB_NAME'],
            host=config['DEFAULT']['MONGODB_HOST'],
            port=int(config['DEFAULT']['MONGODB_PORT']))
except KeyError as e:
    handle_exception(logger, traceback.format_exc())
    raise Exception('Incorrect pyCONFIG.cfg')
