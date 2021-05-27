from typing import Mapping
import uuid

from jsonrpc.backend.flask import api

from neuroAPI.application import server
from neuroAPI.application.ext import CoordinatesDict

from neuroAPI.database import database_handler
from neuroAPI.database.models import Deposit

server.register_blueprint(api.as_blueprint())


@api.dispatcher.add_method(name='service.echo')
def echo(*args, **kwargs):
    """
    Echoes you back.

    :param args: args to echo back
    :type args: Set[]
    :param kwargs: key word args to echo back
    :type kwargs: Mapping[]
    :return: [args, {{kwargs}}]
    :rtype: Set[Set[], Mapping[]]
    """

    return args, kwargs


@api.dispatcher.add_method(name='predict.block.coordinates')
def predict_block_by_coords(coordinates: CoordinatesDict, neural_model_id: str) -> Mapping[str, float]:
    """
    Predicts block by coordinates.

    :param coordinates: Dict with "x", "y", "z" float values
    :type coordinates: CoordinatesDict
    :param neural_model_id: UUID of neural model to predict with
    :type neural_model_id: str

    :return: Predicted dictionary (keys = rocks UUID, values = probability)
    :rtype: Mapping[str, float]
    """
    return {str(uuid.uuid4()): 0.1,
            str(uuid.uuid4()): 0.3,
            str(uuid.uuid4()): 0.6}


@api.dispatcher.add_method(name='predict.block.id')
def predict_block_by_id(block_id: str, neural_model_id: str) -> Mapping[str, float]:
    """
    Predicts block by known block.

    :param block_id: UUID of known block to predict with
    :type block_id: str (format: UUID)
    :param neural_model_id: UUID of neural model to predict with (if not specified will train on non-CV NN found on KnownBlock.well_id.deposit_id)
    :type neural_model_id: str (format: UUID)

    :return: Predicted dictionary (keys = rocks UUID, values = probability)
    :rtype: Mapping[str, float]
    """
    return {str(uuid.uuid4()): 0.1,
            str(uuid.uuid4()): 0.3,
            str(uuid.uuid4()): 0.6}


@api.dispatcher.add_method(name='train.singular')
def train_neural_network(deposit_id: str, max_epoch: int, block_size: float) -> str:
    """
    Starts training.

    :param deposit_id: UUID of deposit to train on
    :type deposit_id: str (format: UUID)
    :param max_epoch: max epoch number to train
    :type max_epoch: int
    :param block_size: size of blocks to create (if not found in DB)
    :type block_size: float

    :return: Returns UUID of NN, which WILL BE created in DB in case of succesfull training
    :rtype: str
    """

    with database_handler.get_session() as session:
        deposit = database_handler.get_object_by_id(Deposit, uuid.UUID(deposit_id))
        df

    return str(uuid.uuid4())


@api.dispatcher.add_method(name='train.cross_validation')
def crossvalidate_neural_network(deposit_id: str, max_epoch: int, block_size: float) -> str:
    """
    Starts cross-validation training.

    :param deposit_id: UUID of deposit to train on
    :type deposit_id: str (format: UUID)
    :param max_epoch: max epoch number to train
    :type max_epoch: int
    :param block_size: size of blocks to create (if not found in DB)
    :type block_size: float

    :return: Returns UUID of cross-validation object, which WILL BE created in DB in case of succesfull training
    :rtype: str
    """
    return str(uuid.uuid4())
