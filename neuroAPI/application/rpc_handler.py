from typing import Mapping
import uuid

from jsonrpc.backend.flask import api

from neuroAPI.application import server
from neuroAPI.application.ext import CoordinatesDict

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
    :param neural_model_id: UUID of neural model to predict with
    :type neural_model_id: str (format: UUID)

    :return: Predicted dictionary (keys = rocks UUID, values = probability)
    :rtype: Mapping[str, float]
    """
    return {str(uuid.uuid4()): 0.1,
            str(uuid.uuid4()): 0.3,
            str(uuid.uuid4()): 0.6}
