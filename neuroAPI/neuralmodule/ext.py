# TODO: ext to __init__

from uuid import UUID
from typing import Union
import io

import torch

from neuroAPI.database.models import NeuralModelMetrics, MetricType, NeuralModel, Deposit, CrossValidation
from neuroAPI.neuralmodule.metrics import Metric
from neuroAPI.neuralmodule.network import NeuralNetwork as _NeuralNetwork

_METRIC_ID_BUFFER: dict[str, UUID] = {}
_ROCK_ID_BUFFER: dict[UUID, dict[int, UUID]] = {}


class NeuralNetwork(NeuralModel, _NeuralNetwork):
    def __init__(self, output_count: int, deposit: Deposit, block_size: float, max_epochs: int,
                 cross_validation: CrossValidation = None):
        assert type(deposit) == Deposit, TypeError(f'type(`deposit`) == {type(deposit)}. '
                                                   'Expected neuroAPI.database.models.Deposit')
        assert not cross_validation or type(cross_validation) == CrossValidation, \
            TypeError(f'type(`cross_validation`) == {type(cross_validation)}. '
                      'Expected neuroAPI.database.models.CrossValidation')
        try:
            block_size = float(block_size)
        except ValueError:
            raise ValueError('`block_size` is not float-able')
        try:
            max_epochs = int(max_epochs)
        except ValueError:
            raise ValueError('`max_epochs` is not int-able')

        NeuralModel.__init__(self)  # TODO: research about super() and refactor 4 flexibility
        _NeuralNetwork.__init__(self, output_count)  # +

        self.deposit_id = deposit.id
        self.block_size = block_size
        self.max_epochs = max_epochs
        self.cross_validation_id = cross_validation.id

    def save(self):
        buff = io.BytesIO()
        torch.save(self, buff)
        buff.seek(0)
        self.dump = buff.read()
        buff.close()


class PYCMMetric(NeuralModelMetrics, Metric):

    def __init__(self, name: str, metric_type: MetricType, value: Union[float, int, str], epoch: int,
                 neural_model: NeuralNetwork, rock_index: int = None):

        assert type(metric_type) == MetricType, TypeError('`metric_type` is not from `MetricType` enum')
        assert type(value) in [float, int, str], TypeError(f'type(`value`) == {type(value)}. '
                                                           'Expected Union[float, int, str]')
        assert type(neural_model) == _NeuralNetwork, TypeError(f'type(`neural_model`) == {type(neural_model)}. '
                                                               'Expected neuroAPI.neuralmodule.ext.NeuralNetwork')

        NeuralModelMetrics.__init__(self)  # TODO: research about super() and refactor 4 flexibility
        Metric.__init__(self, name=name, value=value)  # +

        self.neural_model_id = neural_model.id
        self.metric_id = self.__get_metric_id(metric_type)
        try:
            self.epoch = int(epoch)
        except ValueError:
            raise ValueError('`epoch` is not int-able')
        if not rock_index:
            self.rock_id = self.__get_rock_id(rock_index, neural_model)
        self.value = self._value

    def __get_metric_id(self, metric_type: MetricType) -> UUID:
        try:
            return _METRIC_ID_BUFFER[self.name]
        except KeyError:
            idx = self._get_create_metric(self.name, metric_type)
            _METRIC_ID_BUFFER[self.name] = idx
            return idx

    def __get_rock_id(self, rock_index: int, neural_model: NeuralNetwork) -> UUID:
        try:
            return _ROCK_ID_BUFFER[neural_model.id][rock_index]
        except KeyError:
            idx = self._get_rock_id(rock_index, neural_model.deposit_id)
            assert type(idx) == UUID, Exception(f'no rock with index {rock_index} '
                                                f'for deposit {neural_model.deposit_id} in database')
            try:
                _ROCK_ID_BUFFER[neural_model.id][rock_index] = idx
            except KeyError:
                _ROCK_ID_BUFFER[neural_model.id] = {}
                _ROCK_ID_BUFFER[neural_model.id][rock_index] = idx

            return idx

    @staticmethod
    def _calculate(pred, true) -> float:
        raise NotImplementedError
