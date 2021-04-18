import abc
from uuid import uuid4, UUID
from typing import Union
from pycm import ConfusionMatrix
import torch
import pandas as pd

import neuroAPI.database.models
import neuroAPI.neuralmodule.network

_METRIC_ID_CACHE: dict[str, UUID] = {}  # to buffer metric.name -> metric.id mapping


class BaseMetric(metaclass=abc.ABCMeta):
    """
    Abstract class for metrics.

    All metrics MUST inherit this class and redefine _calculate() method.
    """
    _name = f'metric{uuid4()}'
    _value = None

    def __init__(self, true: torch.Tensor, pred: torch.Tensor, name: str = None):
        if name:
            if isinstance(name, str):
                self._name = name
            else:
                raise ValueError('BaseMetric:__init__:Name!')

        self._value = self.calculate_result(true, pred)

    @property
    def get_value(self):
        return self._value

    @property
    def get_name(self):
        return self._name

    @abc.abstractmethod
    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> Union[float, list[list[int]]]:
        """
        MUST be implemented in non-abstract subclasses.

        :param true: Tensor with true data.
        :param pred: Tensor with predicted data.

        :return: Metric value.
        :rtype: float or list[list[int]] (for confusion matrix)
        """
        raise NotImplementedError

    def calculate_result(self, true: torch.Tensor, pred: torch.Tensor):
        if not isinstance(true, torch.Tensor):
            raise ValueError
        if not isinstance(pred, torch.Tensor):
            raise ValueError

        return self._calculate(true, pred)

c = ConfusionMatrix()
c.save_obj()

class DatabaseMetricMixin(BaseMetric, neuroAPI.database.models.NeuralModelMetrics, metaclass=abc.ABCMeta):

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 session=None, *args, **kwargs):
        if not isinstance(neural_network, neuroAPI.neuralmodule.network.NeuralNetwork):
            raise TypeError  # TODO: log exception

        epoch = int(epoch)  # TODO: catch exception

        super().__init__(*args, **kwargs)

        try:
            metric_id = _METRIC_ID_CACHE[self.get_name]  # TODO: checksum/hash of table for sanity?
        except KeyError:
            metric_id = self.get_create_metric(name=self.get_name, session=session)
            _METRIC_ID_CACHE[self.get_name] = metric_id

        self.neural_model_id = neural_network.id
        self.metric_id = metric_id
        self.epoch = epoch
        self.value = self.get_value

#
# class CategoricalAccuracy(DatabaseMetricMixin):
#     _name = 'CategoricalAccuracy'
#
#     def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
#                  true: torch.Tensor, pred: torch.Tensor, name: str = None):
#         super(CategoricalAccuracy, self).__init__(neural_network, epoch, true, pred, name)
#
#     def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
#         return (sum(torch.argmax(pred, dim=1) == torch.argmax(true, dim=1)) / len(pred)).item()
#
