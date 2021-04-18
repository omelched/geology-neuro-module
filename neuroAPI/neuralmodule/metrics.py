import abc
from uuid import uuid4, UUID
from typing import Union

import torch
import pandas as pd

import neuroAPI.database.models
import neuroAPI.neuralmodule.network


class BaseMetric(metaclass=abc.ABCMeta):
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
    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> Union[float, list[list[float]]]:
        """
        Must be implemented in subclasses

        :param true: Tensor with true data.
        :param pred: Tensor with predicted data.

        :return: Metric value.
        :rtype: float or list[list[float]] (for confusion matrix)
        """
        raise NotImplementedError

    def calculate_result(self, true: torch.Tensor, pred: torch.Tensor):
        if not isinstance(true, torch.Tensor):
            raise ValueError
        if not isinstance(pred, torch.Tensor):
            raise ValueError

        return self._calculate(true, pred)


_METRIC_ID_CACHE: dict[str, UUID] = {}  # to buffer metric.name -> metric.id mapping


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


class CategoricalAccuracy(DatabaseMetricMixin):
    _name = 'CategoricalAccuracy'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(CategoricalAccuracy, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        return (sum(torch.argmax(pred, dim=1) == torch.argmax(true, dim=1)) / len(pred)).item()


class BinaryAccuracy(DatabaseMetricMixin):
    _name = 'BinaryAccuracy'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(BinaryAccuracy, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        b_concat = torch.stack((torch.flatten(true), torch.flatten(pred))).T > 0.5
        return len(b_concat[b_concat[:, 0] == b_concat[:, 1]]) / len(b_concat)


class CategoricalCrossentopy(DatabaseMetricMixin):
    _name = 'CategoricalCrossentopy'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(CategoricalCrossentopy, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        concat = torch.stack((torch.flatten(true), torch.flatten(pred))).T
        return (-concat[:, 1].clamp(min=0.000001, max=0.999999).log() * concat[:, 0]).mean().item()


class BinaryCrossentropy(DatabaseMetricMixin):
    _name = 'BinaryCrossentropy'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(BinaryCrossentropy, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        concat = torch.stack((torch.flatten(true), torch.clamp(torch.flatten(pred), min=0.000001, max=0.999999))).T
        return torch.mean(torch.Tensor([-1 * torch.log(row[1]) if row[0] == 1
                                        else -1 * torch.log(1 - row[1]) for row in concat])).item()


class MeanSquaredError(DatabaseMetricMixin):
    _name = 'MeanSquaredError'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(MeanSquaredError, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        concat = torch.stack((torch.flatten(true), torch.flatten(pred))).T
        return torch.div(torch.sum(torch.square(torch.subtract(concat[:, 0], concat[:, 1]))), len(concat)).item()


class Recall(DatabaseMetricMixin):
    _name = 'Recall'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(Recall, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        b_concat = torch.stack((torch.flatten(true), torch.flatten(pred))).T > 0.5
        return len(b_concat[(b_concat[:, 0] == True) & (b_concat[:, 1] == True)]) / (
                len((b_concat[(b_concat[:, 0] == True) & (b_concat[:, 1] == True)]))
                + len(b_concat[(b_concat[:, 0] == True) & (b_concat[:, 1] == False)]))


class Precision(DatabaseMetricMixin):
    _name = 'Precision'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(Precision, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        b_concat = torch.stack((torch.flatten(true), torch.flatten(pred))).T > 0.5
        return len(b_concat[(b_concat[:, 0] == True) & (b_concat[:, 1] == True)]) / (
                len((b_concat[(b_concat[:, 0] == True) & (b_concat[:, 1] == True)]))
                + len(b_concat[(b_concat[:, 0] == False) & (b_concat[:, 1] == True)]))


class CosineSimilarity(DatabaseMetricMixin):
    _name = 'CosineSimilarity'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(CosineSimilarity, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        return torch.nn.CosineSimilarity(dim=0)(torch.flatten(true).T, torch.flatten(pred).T).item()  # noqa


class SymmetricMeanAbsolutePersentageError(DatabaseMetricMixin):
    _name = 'SymmetricMeanAbsolutePersentageError'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(SymmetricMeanAbsolutePersentageError, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        return torch.nn.CosineSimilarity(dim=0)(torch.flatten(true).T, torch.flatten(pred).T).item()  # noqa


class ConfusionMatrix(DatabaseMetricMixin):
    _name = 'ConfusionMatrix'

    def __init__(self, neural_network: neuroAPI.neuralmodule.network.NeuralNetwork, epoch: int,
                 true: torch.Tensor, pred: torch.Tensor, name: str = None):
        super(ConfusionMatrix, self).__init__(neural_network, epoch, true, pred, name)

    def _calculate(self, true: torch.Tensor, pred: torch.Tensor) -> list:

        return pd.crosstab(torch.argmax(true, dim=1), torch.argmax(pred, dim=1)).values.tolist()
