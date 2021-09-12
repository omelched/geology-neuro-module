import io
from typing import Union
from uuid import UUID

import pandas as pd
import torch
from django.db.models import F

from .metrics import Metric
from .network import NeuralNetwork as _NeuralNetwork
from .dataset import FastDataLoader
from ...models import KnownBlock, NeuralModel, Deposit, CrossValidation, NeuralModelMetricValues, Rock
from ...models import Metric as MetricModel

_METRIC_ID_BUFFER: dict[str, UUID] = {}  # TODO: buffers to database module?
_ROCK_ID_BUFFER: dict[UUID, dict[int, UUID]] = {}


class NeuralNetwork(NeuralModel, _NeuralNetwork):
    def __init__(self,
                 output_count: int,
                 deposit: Deposit,
                 block_size: float,
                 max_epochs: int,
                 cross_validation: CrossValidation = None,
                 *args, **kwargs):

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

        NeuralModel.__init__(self,
                             deposit=deposit,
                             block_size=block_size,
                             max_epochs=max_epochs,
                             cross_validation=cross_validation,
                             dump=None,
                             )  # TODO: research about super() and refactor 4 flexibility
        _NeuralNetwork.__init__(self, output_count)  # -//-

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):

        buff = io.BytesIO()
        torch.save(self, buff)
        buff.seek(0)
        self.dump = buff.read()
        buff.close()

        super(NeuralNetwork, self).save(force_insert=force_insert,
                                        force_update=force_update,
                                        using=using,
                                        update_fields=update_fields)


class PYCMMetricValue(NeuralModelMetricValues, Metric):

    def __init__(
            self,
            name: str,
            metric_type: MetricModel.MetricTypeEnum,
            value: Union[float, int, str],
            epoch: int,
            neural_model: NeuralNetwork,
            rock_index: int = None,
            *args, **kwargs
    ):
        assert type(metric_type) == MetricModel.MetricTypeEnum, TypeError('`metric_type` is not from `MetricType` enum')
        assert type(value) in [float, int, str], TypeError(f'type(`value`) == {type(value)}. '
                                                           'Expected Union[float, int, str]')
        assert type(neural_model) == NeuralNetwork, TypeError(f'type(`neural_model`) == {type(neural_model)}. '
                                                              'Expected neuroAPI.neuralmodule.ext.NeuralNetwork')

        Metric.__init__(self, name=name, value=value)  # +
        NeuralModelMetricValues.__init__(
            self,
            neural_model=neural_model,
            metric=MetricModel.objects.get_or_create(name=name, mtype=metric_type),
            epoch=epoch,
            rock=Rock.objects.get(deposit=neural_model.deposit.id, index=rock_index),
            value=self._value
        )

    @staticmethod
    def _calculate(pred, true) -> float:
        raise NotImplementedError


from .training import TrainingSession

def train_network(deposit_id: UUID, max_epochs: int, block_size: int):
    data = pd.DataFrame(
        list(
            KnownBlock.objects
                .filter(well__deposit__id=deposit_id, size=block_size)
                .annotate(index=F('content__index'))
                .values('id', 'x', 'y', 'z', 'index')
        )
    )
    nn = NeuralNetwork(
        len(data['index'].unique()),
        Deposit.objects.get(id=deposit_id),
        block_size=block_size,
        max_epochs=max_epochs,
        cross_validation=None
    )
    ts = TrainingSession(FastDataLoader(dataframe=data, shuffle=True), nn, epochs=max_epochs)

    return nn.id
