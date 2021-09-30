from typing import Union

from ...models import Metric
from ...models import Rock, NeuralModel, NeuralModelMetricValues


class MetricValue(object):

    def __init__(self,
                 name: str,
                 metric_type: Metric.MetricTypeEnum,
                 value: Union[float, int, str],
                 neural_model: NeuralModel,
                 epoch: int,
                 rock_index: int = None,
                 ):
        # if bool(value) == bool(objects):  # TODO: check that only `value` or only `object` is passed
        #     if value is None:
        #         raise ValueError('Neither `value` nor `objects` passed')
        #     else:
        #         raise ValueError('`value` and `objects` passed simultaneously')
        assert type(name) == str, TypeError('`name` is not string')
        assert type(metric_type) == Metric.MetricTypeEnum, TypeError(
            '`metric_type` is not from `MetricType` enum')
        assert type(value) in [float, int, str], TypeError(f'type(`value`) == {type(value)}. '
                                                           'Expected Union[float, int, str]')
        assert type(neural_model) == NeuralModel, TypeError(f'type(`neural_model`) == {type(neural_model)}.')

        self.model = NeuralModelMetricValues(
            neural_model=neural_model,
            metric=Metric.objects.get_or_create(name=name, mtype=metric_type)[0],
            epoch=epoch,
            rock=Rock.objects.get(deposit=neural_model.deposit.id, index=rock_index) if rock_index else None,
            value=value,
        )

    def save(self):
        self.model.save()
