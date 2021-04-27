from typing import Union
from torch import Tensor


class Metric(object):

    _value: Union[float, int, str]
    _name: str

    def __init__(self, name: str, value: Union[float, int, str] = None, objects: tuple[Tensor, Tensor] = None):
        # if bool(value) == bool(objects):  # TODO: check that only `value` or only `object` is passed
        #     if value is None:
        #         raise ValueError('Neither `value` nor `objects` passed')
        #     else:
        #         raise ValueError('`value` and `objects` passed simultaneously')
        assert type(name) == str, TypeError('`name` is not string')

        self._name = name
        if objects:
            self._value = self.calculate(objects)
        else:
            self._value = value

    def calculate(self, objects: tuple[Tensor, Tensor]) -> float:

        assert objects, ValueError('No `objects` passed')
        assert len(objects) == 2, ValueError(f'{len(objects)} items in `objects` passed. 2 items expected')
        assert type(objects[0]) == Tensor, TypeError(f'{type(objects[0])} passed. torch.Tensor expected')
        assert type(objects[1]) == Tensor, TypeError(f'{type(objects[1])} passed. torch.Tensor expected')
        assert objects[0].size()[0] == objects[1].size()[0], ValueError('Tensors length dont match.'
                                                                        f'{objects[0].size()[0]} !='
                                                                        f'{objects[1].size()[0]}')
        return self._calculate(objects[0], objects[1])

    @staticmethod
    def _calculate(pred: Tensor, true: Tensor) -> float:
        raise NotImplementedError
