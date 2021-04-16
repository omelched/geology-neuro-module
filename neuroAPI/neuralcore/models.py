import torch
from torch import nn
from uuid import uuid4


class NeuralNetwork(nn.Module):
    def __init__(self, output_count: int):
        super(NeuralNetwork, self).__init__()

        if not type(output_count) == int or output_count < 1:
            raise Exception(f'Failed to init: <output_count: {output_count} {type(output_count)}>')
        self.linear_stack = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_count)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


class BaseMetric(object):
    name = ''

    def __init__(self, name: str = None):
        if not name:
            self.name = f'metric{uuid4()}'

        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError('BaseMetric:__init__:Name!')

    def _calculate(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def calculate_result(self, x: torch.Tensor, y: torch.Tensor):
        return self._calculate(x, y)


class CategoricalAccuracy(BaseMetric):

    def __init__(self, name: str = None):
        if not name:
            name = 'CategoricalAccuracy'

        super(CategoricalAccuracy, self).__init__(name)

    def _calculate(self, x: torch.Tensor, y: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError
        if not isinstance(y, torch.Tensor):
            raise ValueError

        return (sum(torch.argmax(y, dim=1) == torch.argmax(x, dim=1)) / len(y)).item()


class BinaryCrossentropy(BaseMetric):

    def __init__(self, name: str = None):
        if not name:
            name = 'BinaryCrossentropy'

        super(BinaryCrossentropy, self).__init__(name)

    def _calculate(self, x: torch.Tensor, y: torch.Tensor):
        concat = torch.stack((torch.flatten(x), torch.clamp(torch.flatten(y), min=0.000001, max=0.999999))).T
        return torch.mean(torch.Tensor([-1 * torch.log(row[1]) if row[0] == 1
                                        else -1 * torch.log(1 - row[1]) for row in concat])).item()
