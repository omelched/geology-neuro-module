import io
import uuid

import torch
from torch import nn

from ...models import NeuralModel


class NeuralNetwork(nn.Module):
    def __init__(self,
                 deposit_id: uuid.UUID,
                 output_count: int,
                 block_size: float,
                 max_epochs: int,
                 cross_validation_id: uuid.UUID = None):
        super(NeuralNetwork, self).__init__()

        try:
            block_size = float(block_size)
        except ValueError:
            raise ValueError('`block_size` is not float-able')
        try:
            max_epochs = int(max_epochs)
        except ValueError:
            raise ValueError('`max_epochs` is not int-able')

        if not type(output_count) == int or output_count < 1:
            raise Exception(f'Failed to init: <output_count: {output_count} {type(output_count)}>')

        self.model = NeuralModel(
            deposit_id=deposit_id,
            block_size=block_size,
            max_epochs=max_epochs,
            cross_validation_id=cross_validation_id,
            dump=None,
        )

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

    def save(self):

        buff = io.BytesIO()
        torch.save(self, buff)
        buff.seek(0)
        self.model.dump = buff.read()
        buff.close()

        self.model.save()
