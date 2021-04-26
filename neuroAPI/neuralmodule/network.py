from datetime import datetime

from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from pycm import ConfusionMatrix
import numpy as np

from neuroAPI.neuralmodule.dataset import GeologyDataset
from neuroAPI.database.models import MetricType
from neuroAPI.database import database_handler


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
            nn.Linear(32, output_count),
            nn.Softmax(dim=1)  # TODO: evaluate and resolve warning
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

