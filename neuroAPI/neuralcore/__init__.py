import torch
from torch import nn

from neuroAPI.neuralcore.models import NeuralNetwork
model = NeuralNetwork(3)

learning_rate = 1e-3
batch_size = 64
epochs = 1000

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
