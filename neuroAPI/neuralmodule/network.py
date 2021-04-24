from torch import nn, optim
from torch.utils.data import DataLoader

from neuroAPI.neuralmodule.dataset import GeologyDataset


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


class TrainingSession(object):
    def __init__(self, dataset: GeologyDataset, model: NeuralNetwork,
                 learning_rate: float = 1e-3, batch_size: int = 64, epochs: int = 5):
        if not dataset or not model:
            raise ValueError  # TODO: elaborate
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = model
        self.training_params = {
            'lr': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)

    def train_loop(self):
        size = len(self.dataloader.dataset)  # noqa
        for i, batch in enumerate(self.dataloader):
            pred = self.model(batch['X'])
            loss = self.loss_fn(pred, batch['Y'])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0 or i == 1:
                loss, current = loss.item(), i * len(batch['X'])
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def train(self):
        for i in range(self.training_params['epochs']):
            print(f"Epoch {i + 1}\n-------------------------------")
            self.train_loop()
