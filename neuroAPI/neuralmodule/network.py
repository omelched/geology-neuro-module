from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from pycm import ConfusionMatrix

from neuroAPI.neuralmodule.dataset import GeologyDataset
from neuroAPI.neuralmodule.ext import PYCMMetric, _NeuralNetwork
from neuroAPI.database.models import MetricType


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


class TrainingSession(object):
    def __init__(self, dataset: GeologyDataset, model: _NeuralNetwork,
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

    def train(self):
        self._before_training()
        for epoch in range(self.training_params['epochs']):
            self._before_epoch(epoch)
            self.train_loop()
            self._after_epoch(epoch)
        self._after_training()

    def _before_epoch(self, epoch: int):
        pass

    def _after_epoch(self, epoch: int):
        df = self.dataloader.dataset.get_all()  # noqa
        # TODO: refactor with Dataset.get_all()
        cm = ConfusionMatrix(df['Y'].to_numpy(), self.model(Tensor(df['X'])).argmax(dim=1).numpy())
        metrics = [PYCMMetric(name=m,
                              metric_type=MetricType.overall_stat,
                              value=v,
                              epoch=epoch,
                              neural_model=self.model)
                   for m, v in cm.overall_stat.items() if type(v) in(int, str, float)]  # noqa

        pass

    def _before_training(self):
        pass

    def _after_training(self):
        pass
