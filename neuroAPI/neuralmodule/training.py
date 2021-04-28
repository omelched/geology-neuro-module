
# TODO: split into 2 parents and child classes
from datetime import datetime

import pandas as pd
from pycm import ConfusionMatrix
from torch import nn, optim

from neuroAPI.database import database_handler
from neuroAPI.database.models import MetricType, PredictedBlock, Rock
from neuroAPI.neuralmodule.dataset import FastDataLoader
from neuroAPI.neuralmodule.ext import NeuralNetwork as _NeuralNetwork  # noqa
from neuroAPI.neuralmodule.ext import PYCMMetric  # noqa


class TrainingSession(object):
    def __init__(self, dataloader: FastDataLoader, model: _NeuralNetwork,
                 learning_rate: float = 1e-3, batch_size: int = 64, epochs: int = 5):
        if not dataloader or not model:
            raise ValueError  # TODO: elaborate
        self.dataloader = dataloader
        self.model = model
        self.training_params = {
            'lr': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_loop(self):
        for i, batch in enumerate(self.dataloader):
            pred = self.model(batch[0])
            loss = self.loss_fn(pred, batch[1])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        self._before_training()
        for epoch in range(self.training_params['epochs']):
            self._before_epoch(epoch + 1)
            self.train_loop()
            self._after_epoch(epoch + 1)
        self._after_training()

    def _before_epoch(self, epoch: int):
        print(f'start epoch {epoch} — {datetime.now()}')
        pass

    def _after_epoch(self, epoch: int):
        m = nn.Softmax(dim=1)  # TODO: refactor as method
        cm = ConfusionMatrix(self.dataloader.tensors[1].numpy(),
                             m(self.model(self.dataloader.tensors[0])).argmax(dim=1).numpy())
        metrics = [PYCMMetric(name=m,
                              metric_type=MetricType.overall_stat,
                              value=v,
                              epoch=epoch,
                              neural_model=self.model)
                   for m, v in cm.overall_stat.items() if type(v) in (int, str, float)]  # noqa
        session = database_handler.active_session  # TODO: refactor to generator
        session.add_all(metrics)
        print(f'end epoch {epoch} — {datetime.now()}')

        pass

    def _before_training(self):
        print(f'start training — {datetime.now()}')

        pass

    def _after_training(self):
        self.model.save()
        session = database_handler.active_session  # TODO: refactor to generator
        session.add(self.model)
        m = nn.Softmax(dim=1)  # TODO: refactor as method
        pred = m(self.model(self.dataloader.tensors[0])).argmax(dim=1)  # TODO: refactor as neural network method
        rocks = Rock.get_rocks_in_deposit(self.model.deposit)
        index_id_rocks_dict = {rock[0].index: rock[0].id for rock in rocks}

        coords = pd.DataFrame(self.dataloader.tensors[0].numpy())
        coords = self.dataloader.denormalize(coords)
        predicted_blocks = [PredictedBlock(neural_model_id=self.model.id,
                                           center_x=round(coords.iloc[i, 0].item(), 3),
                                           center_y=round(coords.iloc[i, 1].item(), 3),
                                           center_z=round(coords.iloc[i, 2].item(), 3),
                                           content=index_id_rocks_dict[pred[i].item()]) for i in range(len(pred))]
        session.add_all(predicted_blocks)
        print(f'end training — {datetime.now()}')

