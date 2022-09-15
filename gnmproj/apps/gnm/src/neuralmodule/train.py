import pandas as pd
from pycm import ConfusionMatrix
from torch import nn, optim

from .dataset import FastDataLoader
from gnmproj.apps.gnm.models import Rock, PredictedBlock, NeuralModelMetricValues, PredictedBlockOutputs
from ...models import Metric as MetricModel
from .network import NeuralNetwork
from .metrics import MetricValue


class TrainingSession(object):
    def __init__(self,
                 dataloader: FastDataLoader,
                 model: NeuralNetwork,
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 5,
                 update_state_callback: callable = None,
                 custom_predict_dataloader: FastDataLoader = None,
                 ):
        if not dataloader or not model:
            raise ValueError  # TODO: elaborate
        self.dataloader = dataloader
        self.model = model
        self.training_params = {
            'lr': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
        self.update_state_callback = update_state_callback

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.custom_predict_dataloader = custom_predict_dataloader

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
        print(f'starting epoch {epoch}', end='\n')
        ...

    def _after_epoch(self, epoch: int):
        print(f'finished epoch {epoch}, saving metrics', end='\n')
        m = nn.Softmax(dim=1)  # TODO: refactor as method
        cm = ConfusionMatrix(self.dataloader.data[1].numpy(),
                             m(self.model(self.dataloader.data[0])).argmax(dim=1).numpy())

        metrics = [MetricValue(name=m,
                               metric_type=MetricModel.MetricTypeEnum.OVERALL,
                               value=v,
                               epoch=epoch,
                               neural_model=self.model.model)
                   for m, v in cm.overall_stat.items() if type(v) in (int, str, float)]

        NeuralModelMetricValues.objects.bulk_create([metric.model for metric in metrics])

        if self.update_state_callback:
            self.update_state_callback(f"{round(epoch / self.training_params['epochs'], 3)} %")

    def _before_training(self):
        self.model.save()
        print('Training started', end='\n')

    def _after_training(self):
        self.model.save()
        print('Training finished, predicting', end='\n')
        m = nn.Softmax(dim=1)  # TODO: refactor as method
        if self.custom_predict_dataloader:
            dataloader = self.custom_predict_dataloader
        else:
            dataloader = self.dataloader
        pred = m(self.model(dataloader.data[0]))
        pred_argmax = pred.argmax(dim=1)  # TODO: refactor as neural network method
        rocks = Rock.objects.filter(deposit=self.model.model.deposit)
        index_id_rocks_dict = {rock.index: rock.id for rock in rocks}

        coords = pd.DataFrame(dataloader.data[0].numpy())
        coords = dataloader.denormalize(coords)

        predicted_blocks = []
        predicted_blocks_outputs = []

        for i, row in enumerate(pred_argmax):
            predicted_block = PredictedBlock(
                neural_model=self.model.model,
                x=round(coords.iloc[i, 0].item(), 3),
                y=round(coords.iloc[i, 1].item(), 3),
                z=round(coords.iloc[i, 2].item(), 3),
                content_id=index_id_rocks_dict[row.item()]
            )
            predicted_blocks.append(predicted_block)
            for ii, rock in enumerate(rocks):
                predicted_block_output = PredictedBlockOutputs(
                    predicted_block=predicted_block,
                    rock=rock,
                    value=pred[i][ii].item(),
                )
                predicted_blocks_outputs.append(predicted_block_output)

        PredictedBlock.objects.bulk_create(predicted_blocks)
        PredictedBlockOutputs.objects.bulk_create(predicted_blocks_outputs)

        print('Predicting finished', end='\n')
