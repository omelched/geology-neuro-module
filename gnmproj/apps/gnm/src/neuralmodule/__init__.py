from uuid import UUID
import sys
import torch
import io


import pandas as pd
from django.db.models import F

from .network import NeuralNetwork
from .dataset import FastDataLoader
from .train import TrainingSession
from .predict import predict as _predict
from ...models import KnownBlock, NeuralModel


_METRIC_ID_BUFFER: dict[str, UUID] = {}  # TODO: buffers to database module?
_ROCK_ID_BUFFER: dict[UUID, dict[int, UUID]] = {}


def train_network(deposit_id: UUID, max_epochs: int, block_size: int, update_state_callback: callable = None):
    data = pd.DataFrame(
        list(
            KnownBlock.objects
            .filter(well__deposit__id=deposit_id, size=block_size)
            .annotate(index=F('content__index'))
            .values('id', 'x', 'y', 'z', 'index')
        )
    )
    nn = NeuralNetwork(
        deposit_id=deposit_id,
        output_count=len(data['index'].unique()),
        block_size=block_size,
        max_epochs=max_epochs,
    )
    ts = TrainingSession(
        FastDataLoader(dataframe=data, shuffle=True),
        nn,
        epochs=max_epochs,
        update_state_callback=update_state_callback
    )

    ts.train()

    return nn.model.id


def predict(neural_model_id: UUID):
    nnmodel = NeuralModel.objects.get(pk=neural_model_id)
    oc = KnownBlock.objects.filter(
        well__deposit__id=nnmodel.deposit_id,
        size=nnmodel.block_size).distinct('content_id').count()
    nn = NeuralNetwork(nnmodel.deposit_id, oc, nnmodel.block_size, max_epochs=0)
    nn.model = nnmodel
    nn.load_state_dict(torch.load(io.BytesIO(nnmodel.dump)))
    _predict(nn)
