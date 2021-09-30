from uuid import UUID
import sys


import pandas as pd
from django.db.models import F

from .metrics import Metric
from .network import NeuralNetwork
from .dataset import FastDataLoader
from .training import TrainingSession
from ...models import KnownBlock


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


