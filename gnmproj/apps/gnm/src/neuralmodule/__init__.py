from uuid import UUID
import sys
import torch
import io

import pandas as pd
from django.db.models import F

from gnmproj.apps.gnm.src.neuralmodule.network import NeuralNetwork
from gnmproj.apps.gnm.src.neuralmodule.dataset import FastDataLoader
from .train import TrainingSession
from .predict import predict as _predict
from ...models import KnownBlock, NeuralModel, Well, CrossValidation, DepositBorders

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
    if data.empty:
        raise Exception('NoData')
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


def crooss_validation(deposit_id: UUID, max_epochs: int, block_size: int, ):
    wells = Well.objects.filter(deposit__pk=deposit_id)
    cross_validation_obj = CrossValidation.objects.create(
        name=f'CV for {deposit_id}'
    )
    for well in wells:

        data = pd.DataFrame(
            list(
                KnownBlock.objects
                .filter(well__deposit__id=deposit_id, size=block_size)
                .exclude(well=well)
                .annotate(index=F('content__index'))
                .values('id', 'x', 'y', 'z', 'index')
            )
        )
        predict_data = pd.DataFrame(
            list(
                KnownBlock.objects
                .filter(well=well)
                .annotate(index=F('content__index'))
                .values('id', 'x', 'y', 'z', 'index')
            )
        )
        if data.empty:
            raise Exception('NoData')
        nn = NeuralNetwork(
            deposit_id=deposit_id,
            output_count=len(data['index'].unique()),
            block_size=block_size,
            max_epochs=max_epochs,
            cross_validation_id=cross_validation_obj.pk,
            excluded_wells=[well.pk],
        )

        bs = well.deposit.borders.all()

        borders = {
            'X_x': {border.point_type: border.x for border in bs},
            'X_y': {border.point_type: border.y for border in bs},
            'X_z': {border.point_type: border.z for border in bs},
        }
        train_dataloader = FastDataLoader(dataframe=data, shuffle=True, borders=borders)
        predict_dataloader = FastDataLoader(dataframe=predict_data, borders=borders)

        ts = TrainingSession(
            train_dataloader,
            nn,
            epochs=max_epochs,
            custom_predict_dataloader=predict_dataloader
        )

        ts.train()
