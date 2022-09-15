import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
import uuid

from .network import NeuralNetwork
from ...models import Rock, PredictedBlock, PredictedBlockOutputs

conn = create_engine('postgresql+psycopg2://omelched:dtkbrjktgbt!@localhost:30000/gnm')


def predict(model: NeuralNetwork):
    qs = model.model.deposit.borders
    deposit_borders = qs.get(point_type='min'), qs.get(point_type='max')
    block_size = model.model.block_size

    _tx = torch.arange(deposit_borders[0].x, deposit_borders[1].x + block_size, block_size)
    _ty = torch.arange(deposit_borders[0].y, deposit_borders[1].y + block_size, block_size)
    _tz = torch.arange(deposit_borders[0].z, deposit_borders[1].z + block_size, block_size)

    n_tx = torch.div(torch.sub(_tx, min(_tx)), (max(_tx) - min(_tx)))
    n_ty = torch.div(torch.sub(_ty, min(_ty)), (max(_ty) - min(_ty)))
    n_tz = torch.div(torch.sub(_tz, min(_tz)), (max(_tz) - min(_tz)))

    tensor = torch.cartesian_prod(n_tx, n_ty, n_tz)

    # tensor = (tensor * 10**2).round() / (10**2)  # round to 0.01

    m = torch.nn.Softmax(dim=1)
    pred = m(model(tensor))
    pred_argmax = pred.argmax(dim=1)

    rocks = Rock.objects.filter(deposit=model.model.deposit)
    index_id_rocks_dict = {rock.index: rock.id for rock in rocks}

    predicted_blocks = []
    predicted_blocks_outputs = []

    coords = pd.DataFrame(tensor.numpy())

    # DENORMALIZE
    pd.options.mode.chained_assignment = None

    for i, column in enumerate(['x', 'y', 'z']):
        _min = np.float64(getattr(deposit_borders[0], column))
        _max = np.float64(getattr(deposit_borders[1], column))
        coords.iloc[:, i] = coords.iloc[:, i] * (_max - _min) + _min

    df = pd.concat([coords, pd.DataFrame(pred_argmax.numpy())], axis=1)
    df = pd.concat([df, pd.DataFrame(pred.detach().numpy())], axis=1)
    # df = df.head(100)
    df.columns = ['x', 'y', 'z', 'content_id'] + [rock_id for rock_id in index_id_rocks_dict.values()]
    df['content_id'] = df['content_id'].map(index_id_rocks_dict)
    df['neural_model_id'] = str(model.model.pk)
    df['id'] = [uuid.uuid4() for _ in range(len(df.index))]
    df['predicted_block_id'] = df['id']
    df[['x', 'y', 'z', 'content_id', 'neural_model_id', 'id']].to_sql(
        'gnm_predictedblock',
        conn,
        if_exists='append',
        index=False,
        chunksize=100000,
        method='multi',
    )
    df = df.drop(labels=['x', 'y', 'z', 'neural_model_id', 'id'], axis=1)
    df = df.melt(id_vars=['predicted_block_id', 'content_id'], var_name='rock_id', value_name='value')
    df = df[df['content_id'] == df['rock_id']]
    df['id'] = [uuid.uuid4() for _ in range(len(df.index))]

    df[['id', 'predicted_block_id', 'rock_id', 'value']].to_sql(
        'gnm_predictedblockoutputs',
        conn,
        if_exists='append',
        index=False,
        chunksize=100000,
        method='multi',
    )
