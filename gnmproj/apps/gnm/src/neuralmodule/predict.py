import torch
import pandas as pd
import numpy as np

from .network import NeuralNetwork
from ...models import Rock, PredictedBlock, NeuralModelMetricValues, PredictedBlockOutputs
from ...models import Metric as MetricModel


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

    for i, row in enumerate(pred_argmax):
        predicted_block = PredictedBlock(
            neural_model=model.model,
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
    
