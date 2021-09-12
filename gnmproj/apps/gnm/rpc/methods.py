from uuid import UUID
from typing import Any

from ..rpc import api
from ..src.utils import check_typing
from ..models import Deposit
from ..tasks import train_network


@api.dispatcher.add_method(name='service.echo')
def echo(request, *args, **kwargs):
    return args, kwargs


@api.dispatcher.add_method(name='train.singular')
@check_typing
def train_neural_network(request, deposit_id: UUID, max_epochs: int, block_size: int) -> Any:
    deposit = Deposit.objects.get(id=deposit_id)

    queryset = deposit.wells

    if not queryset.exists():
        raise queryset.model.DoesNotExist(f'{queryset.model.__name__} matching query does not exist.')

    for well in queryset.prefetch_related('known_blocks').all():

        well_queryset = well.known_blocks.filter(size=block_size)

        if not well_queryset.exists():
            raise well_queryset.model.DoesNotExist(f'{well_queryset.model.__name__} matching query does not exist.')

    task = train_network.delay(deposit.id, max_epochs, block_size)

    return {'task-id': task.id}
