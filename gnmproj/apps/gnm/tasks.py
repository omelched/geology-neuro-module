from contextlib import contextmanager

from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache

from .src.utils import check_typing
from .src.neuralmodule import train_network

logger = get_task_logger(__name__)


@contextmanager
def acquire_lock(lock, *args, **kwargs):
    status = cache.add(lock, True)
    try:
        yield status
    finally:
        if status:
            cache.delete(lock)


@shared_task
@check_typing
def train_network(deposit_id: str, max_epochs: int, block_size: int):
    lock = {
        'deposit_id': deposit_id,
        max_epochs: max_epochs,
        block_size: int
    }
    logger.debug('Task started')

    with acquire_lock(lock) as acquired:
        if acquired:

            neural_model_id = train_network(deposit_id, max_epochs, block_size)

            logger.debug('Task finished')
            return neural_model_id

    logger.debug('Task is already in process by another worker')
