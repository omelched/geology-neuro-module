from contextlib import contextmanager
import uuid
import sys

from celery import shared_task, states
from celery.utils.log import get_task_logger
from celery.exceptions import Ignore
from django.core.cache import cache

from .src import check_typing
from .src.neuralmodule import train_network as _train_network

logger = get_task_logger(__name__)


@contextmanager
def acquire_lock(lock, *args, **kwargs):
    status = cache.add(lock, True)
    try:
        yield status
    finally:
        if status:
            cache.delete(lock)


@shared_task(bind=True)
@check_typing
def train_network(self, deposit_id: uuid.UUID, max_epochs: int, block_size: int):

    def callback(perc: str):
        self.update_state(state='TRAINING', meta={'progress': perc})

    lock = {
        'deposit_id': deposit_id,
        max_epochs: max_epochs,
        block_size: block_size
    }

    with acquire_lock(lock) as acquired:
        if acquired:
            old_outs = sys.stdout, sys.stderr
            rlevel = self.app.conf.worker_redirect_stdouts_level

            try:
                self.app.log.redirect_stdouts_to_logger(logger, rlevel)
                neural_model_id = _train_network(deposit_id, max_epochs, block_size, update_state_callback=callback)
            finally:
                sys.stdout, sys.stderr = old_outs

            return str(neural_model_id)

    self.update_state(
        state=states.FAILURE,
        meta=Exception('Training in progress')
    )

    raise Ignore()
