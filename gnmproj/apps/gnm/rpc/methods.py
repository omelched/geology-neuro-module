from uuid import UUID
from typing import Any
from datetime import datetime, timedelta

from celery.result import AsyncResult
from django.contrib.auth import authenticate
from django.conf import settings
import jwt

from ..rpc import api
from ..src import check_typing, requires_jwt, InvalidCredentials, generate_DNE, TaskDoesNotExist, predict
from ..models import Deposit
from ..tasks import train_network


@api.dispatcher.add_method(name='service.echo')
@requires_jwt
def echo(request, *args, **kwargs):
    return args, kwargs


@api.dispatcher.add_method(name='auth.login')
def login(request, username: str, password: str) -> Any:
    user = authenticate(username=username, password=password)
    if not user:
        raise InvalidCredentials
    now = datetime.now().replace(microsecond=0)

    payload = {
        'username': user.username,
        'iat': int(now.timestamp()),
        'exp': int((now + timedelta(days=14)).timestamp())
    }

    user.gnm_user.jwt_issuance_time = now
    user.gnm_user.save()

    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


@api.dispatcher.add_method(name='train.singular')
@requires_jwt
@check_typing
def train_neural_network(request, deposit_id: UUID, max_epochs: int, block_size: int) -> Any:
    deposit = Deposit.objects.get(id=deposit_id)

    well_queryset = deposit.wells

    if not well_queryset.exists():
        raise generate_DNE(well_queryset.model.DoesNotExist)(f'{well_queryset.model.__name__} matching query '
                                                             f'does not exist.')

    for well in well_queryset.prefetch_related('known_blocks').all():

        kb_queryset = well.known_blocks.filter(size=block_size)

        if not kb_queryset.exists():
            raise generate_DNE(kb_queryset.model.DoesNotExist)(f'{kb_queryset.model.__name__} matching query '
                                                               f'does not exist.')

    task = train_network.delay(deposit.id, max_epochs, block_size)

    return {'task-id': task.id}


@api.dispatcher.add_method(name='predict')
@requires_jwt
@check_typing
def predict(request, neural_model__id: UUID) -> Any:
    predict(neural_model__id)

    return {'ok': True}


@api.dispatcher.add_method(name='train.get_result')
@requires_jwt
@check_typing
def get_result(request, task_id: UUID) -> Any:
    info = None
    aresult = AsyncResult(str(task_id))
    if aresult.state == 'PENDING':
        raise TaskDoesNotExist()
    if aresult.state == 'TRAINING':
        info = aresult.info['progress'] if (info and 'progress' in info) else None
    if aresult.ready():
        try:
            result = aresult.get()
        except Exception as e:
            result = str(e)
    else:
        result = None

    return {'task-id': aresult.task_id, 'state': aresult.state, 'result': result, 'info': info}

