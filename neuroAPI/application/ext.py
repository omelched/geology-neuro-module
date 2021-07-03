from typing import TypedDict
from uuid import UUID
from neuroAPI.utils import LoggedBaseException


class CoordinatesDict(TypedDict):
    x: float
    y: float
    z: float


class ParameterValidationException(LoggedBaseException):
    pass


def validateUUID(value: str) -> UUID:
    try:
        _value = UUID(value)
    except ValueError:
        raise ParameterValidationException(f'UUID: {value}')

    return _value
