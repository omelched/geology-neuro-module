import inspect
import typing
from functools import wraps

from django.contrib.auth import authenticate
from django.http.response import HttpResponseForbidden


def check_typing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        parameters = {name: p for name, p in inspect.signature(func).parameters.items()}

        try:
            for k, v in kwargs.items():

                if not isinstance(v, parameters[k].annotation) \
                        and not parameters[k].annotation is parameters[k].empty \
                        and not isinstance(parameters[k].annotation, typing._SpecialGenericAlias):
                    try:
                        kwargs[k] = parameters[k].annotation(v)
                    except (ValueError, TypeError):
                        raise
        except KeyError:
            pass

        return func(*args, **kwargs)

    return wrapper


def requires_jwt(func):
    @wraps(func)
    def wrapper(request, *args, **kwargs):

        jwt = kwargs.pop('jwt')

        if not authenticate(request=request, jwt=jwt, **kwargs):
            return HttpResponseForbidden()

        return func(request, *args, **kwargs)

    return wrapper
