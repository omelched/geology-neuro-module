import inspect
import typing


def check_typing(func):
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
