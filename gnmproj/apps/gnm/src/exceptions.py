from types import new_class

from django.conf import settings

try:
    SentryWillIgnoreMe = settings.SENTRY_WILL_IGNORE_ME
except AttributeError:
    SentryWillIgnoreMe = new_class('SentryWillIgnoreMe', bases=(Exception,))


class _GNMException(Exception):
    pass


class _IgnorableGMNException(_GNMException, SentryWillIgnoreMe):
    pass


class InvalidCredentials(_IgnorableGMNException):
    pass


class DoesNotExist(_IgnorableGMNException):
    pass


def generate_DNE(does_not_exist_exception):
    return new_class('DoesNotExist', bases=(does_not_exist_exception, SentryWillIgnoreMe))
