from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.conf import settings
from jwt import decode, DecodeError


class JWTModelBackend(ModelBackend):

    def authenticate(self, request, jwt=None, **kwargs):
        if jwt is None:
            return

        try:
            payload = decode(jwt, settings.SECRET_KEY, algorithms=['HS256'])
        except DecodeError:
            return

        user_model = get_user_model()

        try:
            user = user_model.objects.get_by_natural_key(username=payload['username'])
        except user_model.DoesNotExist:
            return

        if self.user_can_authenticate(user):
            request.user = user
            return user

