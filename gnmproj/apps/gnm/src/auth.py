from datetime import datetime
import pytz

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.conf import settings
from jwt import decode, DecodeError


class JWTModelBackend(ModelBackend):

    def authenticate(self, request, jwt=None, **kwargs):

        if jwt is None:
            # no JWT passed
            return

        try:
            payload = decode(jwt, settings.SECRET_KEY, algorithms=['HS256'])
        except DecodeError:
            # JWT is not from server
            return

        if not all([key in payload for key in ['username', 'iat', 'exp']]):
            # JWT malformed
            return

        if datetime.fromtimestamp(payload['exp']) < datetime.now():
            # JWT expired
            return

        user_model = get_user_model()

        try:
            user = user_model.objects.get_by_natural_key(username=payload['username'])
        except user_model.DoesNotExist:
            # user has been deleted since issuance
            return

        if pytz.timezone(settings.TIME_ZONE).localize(datetime.fromtimestamp(payload['iat'])) \
                < user.gnm_user.jwt_issuance_time:
            # JWT `iat` too old
            return

        if self.user_can_authenticate(user):
            # OK
            request.user = user
            return user

        # else user is not active

