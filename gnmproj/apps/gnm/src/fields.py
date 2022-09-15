from uuid import uuid4

from django.db.models.fields import AutoFieldMixin, UUIDField


class UUIDAutoField(AutoFieldMixin, UUIDField):

    def get_internal_type(self):
        return 'UUIDField'

    def rel_db_type(self, connection):
        return UUIDField().db_type(connection=connection)

    def get_default(self):
        return uuid4()

