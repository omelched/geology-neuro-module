from django.contrib import admin
from django.apps import apps

from ..apps import OmelchedDotDevConfig


for model in apps.get_app_config(OmelchedDotDevConfig.label).models.values():
    admin.site.register(model)
