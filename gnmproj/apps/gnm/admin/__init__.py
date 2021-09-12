from django.contrib import admin
from django.apps import apps

from ..apps import GnmConfig


for model in apps.get_app_config(GnmConfig.label).models.values():
    admin.site.register(model)
