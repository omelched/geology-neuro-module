from django.urls import include, path
from .rpc import api


urlpatterns = [
    path(r'jsonrpc/', include(api.urls)),
]
