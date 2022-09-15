from django.urls import include, path
from django.shortcuts import render


def test(request):
    return render(request, "omelcheddotdev/base.html")


urlpatterns = [
    path(r'test/', test),
]
