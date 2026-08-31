from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("train/", views.train_stream, name="train"),
]
