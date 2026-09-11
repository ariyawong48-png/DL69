from django.urls import path
from . import views

urlpatterns = [
    path('wk11/train/', views.wk11_train, name='wk11_train'),
    path('wk11/train/stream/', views.wk11_train_stream, name='wk11_train_stream'),
    path('wk11/load/', views.wk11_load, name='wk11_load'),
]