from django.contrib import admin
from django.urls import path

from .views import UploadFileView, TrainModels, AttributesView

urlpatterns = [
    path('uploadfile', UploadFileView.as_view()),
    path('train',TrainModels.as_view()),
    path('att',AttributesView.as_view())
]