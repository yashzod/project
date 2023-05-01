from django.db import models

# Create your models here.
class CsvFilePath(models.Model):
    path = models.TextField()