from django.db import models

class UploadedFile(models.Model):
    measure = models.FileField('Measure')
    key = models.CharField(max_length=64)
    time_signature = models.CharField(max_length=64)