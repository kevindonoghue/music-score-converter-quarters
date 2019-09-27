from django.db import models

class UploadedMeasure(models.Model):
    measure = models.FileField('Measure')
    key = models.CharField(max_length=64)
    time_signature = models.CharField(max_length=64)


class UploadedPage(models.Model):
    page = models.FileField('Page')
    key = models.CharField(max_length=64)
    time_signature = models.CharField(max_length=64)