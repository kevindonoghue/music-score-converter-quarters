from django.forms import ModelForm
from .models import UploadedMeasure, UploadedPage

class UploadedMeasureForm(ModelForm):
    class Meta:
        model = UploadedMeasure
        fields = ['measure', 'key', 'time_signature']

class UploadedPageForm(ModelForm):
    class Meta:
        model = UploadedPage
        fields = ['page', 'key', 'time_signature']