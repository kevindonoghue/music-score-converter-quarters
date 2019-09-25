from django.shortcuts import render
import torch
from model import run_model
from model import Net
from .forms import UploadedFileForm




def upload(request):
    if request.method == 'POST':
        form = UploadedFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            path = 'media/' + str(file.measure)
            key_number = int(str(file.key))
            measure_length = int(str(file.time_signature))
            print(run_model(path, measure_length, key_number))
    elif request.method == 'GET':
        form = UploadedFileForm()
    context = {'form': form}
    return render(request, 'upload.html', context)
    