from django.shortcuts import render
import torch
from model import run_model
from model import Net
from .forms import UploadedMeasureForm, UploadedPageForm
from django.http import HttpResponseRedirect
from django.urls import reverse
import json
import os
from handle_page import handle_page
from bs4 import BeautifulSoup



def measure_upload(request):
    if request.method == 'POST':
        form = UploadedMeasureForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            path = 'media/' + str(file.measure)
            key_number = int(str(file.key))
            measure_length = int(str(file.time_signature))
            s = run_model(path, measure_length, key_number)
            with open('media/output.musicxml', 'w+') as f:
                f.write(s)
            return HttpResponseRedirect(reverse('measure_output'))
    elif request.method == 'GET':
        form = UploadedMeasureForm()
    context = {'form': form}
    return render(request, 'measure_upload.html', context)

def measure_output(request):
    with open('media/output.musicxml') as f:
        musicxml = f.read()
    context = {'musicxml': musicxml}
    return render(request, 'measure_output.html', context)
    

def page_upload(request):
    if request.method == 'POST':
        form = UploadedPageForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            path = 'media/' + str(file.page)
            key_number = int(str(file.key))
            measure_length = int(str(file.time_signature))
            handle_page(str(file.page), measure_length, key_number, 'media/current_measures/')
            
            measures = []
            soup = BeautifulSoup(features='xml')
            score_partwise = soup.new_tag('score-partwise', version='3.1')
            part_list = soup.new_tag('part-list')
            score_part = soup.new_tag('score-part', id='P1')
            part_name = soup.new_tag('part-name')
            soup.append(score_partwise)
            score_partwise.append(part_list)
            part_list.append(score_part)
            score_part.append(part_name)
            part_name.append('Piano')
            part = soup.new_tag('part', id='P1')
            score_partwise.append(part)

            os.chdir('..')
            for i in range(len(os.listdir('./media/current_measures/'))):
                measure_soup = run_model(f'./media/current_measures/subimage{i}.png', 16, 0)
                measure = measure_soup.find('measure')
                if i != 0:
                    attributes = measure.find('attributes')
                    attributes.extract()
                measures.append(measure)
                for measure in measures:
                    part.append(measure)

            with open('media/output_musicxml.musicxml', 'w+') as f:
                f.write(str(soup))

            return HttpResponseRedirect('media/output_musicxml.musicxml')
    elif request.method == 'GET':
        form = UploadedPageForm()
    context = {'form': form}
    return render(request, 'page_upload.html', context)