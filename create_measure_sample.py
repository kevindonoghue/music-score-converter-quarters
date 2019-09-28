import subprocess
import numpy as np
from msc.generation.generate_measure import generate_measure
from msc.generation.crop import crop
from msc.generation.pc_to_xml import pc_to_xml
from msc.generation.xml_to_pc import xml_to_pc
from msc.generation.generate_bboxes import get_bboxes
from msc.model.augment import random_augmentation
import os
import json
import time


sample_size = 10000
output_dir = 'c_major_simple1/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if not os.path.exists(output_dir + 'temp/'):
    os.mkdir(output_dir + 'temp/')



pc_data = dict()
pc_data = []


chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
chord_probs = chord_probs / chord_probs.sum()
rest_prob = 0.2

images = []
measure_lengths = []
key_numbers = []

t = time.time()
for i in range(sample_size):
    measure_length = np.random.choice([12, 16])
    key_number = 0
    measure_lengths.append(measure_length)
    key_numbers.append(key_number)
    soup = generate_measure(measure_length, key_number, rest_prob, chord_probs)
    with open(output_dir + f'temp/{i}.musicxml', 'w+') as f:
        f.write(str(soup))
    pc = xml_to_pc(soup)
    pc = ['<START>'] + pc + ['<END>']
    pc_data.append(pc)

np.save(output_dir + 'measure_lengths', measure_lengths)
np.save(output_dir + 'key_numbers', key_numbers)

with open(output_dir + 'pc_data.json', 'w+') as f:
    json.dump(pc_data, f)



batch_json = []
for i in range(sample_size):
    batch_json.append({
        'in': output_dir + f'temp/{i}.musicxml',
        'out': [output_dir + f'temp/{i}.png', output_dir + f'temp/{i}.svg']
    })

with open(output_dir + 'temp/batch.json', 'w+') as f:
    json.dump(batch_json, f)

subprocess.call(['MuseScore3.exe', '-j', output_dir + 'temp/batch.json'])

for i in range(sample_size):
    png_path = output_dir + f'temp/{i}-1.png'
    svg_path = output_dir + f'temp/{i}-1.svg'
    image = crop(png_path, svg_path)
    image = random_augmentation(image, 224, 224)
    image = (image*255).astype(np.uint8)
    images.append(image)

np.save(output_dir + 'images', images)

for filename in os.listdir(output_dir + 'temp/'):
    os.remove(output_dir + 'temp/' + filename)
os.rmdir(output_dir + 'temp/')