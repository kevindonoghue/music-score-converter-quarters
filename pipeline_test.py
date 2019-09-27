import subprocess
import numpy as np
from msc.generation.generate_score import generate_score
from msc.generation.crop import crop
from msc.generation.pc_to_xml import pc_to_xml
from msc.generation.generate_bboxes import get_bboxes
import os
import json


key_number = 0
measure_length = 16
rest_prob = 0.2
chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
chord_probs = chord_probs / chord_probs.sum()

soup = generate_score(64, measure_length, key_number, rest_prob, chord_probs)
pc = pc_to_xml

filename = 'filename'

with open(f'{filename}.musicxml', 'w+') as f:
    f.write(str(soup))


MS = 'MuseScore3.exe'

subprocess.call([MS, f'{filename}.musicxml', '-o', f'{filename}.png'])
subprocess.call([MS, f'{filename}.musicxml', '-o', f'{filename}.svg'])

i = 1
while os.path.exists(f'{filename}-{i}.musicxml'):
    bboxes = get_bboxes(f'{filename}-{i}.svg')
    data = dict()
    data['pc'] = pc
    data['bboxes'] = bboxes
    with open(f'{filename}-{i}_data.json', 'w+') as f:
        json.dump(data, f)
    i += 1