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

sample_size = 10

lexicon = ['6', '<START>', '12', '7', 'rest', 'pitch', '3', 'E', '}', 'quarter', '2', '16', '4', 'C', 'measure', '<END>', '0', 'staff', 'duration', 'G', 'chord', 'D', '5', 'F', 'B', 'note', 'backup', 'type', 'A', '-1', '1', '<PAD>']
word_to_ix = {word: ix for ix, word in enumerate(lexicon)}
ix_to_word = {ix: word for ix, word in enumerate(lexicon)}

pc_data = dict()
pc_data['lexicon'] = dict()
pc['lexicon']['word_to_ix'] = word_to_ix
pc['lexicon']['ix_to_word'] = ix_to_word
pc_data['pc'] = []

output_dir = 'c_major_quarters_balanced/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
chord_probs = chord_probs / chord_probs.sum()
rest_prob = 0.2

images = []
measure_lengths = []
key_numbers = []

for i in range(sample_size):
    measure_length = np.random.choice([12, 16])
    key_number = 0
    soup = generate_measure(measure_length, key_number, rest_prob, chord_probs)
    pc = xml_to_pc(soup)
    pc = ['<START>'] + pc + ['<END>']
    pc = [word_to_ix[word] for word in pc]
    subprocess.call(['MuseScore3.exe', f'temp.musicxml', '-o', f'temp.png'])
    subprocess.call(['MuseScore3.exe', f'temp.musicxml', '-o', f'temp.svg'])
    image = crop('temp.png', 'temp.svg')
    image = random_augmentation(image, 224, 224)
    image = (image*255).astype(np.uint8)
    images.append(image)
    measure_lengths.append(measure_length)
    key_numbers.append(key_number)

np.save(output_dir + 'images', images)
np.save(output_dir + 'measure_lengths', measure_lengths)
np.save(output_dir + 'key_numbers', key_numbers)

with open(output_dir + 'pc_data.json') as f:
    json.dump(pc_data, f)