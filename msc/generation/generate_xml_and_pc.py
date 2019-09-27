import numpy as np
from xml_to_pc import xml_to_pc
from generate_measure import generate_measure
import json
import os


# generates random musicxml files and their corresponding pseudocode
# the ith musicxml file is called i.musicxml and its pseudocode, measure length, and key sig are stored in a dictionary in the ith entry of a list



word_to_ix = {'6': 0, '<START>': 1, '12': 2, '7': 3, 'rest': 4, 'pitch': 5, '3': 6, 'E': 7, '}': 8, 'quarter': 9, '2': 10, '16': 11, '4': 12, 'C': 13, 'measure': 14, '<END>': 15, '0': 16, 'staff': 17, 'duration': 18, 'G': 19, 'chord': 20, 'D': 21, '5': 22, 'F': 23, 'B': 24, 'note': 25, 'backup': 26, 'type': 27, 'A': 28, '-1': 29, '1': 30, '<PAD>': 31}
ix_to_word = {'0': '6', '1': '<START>', '2': '12', '3': '7', '4': 'rest', '5': 'pitch', '6': '3', '7': 'E', '8': '}', '9': 'quarter', '10': '2', '11': '16', '12': '4', '13': 'C', '14': 'measure', '15': '<END>', '16': '0', '17': 'staff', '18': 'duration', '19': 'G', '20': 'chord', '21': 'D', '22': '5', '23': 'F', '24': 'B', '25': 'note', '26': 'backup', '27': 'type', '28': 'A', '29': '-1', '30': '1', '31': '<PAD>'}

other_data = dict()
other_data['lexicon'] = dict()
other_data['lexicon']['word_to_ix'] = word_to_ix
other_data['lexicon']['ix_to_word'] = ix_to_word
other_data['aux_data'] = []

if not os.path.exists(f'measure_data_quarters'):
    os.mkdir(f'measure_data_quarters')
for i in range(30000):
    chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
    chord_probs = chord_probs / chord_probs.sum()
    measure_length = int(4*np.random.choice([3, 4]))
    rest_prob = 0.2
    # key_number = int(np.random.randint(-7, 8))
    key_number = 0
    soup = generate_measure(measure_length, key_number, rest_prob, chord_probs)
    with open(f'measure_data_quarters/{i}.musicxml', 'w+') as f:
        f.write(str(soup))
    # xml_to_pc modifies soup in place
    # unfortunately, BeautifulSoup doesn't seem to have a cloning method
    pc = ['<START>'] + xml_to_pc(soup) + ['<END>']
    other_data['aux_data'].append({'measure_length': measure_length, 'key_number': key_number, 'pc': pc})
    
with open(f'measure_data_quarters/other_data.json', 'w+') as f:
    json.dump(other_data, f)