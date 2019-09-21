import numpy as np
from generate_score import generate_score
from xml_to_pc import xml_to_pc
import json


extra_score_data = []
for i in range(10000):
    chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
    chord_probs = chord_probs / chord_probs.sum()
    measure_length = int(4*np.random.choice([3, 4]))
    rest_prob = 0.2
    key_number = int(np.random.randint(-7, 8))
    soup = generate_score(64, measure_length, key_number, rest_prob, chord_probs)
    with open(f'score_data/{i}.musicxml', 'w+') as f:
        f.write(str(soup))
    # xml_to_pc modifies soup in place
    # unfortunately, BeautifulSoup doesn't seem to have a cloning method
    pc = ['<START>'] + xml_to_pc(soup) + ['<END>']
    extra_score_data.append({'measure_length': measure_length, 'key_number': key_number, 'pc': pc})

with open(f'score_data/extra_score_data.json', 'w+') as f:
    json.dump(extra_score_data, f)
