import numpy as np
import json


# gets two numpy arrays: one an array of subsequences (of length 65) and the other the list of image indices corresponding to those subsequences

with open("data/quarter_measure_data/other_data.json") as f:
    other_data = json.load(f)
    aux_data = other_data['aux_data']
    word_to_ix = other_data['lexicon']['word_to_ix']

subsequences = []
image_indices = []

seq_len = 65

for i in range(40000):
    if i % 100 == 0:
        print(i)
    pc = aux_data[i]['pc']
    padded_pc = [int(word_to_ix['<PAD>'])]*(seq_len-2) + pc + [int(word_to_ix['<PAD>'])]*(seq_len-2)
    for j in range(len(padded_pc) - seq_len + 1):
        subseq = padded_pc[j:j+seq_len]
        subsequences.append(subseq)
        image_indices.append(i)

np.save('data/quarter_measure_data_subsequences.npy', subsequences)
np.save('data/quarter_measure_data_subsequence_image_indices.npy', image_indices)
