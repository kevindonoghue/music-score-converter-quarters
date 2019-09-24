import numpy as np
import json
import os
from skimage import io
from msc.model.augment import random_augmentation



size = 100


if not os.path.exists('data/preprocessed_small/'):
    os.mkdir('data/preprocessed_small/')

with open('data/quarter_measure_data/other_data.json') as f:
    other_data = json.load(f)

word_to_ix = other_data['lexicon']['word_to_ix']

subsequences = []
image_indices = []
images = []
other_data_small = dict()
other_data_small['lexicon'] = other_data['lexicon']
other_data_small['aux_data'] = []

seq_len = 65

for i in range(size):
    if i % 100 == 0:
        print(i)
    pc = other_data['aux_data'][i]['pc']
    padded_pc = [int(word_to_ix['<PAD>'])]*(seq_len-2) + pc + [int(word_to_ix['<PAD>'])]*(seq_len-2)
    for j in range(len(padded_pc) - seq_len + 1):
        subseq = padded_pc[j:j+seq_len]
        subsequences.append(subseq)
        image_indices.append(i)
    raw_image = io.imread(f'data/quarter_measure_data/{i}.png')/255
    processed_image = (random_augmentation(raw_image, 200, 200)*255).astype(np.uint8)
    images.append(processed_image)

images = np.array(images)
subsequences = np.array(subsequences)
image_indices = np.array(image_indices)
np.save('data/preprocessed_small/quarter_measure_data_images_preprocessed.npy', images)
np.save('data/preprocessed_small/quarter_measure_data_subsequences.npy', subsequences)
np.save('data/preprocessed_small/quarter_measure_data_subsequence_image_indices.npy', image_indices)
