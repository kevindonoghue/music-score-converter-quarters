import numpy as np
import json
from skimage import io
from msc.model.augment import random_augmentation
from msc.generation.crop import crop
import os
from skimage import io


height = 299
width = 299

source_dir = 'c_major_dataset_raw/'
target_dir = 'c_major_dataset/'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

if not os.path.exists(target_dir + 'images/'):
    os.mkdir(target_dir + 'images/')

with open(source_dir + 'other_data.json') as f:
    other_data = json.load(f)

pc_data = dict()
pc_data['lexicon'] = other_data['lexicon']
word_to_ix = pc_data['lexicon']['word_to_ix']
pc_data['pc'] = []
key_numbers = []
measure_lengths = []
augmented_images = []


def get_time_signature_layer(measure_length, height, width):
    # measure length is 12 or 16
    x = np.zeros((height, width)).astype(np.uint8)
    if measure_length == 12:
        x[:int(height/2)] += 255
    if measure_length == 16:
        x[int(height/2):] += 255
    return x

def get_key_signature_layer(key_number, height, width):
    # key number is between -7 and 7 inclusive
    x = np.zeros((height, width)).astype(np.uint8)
    splits = np.array_split(x, 15)
    splits[key_number+7] += 255
    return x

j = 0
for i in range(30000):
    if i % 100 == 0:
        print(i)
    png_path = source_dir + f'{i}-1.png'
    svg_path = source_dir + f'{i}-1.svg'
    if os.path.exists(png_path) and os.path.exists(svg_path):
        image_np = crop(png_path, svg_path)
        augmented_image = (random_augmentation(image_np, height, width)*255).astype(np.uint8)
        measure_length = other_data['aux_data'][i]['measure_length']
        key_number = other_data['aux_data'][i]['key_number']
        # time_signature_layer = get_time_signature_layer(measure_length, height, width)
        # key_signature_layer = get_key_signature_layer(key_number, height, width)
        # arr = np.array([augmented_image, time_signature_layer, key_signature_layer])
        augmented_images.append(augmented_image)
        image_np = (image_np*255).astype(np.uint8)
        io.imsave(target_dir + 'images/' + f'{j}.png', image_np)
        pc = other_data['aux_data'][i]['pc']
        pc_as_num = [word_to_ix[x] for x in pc]
        pc_data['pc'].append(pc_as_num)
        j += 1

augmented_images = np.array(augmented_images)
np.save(target_dir + 'augmented_images.npy', augmented_images)

with open(target_dir + 'pc_data.json', 'w+') as f:
    json.dump(pc_data, f)

        