import numpy as np
import json
from skimage import io
from msc.model.augment import random_augmentation

images = []
for i in range(40000):
    if i % 100 == 0:
        print(i)
    raw_image = io.imread(f'data/quarter_measure_data/{i}.png')/255
    processed_image = (random_augmentation(raw_image, 200, 200)*255).astype(np.uint8)
    images.append(processed_image)
images = np.array(images)
np.save('data/quarter_measure_data_images_preprocessed.npy', images)