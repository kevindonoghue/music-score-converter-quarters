import numpy as np
import json
import matplotlib.pyplot as plt


images = np.load('c_major_simple1/images.npy')
measure_lengths = np.load('c_major_simple1/measure_lengths.npy')
key_numbers = np.load('c_major_simple1/key_numbers.npy')
with open('c_major_simple1/pc_data.json') as f:
    pc_data = json.load(f)

print(images.shape)
n = np.random.randint(len(pc_data))
print(measure_lengths[n])
print(key_numbers[n])
print(pc_data[n])
plt.imshow(images[n], cmap='bone')
plt.show()