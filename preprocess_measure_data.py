import numpy as np
from msc.model.data import get_measure_data_channel
import json


with open('data/quarter_measure_data/other_data.json') as f:
    other_data = json.load(f)

tiles = []
for i in range(40000):
    if i % 100 == 0:
        print(i)
    measure_length = int(other_data['aux_data'][i]['measure_length'])
    key_number = int(other_data['aux_data'][i]['key_number'])
    tile = get_measure_data_channel(measure_length, key_number, 200, 200)
    tiles.append(tile)

tiles = np.array(tiles)
np.save('data/preprocessed/quarter_measure_data_time_and_key_tiles.npy', tiles)