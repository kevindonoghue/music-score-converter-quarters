import os
import json


with open('data/extra_data.json') as f:
    old_extra_data = json.load(f)

new_exta_data = []

for i in range(10000):
    png_exists = os.path.exists(f'data/{i}-1.png')
    svg_exists = os.path.exists(f'data/{i}-1.svg')
    if png_exists and svg_exists:
        os.rename(f'data/{i}-1.png', f'completed_data/{i}-1.png')
        os.rename(f'data/{i}-1.svg', f'completed_data/{i}-1.svg')
        new_extra_data.append()
    
    