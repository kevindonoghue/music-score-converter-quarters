import numpy as np
from bs4 import BeautifulSoup
from PIL import Image

def crop(png_path, svg_path):
    # crops the image in png_path using the data in svg_path
    # crops the image so that the clef, key, and time signatures are cropped out
    with open(svg_path) as f:
        soup = BeautifulSoup(f, 'xml')
    image = Image.open(png_path)
    
    svg_tag = soup.find('svg')
    width = float(svg_tag['width'].strip('px'))
    height = float(svg_tag['height'].strip('px'))

    barline = [x for x in soup.find_all('polyline') if x['class'] == 'BarLine'][1]
    right = float(barline['points'].split(',')[0])
    time_sigs = [x for x in soup.find_all('path') if x['class'] == 'TimeSig']
    def get_coords(s):
        # s is a string like M425.006,618.084
        # returns a pair of floats (425.006, 618.084)
        x = s.split()
        x = [y.strip('MCL') for y in x]
        x = np.array(list(map(lambda y: list(map(float, y.split(','))), x)))
        return x

    all_coords = np.concatenate([get_coords(time_sigs[0]['d']), get_coords(time_sigs[1]['d'])], axis=0)
    left = np.max(all_coords[:, 0])
    
    left_frac = left/width
    right_frac = right/width
    
    im_width, im_height = image.size
    image = image.crop((left_frac*im_width, 0, right_frac*im_width, im_height))
    image = image.crop(image.getbbox())
    image_np = np.array(image)[:, :-3, 3]
    image_np = 1-image_np/255
    return(image_np)
