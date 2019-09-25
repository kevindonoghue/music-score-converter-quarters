import numpy as np
from bs4 import BeautifulSoup
from skimage import io, transform
from skimage.color import gray2rgb
from msc.generation.generate_bboxes import get_bboxes
import os
from msc.model.augment import random_score_augmentation

# prepare pngs of score files for input into yolo

def produce_annotation_file(bboxes):
    # bboxes is a list of sublists of coordinates for the bboxes
    # each entry of the sublist is a tuple ((x1, y1), (x2, y2))
    # this produces an annotation file with rows
    # 0 x_center y_center width height
    # each row corresponds to one bounding box
    # returns the result as a string
    s = ''
    for sublist in bboxes:
        for box in sublist:
            (x1, y1), (x2, y2) = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            s += f'0 {x_center} {y_center} {width} {height}\n'
    return s

def resize(source_path, target_path):
    image = io.imread(source_path)/255
    image = 1-image[:, :, 3]
    image = gray2rgb(image)
    image_resized = (transform.resize(image, (416, 416))*255).astype(np.uint8)
    io.imsave(target_path, image_resized)


source_dir = 'data/score_data/'
target_dir = 'data/score_data_distorted_yolo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

if not os.path.exists(target_dir + 'images/'):
    os.mkdir(target_dir + 'images/')

if not os.path.exists(target_dir + 'labels/'):
    os.mkdir(target_dir + 'labels/')


for filename in os.listdir(source_dir):
    if filename[-4:] == '.png' and filename not in os.listdir(target_dir + 'images/'):
        svg_path = source_dir + filename[:-4] + '.svg'
        bboxes = get_bboxes(svg_path)
        with open(target_dir + 'labels/' + filename[:-4] + '.txt', 'w+') as f:
            f.write(produce_annotation_file(bboxes))
        image = io.imread(source_dir + filename)/255
        image = 1-image[:, :, 3]
        image = random_score_augmentation(image, 416, 416)
        image = gray2rgb(image)
        image = (image*255).astype(np.uint8)
        io.imsave(target_dir + 'images/' + filename, image)

        resize(source_dir + filename, target_dir + 'images/' + filename)
