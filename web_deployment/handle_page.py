import os
import subprocess
from skimage import io


def handle_page(filename, measure_length, key_number, dir):
    if os.path.exists('media/current_measures/'):
        for file in os.scandir('media/current_measures/'):
            os.remove(file)
    else:
        os.mkdir('media/current_measures/')
    path = 'media/' + filename
    os.chdir('./yolov3/')
    subprocess.call(['python', 'detect.py', '--source', '../'+ path, '--cfg', 'cfg/score_yolo.cfg', '--weights', 'weights/last-2019-09-26-3.pt', '--output', '../media/yolo_output/'])
    image = io.imread(f'../media/{filename}')
    image_with_boxes = io.imread(f'../media/yolo_output/{filename}')
    height, width, _ = image.shape
    with open(f'../media/yolo_output/{filename}.txt') as f:
        bbox_string = f.read()
    lines = bbox_string.split('\n')[:-1]
    coordinates = [tuple(map(int, x.split()[:4])) for x in lines if float(x.split()[5])>0.9]
    coordinates = sorted(coordinates, key=lambda tup: tup[1])
    rows = []
    rows.append([coordinates[0]])
    for x in coordinates[1:]:
        last = rows[-1][-1]
        height = last[3] - last[1]
        if x[1] > last[1] + height/2:
            rows.append([x])
        else:
            rows[-1].append(x)
    sorted_coordinates = []
    for row in rows:
        row.sort(key=lambda t: t[0])
        sorted_coordinates.extend(row)
    print(sorted_coordinates)
    for i, (x1, y1, x2, y2) in enumerate(sorted_coordinates):
        subimage = image[y1:y2, x1:x2]
        print(subimage.shape)
        target_dir = '../media/current_measures/'
        io.imsave(target_dir + f'subimage{i}.png', subimage)



# handle_page('c_major_piece3.png', 16, 0, None)