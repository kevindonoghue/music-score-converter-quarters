import numpy as np
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from skimage import io
from augment import random_augmentation
from model import device

def get_measure_data_channel(measure_length, key_number, height, width):
    # gets an extra channel containing the time and key sig info
    # the image has shape (height, width)
    # this channel will eventually be added onto the (grayscale) image to make an array of shape (2, height, width)
    # for simplicity, assume that height and width are multiplies of 10
    key_number += 7
    vec = np.zeros((2, 20))
    vec[0, key_number] += 1
    if measure_length == 12:
        vec[1, 0] += 1
    elif measure_length == 16:
        vec[1, 1] += 1
    tiled = np.tile(vec, (int(height/2), int(width/20)))
    return tiled


class MeasureDataset(Dataset):
    def __init__(self, path, seq_len, height, width):
        # path is a path to a folder that contains png files named i.png and a json file other_data.json
        # other_data.json contains a dictionary d with d['lex_data'] containing two dictionaries word_to_ix and ix_to_word
        # and d['aux_data'] containing a list of dictionaries containing the pseudocode, measure_length, and key_number data for i.png
        # the pseudocode has already been converted into numerical indices via word_to_ix
        # seq_len is the length of a sequence plugged into the lstm during train time
        # the images are rescaled to shape (height, width)
        self.path = path
        self.seq_len = seq_len
        self.height = height
        self.width = width
        with open(path + 'other_data.json') as f:
            d = json.load(f)
            self.word_to_ix = d['lex_data']['word_to_ix']
            self.ix_to_word = d['lex_data']['ix_to_word']
            self.aux_data = d['aux_data']
        item_lengths = []
        for x in self.aux_data:
            item_lengths.append(len(x['pc']) + seq_len - 1)
        cum_lengths = np.cumsum(item_lengths)
        self.length = cum_lengths[-1]
        self.lower_bounds = np.concatenate([[0], cum_lengths[:-1]])
        self.upper_bounds = cum_lengths
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        image_number = np.argmax((self.lower_bounds <= i)*(self.upper_bounds > i))
        measure_data = self.aux_data[image_number]
        pc = measure_data['pc']
        padded_pc = [self.word_to_ix('<PAD>')]*(self.seq_len-1) + pc + [self.word_to_ix('<PAD>')]*(self.seq_len-1)
        padded_pc = np.array(padded_pc)
        measure_length = measure_data['measure_length']
        key_number = measure_data['key_number']
        measure_data_channel = get_measure_data_channel(measure_length, key_number, self.height, self.width)
        raw_image = io.imread(self.path + f'/{image_number}.png')
        processed_image = random_augmentation(raw_image, self.height, self.width)
        arr = np.concatenate([processed_image, measure_data_channel])
        arr = torch.Tensor(arr).astype(torch.float).to(device)
        start_index = i - self.lower_bounds[image_number]
        seq1 = torch.Tensor(pc[start_index:start_index+self.seq_len]).astype(torch.long).to(device)
        seq2 = torch.Tensor(pc[start_index+1:start_index+self.seq_len+1]).astype(torch.long).to(device)
        pc = torch.Tensor(pc).astype(torch.long).to(device)
        measure_length = torch.Tensor(measure_length).astype(torch.long).to(device)
        key_number = torch.Tensor(key_number).astype(torch.long).to(device)
        return {'arr': arr, 'seq1': seq1, 'seq2': seq2, 'pc': pc, 'measure_length': measure_length, 'key_number': key_number}


def get_data(path, batch_size, seq_len, height, width):
    dataset = MeasureDataset(path, seq_len, height, width)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataset, dataloader