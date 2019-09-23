import numpy as np
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from skimage import io
from .augment import random_augmentation


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
        # other_data.json contains a dictionary d with d['lexicon'] containing two dictionaries word_to_ix and ix_to_word
        # and d['aux_data'] containing a list of dictionaries containing the pseudocode, measure_length, and key_number data for i.png
        # the pseudocode has already been converted into numerical indices via word_to_ix
        # seq_len is the length of a sequence plugged into the lstm during train time
        # the images are rescaled to shape (height, width)
        # device is either 'cpu' or 'cuda'
        self.path = path
        self.seq_len = seq_len
        self.height = height
        self.width = width
        with open(path + 'other_data.json') as f:
            d = json.load(f)
            self.word_to_ix = d['lexicon']['word_to_ix']
            self.ix_to_word = d['lexicon']['ix_to_word']
            self.aux_data = d['aux_data']
        item_lengths = []
        for x in self.aux_data:
            item_lengths.append(len(x['pc']) + seq_len - 2) # seq_len-1 for appended padding and -1 because you'll want two sequences offset by 1
        cum_lengths = np.cumsum(item_lengths)
        self.length = cum_lengths[-1]
        self.lower_bounds = np.concatenate([[0], cum_lengths[:-1]])
        self.upper_bounds = cum_lengths
        self.raw_images = []
        self.image_indices = []
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        image_number = np.argmax((self.lower_bounds <= i)*(self.upper_bounds > i))
        measure_data = self.aux_data[image_number]
        pc = measure_data['pc']
        padded_pc = [self.word_to_ix['<PAD>']]*(self.seq_len-1) + pc + [self.word_to_ix['<PAD>']]*(self.seq_len-1)
        padded_pc = np.array(padded_pc)
        measure_length = measure_data['measure_length']
        key_number = measure_data['key_number']
        measure_data_channel = get_measure_data_channel(measure_length, key_number, self.height, self.width)
        raw_image = io.imread(self.path + f'/{image_number}.png')/255
        processed_image = random_augmentation(raw_image, self.height, self.width)
        arr = np.array([processed_image, measure_data_channel])
        arr = torch.Tensor(arr).type(torch.float)
        start_index = i - self.lower_bounds[image_number]
        seq1 = torch.Tensor(padded_pc[start_index:start_index+self.seq_len]).type(torch.long)
        seq2 = torch.Tensor(padded_pc[start_index+1:start_index+self.seq_len+1]).type(torch.long)
        image_number = torch.Tensor([image_number]).type(torch.long)
        return {'arr': arr, 'seq1': seq1, 'seq2': seq2, 'image_number': image_number}


class MeasureDatasetPreprocessed(Dataset):
    def __init__(self, images, other_data, seq_len):
        # images is a numpy array containing the images, and other data is as for MeasureDataset
        self.images = images
        self.seq_len = seq_len
        self.word_to_ix = other_data['lexicon']['word_to_ix']
        self.ix_to_word = other_data['lexicon']['ix_to_word']
        self.aux_data = other_data['aux_data']
        item_lengths = []
        for x in self.aux_data:
            item_lengths.append(len(x['pc']) + seq_len - 2) # seq_len-1 for appended padding and -1 because you'll want two sequences offset by 1
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
        padded_pc = [self.word_to_ix['<PAD>']]*(self.seq_len-1) + pc + [self.word_to_ix['<PAD>']]*(self.seq_len-1)
        padded_pc = np.array(padded_pc)
        measure_length = measure_data['measure_length']
        key_number = measure_data['key_number']
        measure_data_channel = get_measure_data_channel(measure_length, key_number, self.images.shape[1], self.images.shape[2])
        image = self.images[image_number]/255
        arr = np.array([image, measure_data_channel])
        arr = torch.Tensor(arr).type(torch.float)
        start_index = i - self.lower_bounds[image_number]
        seq1 = torch.Tensor(padded_pc[start_index:start_index+self.seq_len]).type(torch.long)
        seq2 = torch.Tensor(padded_pc[start_index+1:start_index+self.seq_len+1]).type(torch.long)
        image_number = torch.Tensor([image_number]).type(torch.long)
        return {'arr': arr, 'seq1': seq1, 'seq2': seq2, 'image_number': image_number}


class MeasureDatasetRaw(Dataset):
    def __init__(self, path, num_images, seq_len, height, width):
        self.path = path
        self.num_images = num_images
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.raw_images = None
        self.image_indices = None
        with open(path + 'other_data.json') as f:
            d = json.load(f)
            self.word_to_ix = d['lexicon']['word_to_ix']
            self.ix_to_word = d['lexicon']['ix_to_word']
            self.aux_data = d['aux_data']
        self.length = None
        self.lower_bounds = None
        self.upper_bounds = None

    def get_images(self):
        self.image_indices = np.random.choice(40000, size=self.num_images)
        self.raw_images = []
        for i in self.image_indices:
            self.raw_images.append(io.imread(self.path + f'/{i}.png'))
        item_lengths = []
        for i in self.image_indices:
            item_lengths.append(len(self.aux_data[i]['pc']) + self.seq_len - 2) # seq_len-1 for appended padding and -1 because you'll want two sequences offset by 1
        cum_lengths = np.cumsum(item_lengths)
        self.length = cum_lengths[-1]
        self.lower_bounds = np.concatenate([[0], cum_lengths[:-1]])
        self.upper_bounds = cum_lengths
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        int_image_number = np.argmax((self.lower_bounds <= i)*(self.upper_bounds > i))
        ext_image_number = self.image_indices[int_image_number]
        measure_data = self.aux_data[ext_image_number]
        pc = measure_data['pc']
        padded_pc = [self.word_to_ix['<PAD>']]*(self.seq_len-1) + pc + [self.word_to_ix['<PAD>']]*(self.seq_len-1)
        padded_pc = np.array(padded_pc)
        measure_length = measure_data['measure_length']
        key_number = measure_data['key_number']
        measure_data_channel = get_measure_data_channel(measure_length, key_number, self.height, self.width)
        raw_image = np.array(self.raw_images[int_image_number])/255
        processed_image = random_augmentation(raw_image, self.height, self.width)
        arr = np.array([processed_image, measure_data_channel])
        arr = torch.Tensor(arr).type(torch.float)
        start_index = i - self.lower_bounds[int_image_number]
        seq1 = torch.Tensor(padded_pc[start_index:start_index+self.seq_len]).type(torch.long)
        seq2 = torch.Tensor(padded_pc[start_index+1:start_index+self.seq_len+1]).type(torch.long)
        ext_image_number = torch.Tensor([ext_image_number]).type(torch.long)
        return {'arr': arr, 'seq1': seq1, 'seq2': seq2, 'ext_image_number': ext_image_number}




def get_data(path, batch_size, seq_len, height, width, num_workers=4):
    # produces a measure dataset (arguments same as in that class) and its dataloader (with batch size batch_size)
    dataset = MeasureDataset(path, seq_len, height, width)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, dataloader