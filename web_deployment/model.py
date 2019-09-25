import numpy as np
import json
import torch
import torch.nn as nn
import os
import time
from PIL import Image
from pc_to_xml import pc_to_xml
from skimage import io, transform
from skimage.color import rgb2gray

device = 'cuda'


lexicon = ['6', '<START>', '12', '7', 'rest', 'pitch', '3', 'E', '}', 'quarter', '2', '16', '4', 'C', 'measure', '<END>', '0', 'staff', 'duration', 'G', 'chord', 'D', '5', 'F', 'B', 'note', 'backup', 'type', 'A', '-1', '1', '<PAD>']
word_to_ix = {word: ix for ix, word in enumerate(lexicon)}
ix_to_word = {ix: word for ix, word in enumerate(lexicon)}
len_lexicon = len(lexicon)

batch_size = 128
seq_len = 64
lstm_hidden_size = 256
fc1_output_size = 256

class ConvSubunit(nn.Module):
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size, filter_size, stride=stride, padding=padding)
        self.dp = nn.Dropout2d(p=dropout)
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()
        self.sequential = nn.Sequential(self.conv, self.dp, self.bn, self.relu)

    def forward(self, x):
        return self.sequential(x)

class ConvUnit(nn.Module):
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.subunit1 = ConvSubunit(input_size, output_size, filter_size, 1, padding, dropout)
        self.subunit2 = ConvSubunit(output_size, output_size, filter_size, stride, padding, dropout)
        
    def forward(self, x):
        x = self.subunit1(x)
        x = self.subunit2(x)
        return x

class Net(nn.Module):
    def __init__(self, len_lexicon, lstm_hidden_size, fc1_output_size, device):
        super().__init__()
        self.len_lexicon = len_lexicon
        self.lstm_hidden_size = lstm_hidden_size
        self.fc1_output_size = fc1_output_size
        self.num_iterations = 0
        self.train_time = 0
        self.cnn = nn.Sequential(ConvUnit(2, 64, 3, 2, 1, 0.1), # (200, 200) --> (100, 100)
                                 ConvUnit(64, 128, 3, 2, 1, 0.1), # (100, 100) --> (50, 50)
                                 ConvUnit(128, 128, 3, 5, 1, 0.1), # (50, 50) --> (10, 10)
                                 ConvUnit(128, 128, 3, 5, 1, 0.1)) # (10, 10) --> (2, 2)
        self.fc1 = nn.Linear(512, self.fc1_output_size)
        self.embed = nn.Embedding(num_embeddings=self.len_lexicon, embedding_dim=5)
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.25)
        self.lstm2 = nn.LSTM(input_size=self.fc1_output_size+self.lstm_hidden_size, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.25)
        self.fc2 = nn.Linear(self.lstm_hidden_size, self.len_lexicon)
        
    def forward(self, image_input, language_input, internal1=None, internal2=None):
        bs = image_input.shape[0]
        sl = language_input.shape[1]
        if internal1:
            h1, c1 = internal1
        else:
            h1 = torch.zeros(2, bs, self.lstm_hidden_size).to(device)
            c1 = torch.zeros(2, bs, self.lstm_hidden_size).to(device)
        if internal2:
            h2, c2 = internal2
        else:
            h2 = torch.zeros(2, bs, self.lstm_hidden_size).to(device)
            c2 = torch.zeros(2, bs, self.lstm_hidden_size).to(device)
        image_output = self.fc1(self.cnn(image_input).view(bs, 512))
        image_output = image_output.repeat(1, sl).view(bs, sl, self.fc1_output_size)
        language_output, (h1, c1) = self.lstm1(self.embed(language_input), (h1, c1))
        concatenated = torch.cat([image_output, language_output], 2)
        lstm2_out, (h2, c2) = self.lstm2(concatenated, (h2, c2))
        out = self.fc2(lstm2_out)
        return out, (h1, c1), (h2, c2)
    
    # def fit(self, total_iterations, optimizer, loss_fn, rate_decay, print_every=100):
    #     train_start_time = time.time()
    #     for i in range(total_iterations):
    #         batch_indices = np.random.choice(len(subsequences), size=batch_size)
    #         image_batch = images[image_indices[batch_indices]].reshape(-1, 1, 200, 200)/255
    #         measure_data_channel_batch = measure_data_channels[image_indices[batch_indices]].reshape(-1, 1, 200, 200)
    #         arr = np.concatenate([image_batch, measure_data_channel_batch], axis=1)
    #         sequence_batch = subsequences[batch_indices]
    #         arr = torch.Tensor(arr).type(torch.float).to(device)
    #         seq1 = torch.Tensor(sequence_batch[:, :-1]).type(torch.long).to(device)
    #         seq2 = torch.Tensor(sequence_batch[:, 1:]).type(torch.long).to(device)
    #         out, _, _ = self.forward(arr, seq1)
    #         out = out.view(-1, len_lexicon)
    #         targets = seq2.view(-1)
    #         loss = loss_fn(out, targets)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         if i % print_every == 0:
    #             n = np.random.randint(len(subsequences))
    #             image_index = image_indices[n]
    #             image = images[image_index].reshape(1, 1, 200, 200)/255
    #             measure_data_channel = measure_data_channels[image_index].reshape(1, 1, 200, 200)
    #             arr = np.concatenate([image, measure_data_channel], axis=1)
    #             arr = torch.Tensor(arr).type(torch.float).to(device)
    #             pc = other_data['aux_data'][image_index]['pc']
    #             pc = ' '.join([ix_to_word[str(ix)] for ix in pc])
    #             pred_seq = self.predict(arr)
    #             pred_seq = ' '.join(pred_seq)
    #             with open('storage/preprocessed/log_preprocessed_model-2019-09-23.txt', 'a+') as f:
    #                 info_string = f"""
    #                 ----
    #                 iteration: {i}
    #                 time elapsed: {time.time() - train_start_time}
    #                 loss: {loss}
    #                 ----
    #                 pred: {pred_seq}
    #                 ----
    #                 true: {pc}
    #                 ----



    #                 """.replace('    ', '')
    #                 print(info_string)
    #                 f.write(info_string)
    #         if i % 5000 == 0 and i != 0:
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] *= rate_decay
    #             torch.save(self, f'storage/preprocessed/preprocessed_model_checkpoint_iteration_{i}-2019-09-23.pt')
            
             
    def predict(self, arr):
        self.eval()    
        with torch.no_grad():
            arr = arr.view(1,2, 200, 200)
            output_sequence = ['<START>']
            h1 = torch.zeros(2, 1, self.lstm_hidden_size).to(device)
            c1 = torch.zeros(2, 1, self.lstm_hidden_size).to(device)
            h2 = torch.zeros(2, 1, self.lstm_hidden_size).to(device)
            c2 = torch.zeros(2, 1, self.lstm_hidden_size).to(device)
            while output_sequence[-1] != '<END>' and len(output_sequence)<400:
                language_input = torch.Tensor([word_to_ix[output_sequence[-1]]]).type(torch.long).view(1, 1).to(device)
                out, (h1, c1), (h2, c2) = self.forward(arr, language_input, (h1, c1), (h2, c2))
                _, language_input = out[0, 0, :].max(0)
                output_sequence.append(ix_to_word[language_input.item()])
        self.train()
        return output_sequence


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
    tiled = np.tile(vec, (int(height/2), int(width/20))).astype(np.uint8)
    return tiled

def predict_from_image_file(model, path, measure_length, key_number):
    image = io.imread(path)/255
    image = transform.resize(image, (200, 200))
    image = rgb2gray(image).reshape(1, 1, 200, 200)
    measure_length = 16
    key_number = 0
    measure_data_channel = get_measure_data_channel(measure_length, key_number, 200, 200).reshape(1, 1, 200, 200)
    arr = np.concatenate([image, measure_data_channel], axis=1)
    arr = torch.Tensor(arr).type(torch.float).to(device)
    pred_array = model.predict(arr)
    s = str(pc_to_xml(pred_array[1:-1], measure_length, key_number))
    return s


def run_model(path, measure_length, key_number):
    model = Net(len_lexicon, lstm_hidden_size, fc1_output_size, device).to(device)
    model.load_state_dict(torch.load('preprocessed_model_checkpoint_iteration_85000-2019-09-23_state_dict.pt'))
    return predict_from_image_file(model, path, measure_length, key_number)

# print(run_model('media/sample_image.png', 16, -4))
