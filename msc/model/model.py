import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .data import MeasureDataset, get_measure_data_channel
import time
from skimage import io, transform
from skimage.color import rgb2gray


def convert_seconds(x):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)
    return f'{hours}h {minutes}m {seconds}s'


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
        self.subunit1 = ConvSubunit(input_size, output_size, filter_size, stride, padding, dropout)
        
    def forward(self, x):
        x = self.subunit1(x)
        return x

class Net(nn.Module):
    def __init__(self, len_lexicon, lstm_hidden_size, fc1_output_size, device):
        super().__init__()
        self.len_lexicon = len_lexicon
        self.lstm_hidden_size = lstm_hidden_size
        self.fc1_output_size = fc1_output_size
        self.device = device
        self.num_iterations = 0
        self.train_time = 0
        self.cnn = nn.Sequential(ConvUnit(2, 64, 3, 2, 1, 0.25), # (200, 200) --> (100, 100)
                                 ConvUnit(64, 128, 3, 2, 1, 0.25), # (100, 100) --> (50, 50)
                                 ConvUnit(128, 128, 3, 5, 1, 0.25), # (50, 50) --> (10, 10)
                                 ConvUnit(128, 128, 3, 5, 1, 0.25)) # (10, 10) --> (2, 2)
        self.fc1 = nn.Linear(512, self.fc1_output_size)
        self.embed = nn.Embedding(num_embeddings=self.len_lexicon, embedding_dim=5)
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.25)
        self.lstm2 = nn.LSTM(input_size=self.fc1_output_size+self.lstm_hidden_size, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.25)
        self.fc2 = nn.Linear(self.lstm_hidden_size, self.len_lexicon)
        self.word_to_ix = None
        self.ix_to_word = None
        self.height = None
        self.width = None
        
    def forward(self, image_input, language_input, internal1=None, internal2=None):
        bs = image_input.shape[0]
        sl = language_input.shape[1]
        if internal1:
            h1, c1 = internal1
        else:
            h1 = torch.zeros(2, bs, self.lstm_hidden_size).to(self.device)
            c1 = torch.zeros(2, bs, self.lstm_hidden_size).to(self.device)
        if internal2:
            h2, c2 = internal2
        else:
            h2 = torch.zeros(2, bs, self.lstm_hidden_size).to(self.device)
            c2 = torch.zeros(2, bs, self.lstm_hidden_size).to(self.device)
        image_output = self.fc1(self.cnn(image_input).view(bs, 512))
        image_output = image_output.repeat(1, sl).view(bs, sl, self.fc1_output_size)
        language_output, (h1, c1) = self.lstm1(self.embed(language_input), (h1, c1))
        concatenated = torch.cat([image_output, language_output], 2)
        lstm2_out, (h2, c2) = self.lstm2(concatenated, (h2, c2))
        out = self.fc2(lstm2_out)
        return out, (h1, c1), (h2, c2)
    
    def fit(self, dataset, dataloader, optimizer, loss_fn, num_epochs, rate_decay, print_every=100):
        self.word_to_ix = dataset.word_to_ix
        self.ix_to_word = dataset.ix_to_word
        self.height = dataset.height
        self.width = dataset.width
        aux_data = dataset.aux_data
        total_iterations = num_epochs * len(dataloader)
        print('starting fit')
        for epoch_num in range(num_epochs):
            iteration_start_time = time.time()
            for batch in dataloader:
                self.num_iterations += 1
                self.train()
                arr = batch['arr']
                seq1 = batch['seq1']
                seq2 = batch['seq2']
                bs = arr.shape[0]
                sl = seq1.shape[1]
                out, _, _ = self.forward(arr, seq1)
                out = out.view(bs*sl, self.len_lexicon)
                targets = seq2.view(bs*sl)
                loss = loss_fn(out, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if self.num_iterations % print_every == 0:
                    time_elapsed = self.train_time
                    hours_elapsed = time_elapsed // 3600
                    minutes_elapsed = (time_elapsed % 3600) // 60
                    seconds_elapsed = time_elapsed % 60
                    if time_elapsed == 0:
                        time_per_iteration = 0
                    else:
                        time_per_iteration = self.num_iterations/time_elapsed
                    time_left = (total_iterations - self.num_iterations) * time_per_iteration
                    n = np.random.randint(len(dataset))
                    item = dataset[n]
                    arr = item['arr']
                    image_number = item['image_number'].item()
                    pc = aux_data[image_number]['pc']
                    pc = ' '.join([self.ix_to_word[str(ix)] for ix in pc])
                    pred_seq = self.predict(arr)
                    pred_seq = ' '.join(pred_seq)
                    with open('./log.txt', 'a+') as f:
                        info_string = f"""
                        ----
                        iteration: {self.num_iterations} epochs: {self.num_iterations/len(dataloader)}
                        time elapsed: {convert_seconds(time_elapsed)} time left: {convert_seconds(time_left)}
                        ----
                        pred: {pred_seq}
                        ----
                        true: {pc}
                        ----



                        """.replace('    ', '')
                        print(info_string)
                        f.write(info_string)
                self.train_time += time.time() - iteration_start_time
                iteration_start_time = time.time()
        for param_group in optimizer.param_groups:
            param_group['lr'] *= rate_decay
        torch.save(self, f'./epoch_{epoch_num+1}.pt')
                
             
    def predict(self, arr):
        # arr is a torch Tensor of shape (2, self.height, self.width)
        # arr[0] is the image data and arr[1] is the time/key sig data
        self.eval()    
        with torch.no_grad():
            arr = arr.view(1,2, self.height, self.width)
            output_sequence = ['<START>']
            h1 = torch.zeros(2, 1, self.lstm_hidden_size).to(self.device)
            c1 = torch.zeros(2, 1, self.lstm_hidden_size).to(self.device)
            h2 = torch.zeros(2, 1, self.lstm_hidden_size).to(self.device)
            c2 = torch.zeros(2, 1, self.lstm_hidden_size).to(self.device)
            while output_sequence[-1] != '<END>' and len(output_sequence)<400:
                language_input = torch.Tensor([self.word_to_ix[output_sequence[-1]]]).type(torch.long).view(1, 1).to(self.device)
                out, (h1, c1), (h2, c2) = self.forward(arr, language_input, (h1, c1), (h2, c2))
                _, language_input = out[0, 0, :].max(0)
                output_sequence.append(self.ix_to_word[str(language_input.item())])
        self.train()
        return output_sequence

    def predict_from_image(self, path, measure_length, key_number):
        # path should be path to a png
        image = io.imread(path)/255
        # handle the conversion from rgb and rgba pngs
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = rgb2gray(image)
            elif image.shape[3] == 4:
                image = rgb2gray(image[:, :, :3])
        image = transform.resize(image, (self.height, self.width), cval=1)
        measure_channel = get_measure_data_channel(measure_length, key_number, self.height, self.width)
        arr = np.array([image, measure_channel])
        arr = torch.Tensor(arr).type(torch.float).to(self.device)
        return self.predict(arr)

