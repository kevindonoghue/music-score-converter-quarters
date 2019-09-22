import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import MeasureDataset, get_measure_channel_data
import time

device = 'cuda'

batch_size = 1
seq_len = 3
height = 200
width = 200

dataset = MeasureDataset('../data/', seq_len, height, width)
train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

lexicon = list(dataset.words_to_ix)

class ConvSubunit(nn.Module):
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size, filter_size, stride=stride, padding=padding)
        self.dp = nn.Dropout2d(p=dropout)
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()
        self.sequential = nn.Sequential([self.conv, self.dp. self.bn, self.relu])

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
    def __init__(self, len_lexicon, lstm_hidden_size, fc1_output_size, height, width):
        super().__init__()
        self.len_lexion = len_lexicon
        self.lstm_hidden_size = lstm_hidden_size
        self.fc1_output_size = fc1_output_size
        self.num_epochs_trained = 0
        self.cnn = nn.Sequential(ConvUnit(2, 64, 3, 2, 1, 0.25), # (200, 200) --> (100, 100)
                                 ConvUnit(64, 128, 3, 2, 1, 0.25), # (100, 100) --> (50, 50)
                                 ConvUnit(128, 128, 3, 4, 5, 0.25), # (50, 50) --> (10, 10)
                                 ConvUnit(128, 128, 3, 5, 5, 0.25)) # (10, 10) --> (2, 2)
        self.fc1 = nn.Linear(512, self.fc1_output_size)
        self.embed = nn.Embedding(num_embeddings=self.len_lexicon, embedding_dim=5)
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.25)
        self.lstm2 = nn.LSTM(input_size=256+self.lstm_hidden_size, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.25)
        self.fc2 = nn.Linear(self.lstm_hidden_size, self.len_lexicon)
        
    def forward(self, image_input, language_input, internal1=None, internal2=None):
        bs = image_input.shape[0]
        sl = language_input.shape[1]
        if internal1:
            h1, c1 = internal1
        else:
            h1 = torch.zeros(2, bs, self.lstm_hidden_size).cuda()
            c1 = torch.zeros(2, bs, self.lstm_hidden_size).cuda()
        if internal2:
            h2, c2 = internal2
        else:
            h2 = torch.zeros(2, bs, self.lstm_hidden_size).cuda()
            c2 = torch.zeros(2, bs, self.lstm_hidden_size).cuda()
        image_output = self.fc1(self.cnn(image_input).view(bs, 512))
        image_output = image_output.repeat(1, sl).view(bs, sl, self.fc1_output_size)
        language_output, (h1, c1) = self.lstm1(self.embed(language_input), (h1, c1))
        concatenated = torch.cat([image_output, language_output], 2)
        lstm2_out, (h2, c2) = self.lstm2(concatenated, (h2, c2))
        out = self.fc2(lstm2_out)
        return out, (h1, c1), (h2, c2)
    
    def fit(self, dataloader, dataset, optimizer, loss_fn, num_epochs, rate_decay):
        self.word_to_ix = dataset.word_to_ix
        self.ix_to_word = dataset.ix_to_word
        self.height = dataset.height
        self.width = dataset.width
        print('starting fit')
        num_iterations = 0
        for epoch_num in range(num_epochs):
            for batch in dataloader:
                self.train()
                arr = batch['arr']
                seq1 = batch['seq1']
                seq2 = batch['seq2']
                bs = arr.shape[0]
                sl = seq1.shape[1]
                num_iterations += 1
                print(f'starting iteration: {num_iterations}')
                out, _, _ = self.forward(arr, seq1)
                out = out.view(bs*sl, self.len_lexicon)
                targets = seq2.view(bs*sl)
                loss = loss_fn(out, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if num_iterations % 1 == 0:
                    n = np.random.randint(len(dataset))
                    item = dataset[n]
                    arr = item['arr']
                    pc = item['pc'].cpu().numpy()
                    pc = ' '.joint([ix_to_word(ix) for ix in pc])
                    pred_seq = self.predict(arr)
                    pred_seq = ' '.join([ix_to_word(ix) for ix in pred_seq])
                    print('predicted: ' + pred_seq
                    print('true: ' + pc)
                    





        t = time.time()
        for i in range(num_iterations):
            self.train()
            X = language_data[0][:, 0, :]
            y = language_data[0][:, 1, :]
            image_indices = language_data[1]
            batch_indices = np.random.choice(X.shape[0], size=batch_size)
            x = torch.Tensor(X[batch_indices]).type(torch.long).cuda()
            targets = torch.Tensor(y[batch_indices]).type(torch.long).cuda()
            image_batch = torch.Tensor(image_data[image_indices[batch_indices]]).type(torch.float).cuda()
            out, _, _ = self.forward(image_batch, x)
            out = out.view(batch_size*seq_len, len(lexicon))
            targets = targets.view(batch_size*seq_len)
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 100 == 0:
                n = np.random.choice(image_data.shape[0])
                prediction = self.predict(image_data[n])
                print(f'iteration: {i}, loss: {loss}, seconds elapsed: {time.time() - t}')
                print('predicted : ' + prediction)
                print('true      : ' + ' '.join(extra_data[n]['pc']))
                print('---------------------------')
                with open('storage/measure_model_quarters_log_file-2019-09-20.txt', 'a+') as f:
                    f.write(f'iteration: {i}, loss: {loss}, seconds elapsed: {time.time()-t}\n')
                    f.write('predicted :  ' + prediction + '\n')
                    f.write('true      :  ' + ' '.join(extra_data[n]['pc']) + '\n')
                    f.write('------------------------------\n')
                    
            if i % 5000 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= rate_decay
                torch.save(model, f'storage/measure_model_quarters_iteration_{i}_2019-09-20.pt')

                
             
    def predict(self, image):
        self.eval()    
        with torch.no_grad():
            image = torch.Tensor(image).type(torch.float).view(1, 2, 160, 240).cuda()
            output_sequence = ['<START>']
            h1 = torch.zeros(2, 1, 256).cuda()
            c1 = torch.zeros(2, 1, 256).cuda()
            h2 = torch.zeros(2, 1, 256).cuda()
            c2 = torch.zeros(2, 1, 256).cuda()
            while output_sequence[-1] != '<END>' and len(output_sequence)<300:
                language_input = torch.Tensor([word_to_ix[output_sequence[-1]]]).type(torch.long).view(1, 1).cuda()
                out, (h1, c1), (h2, c2) = self.forward(image, language_input, (h1, c1), (h2, c2))
                _, language_input = out[0, 0, :].max(0)
                output_sequence.append(ix_to_word[language_input.item()])
        self.train()
        return ' '.join(output_sequence)