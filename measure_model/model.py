import numpy as np
import json
import torch
import torch.nn as nn
import os
import time
from PIL import Image
from pc_to_xml import pc_to_xml


lexicon = ['6', '<START>', '12', '7', 'rest', 'pitch', '3', 'E', '}', 'quarter', '2', '16', '4', 'C', 'measure', '<END>', '0', 'staff', 'duration', 'G', 'chord', 'D', '5', 'F', 'B', 'note', 'backup', 'type', 'A', '-1', '1', '<PAD>']
word_to_ix = {word: ix for ix, word in enumerate(lexicon)}
ix_to_word = {ix: word for ix, word in enumerate(lexicon)}

batch_size = 128
seq_len = 64

class ConvUnit(nn.Module):
    def __init__(self, input_size, output_size, filter_size, stride, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, filter_size, stride=1, padding=padding)
        self.dp1 = nn.Dropout2d(p=dropout)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_size, output_size, filter_size, stride=stride, padding=padding)
        self.dp2 = nn.Dropout2d(p=dropout)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.dp1(self.conv1(x))))
        x = self.relu2(self.bn2(self.dp2(self.conv2(x))))
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(ConvUnit(2, 64, 3, 2, 1, 0.25), # (160, 240) --> (80, 120)
                                 ConvUnit(64, 128, 3, 2, 1, 0.25), # (80, 120) --> (40, 60)
                                 ConvUnit(128, 128, 3, 4, 1, 0.25), # (40, 60) --> (10, 15)
                                 ConvUnit(128, 128, 3, 5, 1, 0.25)) # (10, 15) --> (2, 3)
        self.fc1 = nn.Linear(768, 256)
        self.embed = nn.Embedding(num_embeddings=len(lexicon), embedding_dim=5)
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=256, num_layers=2, batch_first=True, dropout=0.25)
        self.lstm2 = nn.LSTM(input_size=256+256, hidden_size=256, num_layers=2, batch_first=True, dropout=0.25)
        self.fc2 = nn.Linear(256, len(lexicon))
        
    def forward(self, image_input, language_input, internal1=None, internal2=None):
        bs = image_input.shape[0]
        seq_len = language_input.shape[1]
        if internal1:
            h1, c1 = internal1
        else:
            h1 = torch.zeros(2, bs, 256).cuda()
            c1 = torch.zeros(2, bs, 256).cuda()
        if internal2:
            h2, c2 = internal2
        else:
            h2 = torch.zeros(2, bs, 256).cuda()
            c2 = torch.zeros(2, bs, 256).cuda()
        image_output = self.fc1(self.cnn(image_input).view(bs, 768))
        image_output = image_output.repeat(1, seq_len).view(bs, seq_len, 256)
        language_output, (h1, c1) = self.lstm1(self.embed(language_input), (h1, c1))
        concatenated = torch.cat([image_output, language_output], 2)
        lstm2_out, (h2, c2) = self.lstm2(concatenated, (h2, c2))
        out = self.fc2(lstm2_out)
        return out, (h1, c1), (h2, c2)
    
    # def fit(self, image_data, language_data, optimizer, loss_fn, num_iterations, seq_len, rate_decay):
    #     t = time.time()
    #     for i in range(num_iterations):
    #         self.train()
    #         X = language_data[0][:, 0, :]
    #         y = language_data[0][:, 1, :]
    #         image_indices = language_data[1]
    #         batch_indices = np.random.choice(X.shape[0], size=batch_size)
    #         x = torch.Tensor(X[batch_indices]).type(torch.long).cuda()
    #         targets = torch.Tensor(y[batch_indices]).type(torch.long).cuda()
    #         image_batch = torch.Tensor(image_data[image_indices[batch_indices]]).type(torch.float).cuda()
    #         out, _, _ = self.forward(image_batch, x)
    #         out = out.view(batch_size*seq_len, len(lexicon))
    #         targets = targets.view(batch_size*seq_len)
    #         loss = loss_fn(out, targets)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
            
    #         if i % 100 == 0:
    #             n = np.random.choice(image_data.shape[0])
    #             prediction = self.predict(image_data[n])
    #             print(f'iteration: {i}, loss: {loss}, seconds elapsed: {time.time() - t}')
    #             print('predicted : ' + prediction)
    #             print('true      : ' + ' '.join(extra_data[n]['pc']))
    #             print('---------------------------')
    #             with open('storage/measure_model_quarters_log_file-2019-09-20.txt', 'a+') as f:
    #                 f.write(f'iteration: {i}, loss: {loss}, seconds elapsed: {time.time()-t}\n')
    #                 f.write('predicted :  ' + prediction + '\n')
    #                 f.write('true      :  ' + ' '.join(extra_data[n]['pc']) + '\n')
    #                 f.write('------------------------------\n')
                    
    #         if i % 5000 == 0:
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] *= rate_decay
    #             torch.save(model, f'storage/measure_model_quarters_iteration_{i}_2019-09-20.pt')

                
             
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

    def predict_from_image_file(self, path, measure_length, key_number):
        image = Image.open(path)
        image = image.resize((240, 160))
        image = image.convert('1')
        image_np = np.array(image).astype(np.uint8)
        image_np *= 255
        image_np = 255 - image_np

        def create_aug_array(measure_length, key_number):
            key_number += 7
            vec = np.zeros((2, 20))
            vec[0, key_number] += 1
            if measure_length == 12:
                vec[1, 0] += 1
            elif measure_length == 16:
                vec[1, 1] += 1
            tiled = np.tile(vec, (80, 12))
            return tiled

        image_np = np.array([image_np, create_aug_array(measure_length, key_number)])
        image_np = image_np.reshape(1, 2, 160, 240)
        try:
            pc_pred_str = self.predict(image_np)
            pc_pred = pc_pred_str.split()[1:-1]
            xml_pred = pc_to_xml(pc_pred, measure_length, key_number)
            return str(xml_pred)
        except:
            print('there was an error converting the string')
            print(pc_pred_str)

# model = torch.load('measure_model_quarters_iteration_45000_2019-09-20.pt')
# print(model.predict_from_image_file('sample_image.png', measure_length=16, key_number=-4))
