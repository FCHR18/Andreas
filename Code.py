import torch
from torch import nn
import os
from scipy.io import wavfile
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import sys

def nats2bits(x):
    return x/torch.log(2.*torch.ones_like(x))


def pairwise_distance(x, p = 2):
    #Flatten or expand the tensor dimension as requirred.
    if x.ndim > 2:
        x = x.view(x.size(0), -1)
    elif x.ndim == 1:
        x = x.view(-1,1)


    dx = x[:, None] - x     #Easy way to do pairwise subtraction
    dist = torch.norm(dx, p = p, dim = -1)  #Compute the lp-norm
    return dist


class NIBLoss(nn.Module):
    def __init__(self, var, beta, alpha):
        super(NIBLoss, self).__init__()
        self.var = var
        self.beta = beta
        self.alpha = alpha
        self.MSE = nn.MSELoss()

    def get_distortion(self, y_pred, y_true):
        return self.MSE(y_pred,y_true)

    def get_entropy_rate(self, quant, var):

        n_batch, _ = quant.shape
        # Compute normalisation constant
        c = np.log(n_batch)


        dist = pairwise_distance(quant)**2
        # print(dist)
        kde_contribution = torch.mean(torch.logsumexp(-0.5*dist/var, dim = 1))
        # print(kde_contribution,c)
        IXT = c - kde_contribution
        IXT = nats2bits(IXT)
        return IXT

    def forward(self, y_true, y_pred, encoded):

        rate = self.get_entropy_rate(encoded, self.var)
        distortion = self.get_distortion(y_pred, y_true)
        if self.alpha != 0:
            loss = self.alpha*rate + self.beta*distortion
            return loss, rate, distortion
        loss = self.beta*distortion
        return loss, rate, distortion


class CustomDataset(Dataset):
    def __init__(self, paths, start, stop, overlap, segment_length):

        #Store the paths in object's memory
        self.paths = paths
        self.overlap = overlap
        self.start = start
        self.stop = stop

        self.segment_length = segment_length
        self.duration = stop-start

        temp = []
        for files in paths:
            _ , file = wavfile.read(files)
            file = file / np.max(np.abs(file))
            sub_file = self._get_preprocessing(file)
            if not len(sub_file):
                sys.exit(f'The length chosen is to big, should be less than {len(file)} and is {self.stop-self.start}.')
                break
            segments = self._get_segmented_file(sub_file)
            if not len(segments):
                sys.exit(f'''
\nThe overlap is > 0 ({self.overlap}),
or the segment length is > the duration ({self.segment_length > (self.stop-self.start)}),
or the segment length is < overlap ({self.segment_length <self.overlap})
 ''')
                break
            temp.append(segments)
        self.items = torch.from_numpy(np.vstack(temp)).float()

    def _get_preprocessing(self, file):
        new_file_length = self.duration
        max_length = new_file_length + self.start
        if max_length > len(file):
            return []
        sub_file = file[self.start:max_length]
        return sub_file # returns file of size: duration, with starting point at begin


    def __len__(self):
        return len(self.items)


    def __getitem__(self, index):
        return self.items[index]

    def _get_segmented_file(self, file):
        if self.segment_length >= self.overlap+1:
            if self.overlap > 0:
                a, b = 0, self.segment_length
                temp = [file[:self.segment_length]]
                for _ in range(len(file)):
                    a += (self.segment_length-self.overlap)
                    b += (self.segment_length-self.overlap)
                    if b > len(file):
                        break
                    temp.append(file[a:b])
                return np.array(temp)

            if self.overlap == 0:
                numb_of_splits = len(file)//self.segment_length
                file = [file[i*self.segment_length:(i+1)*self.segment_length] for i in range(numb_of_splits)]
                return np.array(file)

        return []

class QuantiserAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit_depth, quant_size):
        xmin = -1000
        xmax = 1000

        x = input

        x[x < xmin] = xmin
        x[x > xmax] = xmax

        q_int = torch.linspace(xmin, xmax, 2**bit_depth)
        dist = (torch.max(q_int)-torch.min(q_int))/(len(q_int))

        q_int = q_int.to(dev)
        dist = dist.to(dev)

        index = torch.bucketize(torch.subtract(x,dist/2),q_int)

        qx = q_int[index]
        # qx = qx - noise
        return qx

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class Encoder(nn.Module):
    def __init__(self, in_size, hid_sizes, q_size):
        super(Encoder, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_size, hid_sizes[0]),
            nn.ELU(),
            nn.Linear(hid_sizes[0], hid_sizes[1]),
            nn.ELU(),
            nn.Linear(hid_sizes[1], hid_sizes[2]),
            nn.ELU(),
            nn.Linear(hid_sizes[2], q_size)
        )
        # Parameters


    def forward(self, input):

        encoded = self.seq(input)
        return encoded


class Decoder(nn.Module):
    def __init__(self, in_size, hid_sizes, out_size, var, batch, bit_depth):
        super(Decoder, self).__init__()
        self.var = var
        self.in_size = in_size
        self.batch = batch
        self.bit_depth = bit_depth

        self.seq = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ELU(),
            nn.Linear(out_size, out_size),
            nn.ELU(),
            nn.Linear(out_size, out_size),
            nn.ELU(),
            nn.Linear(out_size, out_size)
        )
        # Parameters
        self.quantiser = QuantiserAutograd.apply
    def forward(self, input):
        decoded = input +  self.var*torch.randn((self.batch,1)).to(dev)
        decoded = self.quantiser(decoded, self.bit_depth, self.in_size)
        decoded = self.seq(decoded)
        return decoded


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

folder_name = 'Andreas/Audio files/classical'
all_paths = os.listdir(folder_name)
all_paths = [folder_name + '/' + all_paths[i] for i in range(len(all_paths))]

#Read declared wavfiles
sr , _ = wavfile.read(all_paths[0])

#Declare how many files should be used for training an testing
n_train_files = 20
n_tst_files = 4

#Declare paths to training and testing files
train_paths = all_paths[:n_train_files]
tst_paths = all_paths[n_train_files:n_train_files+n_tst_files]

# Create Dataloader and declare parameters (overlap, start, stop, batch_size)
start = 0 #where to start reading file
stop = 30*sr #where to stop reading file
overlap = 0 #Choose integer >= 0
input_size = 50 #how many segments to load
batch = 10000 # Batch size
quant_size = list(range(1,input_size+1))
quant_size[0] = 1
var = 1
beta = 500
alpha = 1
hidden_sizes = [1000, 1500, 800]

bit_depth = list(range(1,16+1))

train_dataset = CustomDataset(train_paths, start, stop, overlap, input_size)
train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)

test_dataset = CustomDataset(tst_paths, start, stop, overlap, input_size)
test_data_loader = DataLoader(test_dataset,batch_size=batch,shuffle=False)

bit_plt = []
all_loss_plt =[]
all_rate_plt = []
all_distortion_plt = []

np.savetxt('Quant.txt', quant_size)
np.savetxt('bit.txt', bit_depth)

quant_plt = []
for k in bit_depth:
    bit_plt.append(k)
    loss_plt = []
    rate_plt = []
    distortion_plt = []
    for i in quant_size:
        # quant_plt.append(i)

        criterion = NIBLoss(var, beta, alpha)
        tst_criterion = NIBLoss(var, beta, alpha)

        encoder = Encoder(input_size, hidden_sizes, i)
        decoder = Decoder(i, hidden_sizes, input_size, var, batch, k)

        encoder = encoder.to(dev)
        decoder = decoder.to(dev)

        params_to_optimize = [
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ]
        #Define loss function and optimiser
        optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3, weight_decay=1e-5)
        #        model.load_state_dict(torch.load('model_weights_pretrain.pth'))

        #Declare number of epochs to train for
        num_epochs = 40
        for epoch in range(num_epochs):
            #Create tensor which will contain the resulting loss for each epoch
            for data in train_data_loader:
                if len(data) == batch:
                    optimizer.zero_grad()
                    data = data.to(dev)
                    #Compute model output and loss
                    encoded = encoder(data)
                    encoded = encoded.to(dev)
                    recon = decoder(encoded)
                    recon = recon.to(dev)

                    loss, rate, distortion = criterion(data, recon, encoded)

                    #Backward step
                    loss.backward()
                    optimizer.step()
            #Append loss for a given epoch to array
            print(f'Training: {i} Epoch:{epoch+1}/{num_epochs}, Loss:{loss.item():.8f},  rate:{rate}, distortion: {distortion}')

        temp = []
        with torch.no_grad():
            for data in test_data_loader:
                if len(data) == batch:
                    #Compute model output and loss
                    data = data.to(dev)
                    encoded = encoder(data)
                    encoded = encoded.to(dev)
                    recon = decoder(encoded)
                    real = np.concatenate(np.vstack(data.cpu().detach().numpy()))
                    sound = np.concatenate(np.vstack(recon.cpu().detach().numpy()))
                    temp.append(sound)
                    tst_loss, rate, distortion = tst_criterion(data, recon, encoded)
            print(f'loss {tst_loss.item()}, rate:{rate}, distortion: {distortion}')
            loss_plt.append(tst_loss.item())
            rate_plt.append(rate.item())
            distortion_plt.append(distortion.item())

    all_loss_plt.append(loss_plt)
    all_distortion_plt.append(distortion_plt)
    all_rate_plt.append(rate_plt)


np.savetxt('Results_loss.txt', all_loss_plt)
np.savetxt('Results_dist.txt', all_distortion_plt)
np.savetxt('Results_rate.txt', all_rate_plt)

