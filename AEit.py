'''This script is used to generate a video to visualize the network creating a meaningful embedding in the latent space.
To see commented code refer to AE.py'''

import math
import pickle
import torch
import torch.nn as nn
import numpy as np
import math, random
import torch.nn.init as init
from torch import functional as F

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
from scipy import ndimage as ndi
from skimage import feature

#Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_dim, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N0, z_dim)        
            )
    def forward(self, x):
        xgauss = self.layers(x)
        return xgauss
    
# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, N0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N0, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, X_dim),
            nn.Sigmoid()
            )
    def forward(self, x):
        out = self.layers(x)
        return out


batch_size = 100
X_dim = 784
N = 150
N0 = 25
z_dim = 2
# Set learning rates
gen_lr = 0.0001

# Download Data

mnist_train = datasets.MNIST('./', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('./', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)


torch.manual_seed(10)
Q = Q_net()
P = P_net()     # Encoder/Decoder

Q = Q.cuda()
P = P.cuda()

# Set optimizators
P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

criterionMSE = nn.MSELoss().cuda()

imgnumber = 0
'''Same as the Auto Encoder Network except every 100th iteration (each iteration of batch with size 100 so every 10 000 images, since the dataset is composed of 50 000 images it means 5 images per epoch), an image is saved'''
for epoch in range(20):
    print(epoch)
    
    Q.train()
    P.train()
    iteration = 0
    for batch, label in train_loader:
        X = batch.view([100,1,784])
        X = Variable(X).cuda()
        z_sample = Q(X)
        X_sample = P(z_sample)
        recon_loss = criterionL1(X_sample, X)
        P_decoder.zero_grad()
        Q_encoder.zero_grad()
        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()
        iteration += 1
        if iteration%100 == 0:
            

            Q.eval()
            P.eval()

            List0x = []
            List0y = []
            List1x = []
            List1y = []
            List2x = []
            List2y = []
            List3x = []
            List3y = []
            List4x = []
            List4y = []
            List5x = []
            List5y = []
            for i in range(400):
                pair = iter(test_loader).next()
                testimg = pair[0]
                label = pair[1]
                testimg = Variable(testimg.view([1,784])).cuda()
                coor = Q(testimg)
                coorformat0 = coor[0][0].data.cpu().numpy()
                coorformat1 = coor[0][1].data.cpu().numpy()
                label = label.cpu().numpy()[0]
                if label == 0:
                    List0x.append(coorformat0)
                    List0y.append(coorformat1)
                if label == 1:
                    List1x.append(coorformat0)
                    List1y.append(coorformat1)
                if label == 2:
                    List2x.append(coorformat0)
                    List2y.append(coorformat1)
                if label == 3:
                    List3x.append(coorformat0)
                    List3y.append(coorformat1)
                if label == 4:
                    List4x.append(coorformat0)
                    List4y.append(coorformat1)
                if label == 5:
                    List5x.append(coorformat0)
                    List5y.append(coorformat1)

            plt.scatter(List0x,List0y,label='0', s=20)
            plt.scatter(List1x,List1y,label='1', s=20)
            plt.scatter(List2x,List2y,label='2', s=20)
            plt.scatter(List3x,List3y,label='3', s=20)
            plt.scatter(List4x,List4y,label='4', s=20)
            plt.scatter(List5x,List5y,label='5', s=20)
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='upper right')
            plt.title('Epoch '+str(epoch))

            plt.savefig('./AEresults/Hidden' +str(imgnumber) +'.png',dpi = 300)
            plt.clf()
            imgnumber += 1