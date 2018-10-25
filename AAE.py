'''Implementation of the Adversarial Autoencoder'''

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
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
from scipy import ndimage as ndi
from skimage import feature


#Encoder
class Q_net(nn.Module):
    '''This class is for the Encoder.
    The Hyperparamaters such as the number of Neuron per layer and the the dropout rate can be modified before execution
    The architecture is composed of 4 layers
    The activation is Leaky ReLU
    X_dim is the size of the input, here 784
    N is the number of neurons in the hidden layers
    z_dim is the size of the bottleneck, here 2 so it can be visualized easily'''
    def __init__(self):
        super(Q_net, self).__init__()
        self.layers = nn.Sequential(
            #input size is X_dim, here 784 (vectorized 28x28)
            nn.Linear(X_dim, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            #output size is z_dim usually 2 for visualization
            nn.Linear(N, z_dim)        
            )
    def forward(self, x):
        #forward pass
        xgauss = self.layers(x)
        return xgauss
    
# Decoder
class P_net(nn.Module):
    def __init__(self):
        '''This class is for the Decoder.
    The Hyperparamaters such as the number of Neuron per layer and the the dropout rate can be modified before execution
    The architecture is composed of 4 layers
    The activation is Leaky ReLU
    z_dim is the size of the bottleneck, here 2 so it can be visualized easily
    N is the number of neurons in the hidden layers
    X_dim is the output size for the reconstruction, here 784'''
        super(P_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, N),
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


# Discriminator
class D_net_gauss(nn.Module):
    '''This class is for the Discriminator network0
    The goal of this network is to guess if its input comes from the Encoder or a Gaussian distribution computed with randn.
    The Hyperparamaters such as the number of Neuron per layer and the the dropout rate can be modified before execution
    The architecture is composed of 3 layers
    The activation is Leaky ReLU
    z_dim is the size of the bottleneck, here 2 so it can be visualized easily
    N is the number of neurons in the hidden layers
    The output size here is 1 since the discriminator's output is a scalar representing its confidence of its input being from
    the real or encoded distribution'''
    def __init__(self):        
        super(D_net_gauss, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        out = self.layers(x)
        return out

# The hyperparameters can be modified here
batch_size = 100
X_dim = 784
z_dim = 2
N = 200
# Download Data

mnist_train = datasets.MNIST('./', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('./', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)


torch.manual_seed(10)
Q = Q_net()
P = P_net()     # Encoder/Decoder
D_gauss = D_net_gauss()                # Discriminator adversarial

Q = Q.cuda()
P = P.cuda()
D_cat = D_gauss.cuda()
D_gauss = D_net_gauss().cuda()


# Set learning rates
gen_lr, reg_lr = 0.0006, 0.0008


# Set optimizators
P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)
D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

#Reconstruction loss
criterionMSE = nn.MSELoss().cuda()

TINY = 1e-8
'''Training of the Adversarial auto encoder'''
for i in range(100):
    for batch, label in train_loader:
        #Vectorizing the input
        X = batch.view([batch_size,1,784])
        #GPU variable
        X = Variable(X).cuda()
        
        #Compressed digit
        z_sample = Q(X)
        #Reconstruction
        X_sample = P(z_sample)
        
        #reconstruction loss using MSE
        recon_loss = criterionMSE(X_sample, X.resize(batch_size, X_dim))
        
        #Updating the weights of the Encoder and the Decoder
        P_decoder.zero_grad()
        Q_encoder.zero_grad()
        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()
        
        #Evaluation mode so dropout is off
        Q.eval()
        #Generating samples from a Gaussian distribution
        z_real_gauss = Variable(torch.randn(batch_size, z_dim) * 5)   # Sample from N(0,5)
        z_real_gauss = z_real_gauss.cuda()
        #Latent code (compression of the image)
        z_fake_gauss = Q(X)
        
        # Output of the Discriminator for gaussian and compressed z_dim dimensional vector
        D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
        
        #Loss of the discriminator from the template distribution
        D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
        
        # Optimisation of the Discriminator
        D_gauss_solver.zero_grad()
        D_loss_gauss.backward()
        D_gauss_solver.step()
        
        # Updating Generator/Encoder
        Q.train()
        z_fake_gauss = Q(X)
        D_fake_gauss = D_gauss(z_fake_gauss)
        
        Q_encoder.zero_grad()
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
        G_loss.backward()
        Q_encoder.step()

        
#Evaluation mode so dropout is used well
Q.eval()
P.eval()

#Not the best practise but I'm only going to plot 5 digits so I made 10 lists hehe
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
    '''Taking 400 digits from test set (even though it would be ok to take those sample from the training loader
    since we are inspecting the embedding space)'''
    #Iterate on the dataloader to get 1 item
    pair = iter(test_loader).next()
    
    #get the label and image
    testimg = pair[0]
    label = pair[1]
    testimg = Variable(testimg.view([1,784])).cuda()
    
    #Get the 2D coordinates from the data from the Encoder
    coor = Q(testimg)
    coorformat0 = coor[0][0].data.cpu().numpy()
    coorformat1 = coor[0][1].data.cpu().numpy()
    label = label.cpu().numpy()[0]
    
    #Same garbage practices but a copy pasting was really tempting
    #retrieving labels from the and getting coordinates classified by digits
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

        
        
#################################################
############# Saving some results ###############
#################################################

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg.view([1,784])))
rec = rec[0]

plt.clf()
plt.imshow(testimg.data.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitAAE.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy().reshape([28,28]), cmap='gray')
plt.savefig('./reconAAE.png',dpi = 300)

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg.view([1,784])))
rec = rec[0]

plt.clf()
plt.imshow(testimg.data.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitAAE1.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy().reshape([28,28]), cmap='gray')
plt.savefig('./reconAAE1.png',dpi = 300)

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg.view([1,784])))
rec = rec[0]

plt.clf()
plt.imshow(testimg.data.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitAAE2.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy().reshape([28,28]), cmap='gray')
plt.savefig('./reconAAE2.png',dpi = 300)

#################################################
############# Saving Latent space ###############
#################################################

plt.clf()
plt.scatter(List0x,List0y,label='0', s=20)
plt.scatter(List1x,List1y,label='1', s=20)
plt.scatter(List2x,List2y,label='2', s=20)
plt.scatter(List3x,List3y,label='3', s=20)
plt.scatter(List4x,List4y,label='4', s=20)
plt.scatter(List5x,List5y,label='5', s=20)
plt.legend(loc='upper right')

print('Len list')
print(len(List0x))
print(len(List1x))
print(len(List2x))
print(len(List3x))
print(len(List4x))
print(len(List5x))

plt.savefig('./Adv.png',dpi = 300)