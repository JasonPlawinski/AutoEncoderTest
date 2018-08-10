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
            
            nn.Linear(N, z_dim)        
            )
    def forward(self, x):
        xgauss = self.layers(x)
        return xgauss
    
# Decoder
class P_net(nn.Module):
    def __init__(self):
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
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        out = self.layers(x)
        return out
    
batch_size = 100
X_dim = 784
z_dim = 2
N = 200
# Download Data

mnist_train = datasets.MNIST('/home/jasonplawinski/Documents/EncoderTest', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/home/jasonplawinski/Documents/EncoderTest', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

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
Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

criterionBCE = nn.BCELoss().cuda()

TINY = 1e-8

for i in range(30):
    print(i)
    for batch, label in train_loader:
        X = batch.view([100,1,784])
        X = Variable(X).cuda()
        z_sample = Q(X)
        X_sample = P(z_sample)
        recon_loss = criterionBCE(X_sample + TINY, 
                                            X.resize(batch_size, X_dim) + TINY)
        P_decoder.zero_grad()
        Q_encoder.zero_grad()
        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()

        Q.eval()    
        z_real_gauss = Variable(torch.randn(batch_size, z_dim) * 5)   # Sample from N(0,5)
        z_real_gauss = z_real_gauss.cuda()
        z_fake_gauss = Q(X)
        
        # Compute discriminator outputs and loss
        D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
        D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
        D_gauss_solver.zero_grad()
        D_loss_gauss.backward()       # Backpropagate loss
        D_gauss_solver.step()   # Apply optimization step
        
        # Generator
        Q.train()   # Back to use dropout
        z_fake_gauss = Q(X)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
        G_loss.backward()
        Q_generator.step()

        
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

        
        
#################################################        
pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg.view([1,784])))
rec = rec[0].reshape([28,28])
plt.clf()
plt.imshow(testimg.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitAAE.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy(), cmap='gray')
plt.savefig('./reconAAE.png',dpi = 300)

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg.view([1,784])))
rec = rec[0].reshape([28,28])
plt.clf()
plt.imshow(testimg.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitAAE1.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy(), cmap='gray')
plt.savefig('./reconAAE1.png',dpi = 300)

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg.view([1,784])))
rec = rec[0].reshape([28,28])
plt.clf()
plt.imshow(testimg.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitAAE2.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy(), cmap='gray')
plt.savefig('./reconAAE2.png',dpi = 300)

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