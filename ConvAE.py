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
        self.Input = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,64,3,1,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64,128,3,1,0),
            nn.InstanceNorm2d(128),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128,128,3,2,0),
            nn.InstanceNorm2d(128),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,128,3,2,0),
            nn.InstanceNorm2d(128),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True))
            
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,3,2,0),
            nn.InstanceNorm2d(128),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.LinLayers = nn.Sequential(
            nn.Linear(512, 500),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, 500),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True)
        )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, batch_size):
        out = self.Input(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view([batch_size,1,-1])
        out = self.LinLayers(out)
        return out
    
# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.LinLayers = nn.Sequential(
            nn.Linear(500, 800),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(800, 784),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True)
        )
            
        self.layer1 = nn.Sequential(
            nn.Conv2d(16,64,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16,64,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(64,64,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.InstanceNorm2d(64))
            
        
        self.layer2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16,64,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(64,64,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.InstanceNorm2d(64))
        
      
        self.Output = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(128,128,3,1,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv2d(128,1,3,1,1),
            nn.Sigmoid())
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, x, batch_size):
        out = self.LinLayers(x)
        out = out.view([batch_size,-1,7,7])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.Output(out)
        return out


batch_size = 100
X_dim = 784
N = 150
N0 = 25
z_dim = 2

# Download Data

mnist_train = datasets.MNIST('/home/jasonplawinski/Documents/EncoderTest', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('/home/jasonplawinski/Documents/EncoderTest', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=1,shuffle=True)


torch.manual_seed(1)
Q = Q_net()
P = P_net()     # Encoder/Decoder

Q = Q.cuda()
P = P.cuda()

Q_parameters = filter(lambda p: p.requires_grad, Q.parameters())
params = sum([np.prod(p.size()) for p in Q_parameters])
print("Learnable parameters in Encoder :", params)
P_parameters = filter(lambda p: p.requires_grad, P.parameters())
params = sum([np.prod(p.size()) for p in P_parameters])
print("Learnable parameters in Decoder :", params)

# Set learning rates
gen_lr, reg_lr = 0.0005, 0.0008
# Set optimizators
P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

criterionL1 = nn.MSELoss().cuda()

TINY = 1e-8

for i in range(1):
    loss = []
    print(i)
    for batch, label in train_loader:
        X = Variable(batch).cuda()
        z_sample = Q(X, batch_size)
        X_sample = P(z_sample, batch_size)
        recon_loss = criterionL1(X_sample, X)
        P_decoder.zero_grad()
        Q_encoder.zero_grad()
        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()
        loss.append(recon_loss.data)
    print(np.mean(np.array(loss)))
        
    

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
    testimg = Variable(testimg).cuda()
    coor = Q(testimg, 1)
    coorformat0 = coor[0][0][0].data.cpu().numpy()
    coorformat1 = coor[0][0][1].data.cpu().numpy()
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

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg).cuda()
rec = P(Q(testimg, 1),1)

print(testimg.size())
plt.imshow(testimg.cpu().numpy()[0][0][:], cmap='gray')
plt.show()

print(rec.size())
plt.imshow(rec.data.cpu().numpy()[0][0][:], cmap='gray')
plt.show()

print(rec)

plt.scatter(List0x,List0y,label='0', s=20)
plt.scatter(List1x,List1y,label='1', s=20)
plt.scatter(List2x,List2y,label='2', s=20)
plt.scatter(List3x,List3y,label='3', s=20)
plt.scatter(List4x,List4y,label='4', s=20)
plt.scatter(List5x,List5y,label='5', s=20)
plt.legend()

print('Len list')
print(len(List0x))
print(len(List1x))
print(len(List2x))
print(len(List3x))
print(len(List4x))
print(len(List5x))

plt.savefig('./ConvAE.png',dpi = 300)