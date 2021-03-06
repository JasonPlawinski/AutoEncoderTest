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
    '''The Encoder Network is a convolutional network with a few convolutional
    The convolutions are 2D strided to reduce the size of the image. The last layer is fully connected
    Dropout is applied to all layers except the dense one
    Activation is leaky ReLU
    The output of the convolutions are normalized using Instance Normalization
    Output is not Normalized'''
    def __init__(self):
        super(Q_net, self).__init__()
        self.Input = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1,16,3,1,0),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16,32,3,1,0),
            nn.InstanceNorm2d(32),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.InstanceNorm2d(32),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.InstanceNorm2d(32),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,32,3,2,0),
            nn.InstanceNorm2d(32),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
        
        self.LinLayers = nn.Sequential(
            nn.Linear(128, 100),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 2),
            nn.LeakyReLU(0.2))
            
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
    '''The Decoder Network starts off with dense layers then followed up with 2D convolutions and Pixel shuffler to upscale the image
    The pixel shuffler use fractional convolutions and upscale an image by re arranging pixels
    from multiple channels into one larger channel.
    For example for x2 upsampling in 2D, the output of the pixel shuffler will have 4 times less channels
    Activation is Leaky RelU
    The output of a block of convolutions is normalized using Instance Normalization
    Dropout is applied to all hidden layers
    Output is Sigmoid'''
    def __init__(self):
        super(P_net, self).__init__()
        self.LinLayers = nn.Sequential(
            nn.Linear(2, 100),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 784),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2))
            
        self.layer1 = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.PixelShuffle(2),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(4,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.InstanceNorm2d(16))
            
        
        self.layer2 = nn.Sequential(
            nn.PixelShuffle(2),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(4,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.InstanceNorm2d(16))
        
      
        self.Output = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16,16,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.25),
            nn.Conv2d(16,1,3,1,1),
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
z_dim = 2
gen_lr = 0.0005

# Download Data

mnist_train = datasets.MNIST('./', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = datasets.MNIST('./', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

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


# Set optimizators
P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

#MSE for reconstruction loss
criterionMSE = nn.MSELoss().cuda()

for i in range(100):
    for batch, label in train_loader:
        #Vectorizing the input
        X = batch.view([batch_size,1,28,28])
        #GPU variable
        X = Variable(X).cuda()
        
        #Get Encoder to generate the latent space vector
        z_sample = Q(X, batch_size)
        #Get the Decoder to generate the reconstruction
        X_sample = P(z_sample, batch_size)
        
        #Reconstruction loss (MSE)
        recon_loss = criterionMSE(X_sample, X)
        
        #Update the parameters of the Encoder and the Decoder
        P_decoder.zero_grad()
        Q_encoder.zero_grad()
        recon_loss.backward()
        P_decoder.step()
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
    testimg = Variable(testimg.view([1,1,28,28])).cuda()
    
    #Get the 2D coordinates from the data from the Encoder
    coor = Q(testimg, 1)
    coorformat0 = coor[0][0][0].data.cpu().numpy()
    coorformat1 = coor[0][0][1].data.cpu().numpy()
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

testimg = Variable(testimg.view([1,1,28,28])).cuda()
rec = Q(testimg, 1)
rec = P(rec,1)
rec = rec[0]

plt.clf()
plt.imshow(testimg.data.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitConv.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy().reshape([28,28]), cmap='gray')
plt.savefig('./reconConv.png',dpi = 300)

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg.view([1,1,28,28])).cuda()
rec = Q(testimg, 1)
rec = P(rec,1)
rec = rec[0]

plt.clf()
plt.imshow(testimg.data.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitConv1.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy().reshape([28,28]), cmap='gray')
plt.savefig('./reconConv1.png',dpi = 300)

pair = iter(test_loader).next()
testimg = pair[0]
label = pair[1]
testimg = Variable(testimg.view([1,1,28,28])).cuda()
rec = Q(testimg, 1)
rec = P(rec,1)
rec = rec[0]

plt.clf()
plt.imshow(testimg.data.cpu().numpy()[0][0][:], cmap='gray')
plt.savefig('./digitConv2.png',dpi = 300)

plt.clf()
plt.imshow(rec.data.cpu().numpy().reshape([28,28]), cmap='gray')
plt.savefig('./reconConv2.png',dpi = 300)

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
plt.legend()

print('Len list')
print(len(List0x))
print(len(List1x))
print(len(List2x))
print(len(List3x))
print(len(List4x))
print(len(List5x))

plt.savefig('./ConvAE.png',dpi = 300)