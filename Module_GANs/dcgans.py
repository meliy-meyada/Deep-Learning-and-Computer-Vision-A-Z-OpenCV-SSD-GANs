#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:41:09 2022

@author: meyada
"""

# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


#Setting some hyperparameters
batchSize = 64
imageSize = 64

# Create the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)


# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
# Defining the generators
class G(nn.Module):
    
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias= False),
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias= False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias= False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias= False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias= False),
            nn.Tanh()
            )
    
    def forward(self, input):
        output = self.main(input)
        return output
    
# Create the generator
netG = G()
netG.apply(weights_init)


# Defining the discriminator
class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias= False),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Conv2d(64, 128, 4, 2, 1, bias= False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Conv2d(128, 256, 4, 2, 1, bias= False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Conv2d(256, 512, 4, 2, 1, bias= False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Conv2d(512, 1, 4, 1, 0, bias= False),
            nn.Sigmoid()
            )
        
        
        
    
    
    
    
    
        
