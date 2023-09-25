# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:27:58 2022

@author: Han Kheng Teoh 

Encoder modules for cell analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DEncoder(nn.Module):
    '''
    This encoder module make use of 3 convolutional layers follows by fully connected layers to give gaussian params
    Assumption made: length of time series = 750, n_channel = 3
    
    '''
    def __init__(self,n_latent,n_channel =3):
        super(Conv1DEncoder, self).__init__()
        self.latent_dim = n_latent
        self.n_channel = n_channel
        # encoder layer 
        self.conv1 = nn.Conv1d(in_channels =self.n_channel, out_channels =4, kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm1d(self.n_channel)
        self.conv2 = nn.Conv1d(in_channels =4, out_channels =8, kernel_size = 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm1d(4)
        self.conv3 = nn.Conv1d(in_channels =8, out_channels =16, kernel_size = 3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm1d(8)
        
        self.fc1 = nn.Linear(1504,256)
        
        self.fc21 = nn.Linear(256,128)
        self.fc23 = nn.Linear(256,128)
        
        self.fc31 = nn.Linear(128,self.latent_dim)
        self.fc33 = nn.Linear(128,self.latent_dim)
        

    def forward(self, x):
        batch_size = x.size(0)
        #x=x.unsqueeze(1)
        x=torch.tanh(self.conv1(self.bn1(x)))
        x=torch.tanh(self.conv2(self.bn2(x)))
        x=torch.tanh(self.conv3(self.bn3(x)))

        x = x.view(batch_size,-1)
        x = torch.tanh(self.fc1(x))
        
        mu = torch.tanh(self.fc21(x))
        mu = self.fc31(mu)
        
        d = torch.tanh(self.fc23(x))
        d = torch.exp(self.fc33(d))+1e-3

        return mu, d

