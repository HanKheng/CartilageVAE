# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:14:00 2022

@author: Han Kheng Teoh
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

class Conv1DDecoder(nn.Module):
    '''
    This decoder module make use of 3 transpose convolutional layers and fully connected layers to reconstruct time series 
    Assumption made: length of time series = 750, n_channel = 3
    
    '''
    def __init__(self,n_latent,n_channel =3):
        super(Conv1DDecoder, self).__init__()
        self.latent_dim = n_latent
        self.n_channel = n_channel
        # decoder layer 
        self.fc4 = nn.Linear(self.latent_dim,128)
        self.fc5 = nn.Linear(128,256)
        self.fc6 = nn.Linear(256,1504)
        self.convt0 = nn.ConvTranspose1d(in_channels =16, out_channels =8, kernel_size = 3, stride = 2, padding =1, output_padding=1)
        self.bn4 = nn.BatchNorm1d(16)
        self.convt1 = nn.ConvTranspose1d(in_channels =8, out_channels =4, kernel_size = 3, stride = 2, padding =1, output_padding=0)
        self.bn5 = nn.BatchNorm1d(8)
        self.convt2 = nn.ConvTranspose1d(in_channels =4, out_channels =self.n_channel, kernel_size = 3, stride = 2, padding =1, output_padding=1)
        self.bn6 = nn.BatchNorm1d(4)
        

    def forward(self, z):
        batch_size = z.size(0)
        z = torch.tanh(self.fc4(z))
        z = torch.tanh(self.fc5(z))
        z = torch.tanh(self.fc6(z))
        
        z = z.reshape([batch_size,16,-1])
        z = torch.tanh(self.convt0(self.bn4(z)))
        z = torch.tanh(self.convt1(self.bn5(z)))
        z = self.convt2(self.bn6(z))
        
        return z.reshape(batch_size,self.n_channel,-1)
