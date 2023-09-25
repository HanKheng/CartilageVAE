# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:26:52 2022

@author: Han Kheng 

Wrapper functions 

"""

import torch
import torch.nn as nn
import os
from torch.optim import Adam
import numpy as np
from torch.distributions import LowRankMultivariateNormal
import matplotlib.pyplot as plt
import random
from IPython import display
   

class CONV1D_VAE(nn.Module):
    '''
    Wrapper function for VAE
    '''
    def __init__(self, encoder,decoder, latent_dim :int =8 ,save_dir='', lr: float =1e-3,model_precision: float =1.0, device_name="auto"):
        super(CONV1D_VAE, self).__init__()
        
        self.save_dir=save_dir
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        assert device_name != "cuda" or torch.cuda.is_available()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"  
        self.device = torch.device(device_name)    
        
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.model_precision=model_precision
        self.lr=lr
        self.optimizer=Adam(self.parameters(),lr=self.lr)
        self.epoch=0;
        self.loss={'train':{},'test':{}}
        self.to(self.device)
       

    def forward(self, x):
        batch_size = x.shape[0]
        offset = np.zeros(batch_size)
        x_translation = x.clone()
        
        # in order to encourage robustness of the inference network
        # randomly translate the cell data before doing inference
        offset = torch.normal(0, 0.1, size=(batch_size, x.shape[1],1)).to(self.device)
        x_translation += offset
    
        mu,  d= self.encoder(x_translation)
        latent_dist = LowRankMultivariateNormal(mu,0*mu.unsqueeze(-1),d)
        z = latent_dist.rsample()
        x_rec = self.decoder(z)       
        # E_{q(z|x)} p(x|z)
        # reconstruction error
        pxz_term = -0.5* x.shape[1] * (np.log(2*np.pi/self.model_precision))
        l2s = torch.sum(torch.pow(x_translation.view(1,-1) - x_rec.view(1,-1), 2), dim=1)
        
        pxz_term = pxz_term - 0.5 * self.model_precision * torch.sum(l2s) 
        # E_{q(z|x)} log(p(z))
        # KL divergence
        elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.latent_dim * np.log(2*np.pi))
        elbo = elbo + pxz_term
        # H[q(z|x)]
        elbo = elbo + torch.sum(latent_dist.entropy()) 
        return -elbo ,mu,d,x_rec
    
    def train_epoch(self, train_loader):
        self.train()
        train_loss = 0.0

        for _, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            loss,_,_,_ = self.forward(data)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            
        train_loss /= len(train_loader.dataset)
        
        print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
                train_loss))
        self.epoch += 1
        return train_loss


    def test_epoch(self, test_loader):

        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                
                data = data.to(self.device)
                loss,mu,d,x_rec = self.forward(data)
                test_loss += loss.item()

        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        ind_d=random.randint(0,data.shape[0]-1)
        plt.plot(data.detach().cpu().numpy()[ind_d,:].T,label='input')
        plt.plot(x_rec.detach().cpu().numpy().squeeze()[ind_d,:].T,'--',label='reconstructed')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.loss['train'].keys(),self.loss['train'].values(),label='train')
        plt.plot(self.loss['test'].keys(),self.loss['test'].values(),label='test')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        test_loss /= len(test_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss


    def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10):

        print("="*40)
        print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
        print("Training set:", len(loaders['train'].dataset))
        print("Test set:", len(loaders['test'].dataset))
        print("="*40)
        # For some number of epochs...
        for epoch in range(self.epoch, self.epoch+epochs):
            # Run through the training data and record a loss.
            loss = self.train_epoch(loaders['train'])
            self.loss['train'][epoch] = loss
            # Run through the test data and record a loss.
            if (test_freq is not None) and (epoch % test_freq == 0):
                loss = self.test_epoch(loaders['test'])
                self.loss['test'][epoch] = loss
   

