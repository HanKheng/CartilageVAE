# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:34:38 2022

@author: Han Kheng Teoh
"""
import numpy as np
from torch.utils.data import Dataset, DataLoader

def cell_data_loaders(dataset,split=0.8,batch_size=64, shuffle=True, num_workers=4):
        partition={}
        assert(split > 0.0 and split <= 1.0)
        if shuffle:
            np.random.seed(42)
            perm = np.random.permutation(dataset.shape[0])
            dataset = dataset[perm,:,:]
            np.random.seed(None)
        # Split.
        i = int(round(split * dataset.shape[0]))
        partition['train']= dataset[:i,:,:]
        partition['test']= dataset[i:,:,:] 
        
        train_dataloader = DataLoader(partition['train'], batch_size=batch_size, \
            shuffle=shuffle, num_workers=num_workers)
        if len(partition['test'])==0:
            return {'train':train_dataloader, 'test':None}
        test_dataloader = DataLoader(partition['test'], batch_size=batch_size, \
            shuffle=shuffle, num_workers=num_workers)
        return {'train':train_dataloader, 'test':test_dataloader}
    