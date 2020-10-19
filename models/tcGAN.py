#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:01:27 2020

@author: hundredball

Target-conditioned GAN
"""

__all__ = ['generator', 'discregressor']

import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, num_channels=12):
        
        super(Generator, self).__init__()
        self.num_channels = num_channels
        
        self.layer1 = nn.Sequential(
            nn.Linear(120, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128*4*34),
            nn.BatchNorm1d(128*4*34),
            nn.LeakyReLU()
            )
        
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
            )
        
        self.deconv = nn.ConvTranspose2d(64, 128, (4,4), stride=(2,2))
        # Set linear weights
        with torch.no_grad():
            self.deconv.weight = nn.Parameter(torch.from_numpy(linear_kernel(64, 128, 2)))
        self.deconv.requires_grad = True

        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=(3,3), padding=(1,1))
            )

    def forward(self, x):
        '''
        Multiply -> layer1 -> 1. Upsampling -> layer2 -> 2. Upsampling -> layer3

        '''
        # Multiply target/10 with noise
        scale = (x[:,-1]/10).reshape((-1,1))
        x = x[:,:-1]*scale

        # layer1
        x = self.layer1(x)
        
        # 1. Upsampling
        x = x.reshape((x.shape[0],128,4,34))   #(Filter, Channel, Time)
        x = nn.functional.interpolate(x, scale_factor=(2,2), mode='bilinear', align_corners=True)
        
        # layer2
        x = self.layer2(x)
        
        # 2. Upsampling
        x = self.deconv(x)
        x = x[:,:,3:15, 5:133]     # Crop the center region
        
        # layer3
        x = self.layer3(x)
        
        # Remove DC shift
        mean_ = torch.mean(torch.mean(x, dim=3), dim=0)
        x = x - torch.unsqueeze(torch.unsqueeze(mean_, dim=0), dim=3)
        
        return x
    
class DiscRegressor(nn.Module):
    
    def __init__(self):
        super(DiscRegressor, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.LeakyReLU(0.2)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128*3*32,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,2)
            )
        
    def forward(self, x):
        '''
        conv -> fc -> validity, target

        '''
        x = self.conv(x)
        x = x.reshape((x.shape[0],-1))
        x = self.fc(x)      # (validity, target)
        
        return x
        
    
def linear_kernel(in_channels, out_channels, stride): # stride = 2; num_channels = 1

    filter_size = (2 * stride - stride % 2)
    # Create linear weights in numpy array
    linear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            linear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    
    weights = np.zeros((in_channels, out_channels, filter_size, filter_size))
    for i in range(in_channels):
        for j in range(out_channels):
            weights[i, j, :, :] = linear_kernel
    weights = weights.astype('float32')
    
    return weights
        

def generator():
    return Generator()

def discregressor():
    return DiscRegressor()

if __name__ == '__main__':
    
    # Test G
    noise = torch.rand(2, 120)
    target = torch.rand(2, 1)
    input = torch.cat((noise, target), dim=1)
    gen = generator()
    
    result = gen(input)
    print(result.shape)
    
    # Test D
    signal = torch.rand(32, 1, 12, 128)
    
    discriminator = discregressor()
    result = discriminator(signal)
    print(result.shape)
        