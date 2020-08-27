#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:28:17 2020

@author: hundredball
"""


import torch
import numpy as np

__all__ = ['eegnet']

class EEGNet(torch.nn.Module):
    def __init__(self, activation, size_2d, F1=16, D=2, F2=32, KS1=64, KS2=16):
        super(EEGNet, self).__init__()
        self.firstconv = torch.nn.Sequential(
            torch.nn.Conv2d(1, F1, kernel_size=(1,KS1), stride=(1,1), padding=(0,KS1//2), bias=False),
            torch.nn.BatchNorm2d(F1))
        self.depthwiseConv = torch.nn.Sequential(
            torch.nn.Conv2d(F1,D*F1, kernel_size=(2,1), stride=(1,1), groups=F1, bias=False),
            torch.nn.BatchNorm2d(D*F1),
            activation,
            torch.nn.AvgPool2d(kernel_size=(1,4), stride=(1,4)),
            torch.nn.Dropout(p=0.5))
        self.separableConv = torch.nn.Sequential(
            # depthwise separable convolution
            torch.nn.Conv2d(D*F1,D*F1, kernel_size=(1,KS2), stride=(1,1), padding=(0,KS2//2), groups=D*F1, bias=False),
            torch.nn.Conv2d(D*F1,F2, kernel_size=(1,1), stride=(1,1), bias=False),
            # convolution on the lecture
#            torch.nn.Conv2d(D*F1, F2, kernel_size=(1,KS2), stride=(1,1), padding=(0,7), bias=False),
            torch.nn.BatchNorm2d(F2),
            activation,
            torch.nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)),
            torch.nn.Dropout(p=0.5))
        self.classify = torch.nn.Linear(in_features=(size_2d[0]-1)*(size_2d[1]//4//8)*F2,out_features=1)
        
#        self.classify = torch.nn.Linear(in_features=224,out_features=3)
            
    def forward(self,x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        
        # flatten the data
        x = torch.reshape(x, (x.size()[0], np.prod( x.size()[1:]) ))
        x = self.classify(x)
        
        # flatten the output
        x = x.flatten()
        
        return x
    
def eegnet(activation, size_2d):
    model = EEGNet(activation, size_2d)
    return model