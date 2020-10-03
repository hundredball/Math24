#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:28:17 2020

@author: hundredball
"""


import torch
import torch.nn as nn
import numpy as np

__all__ = ['eegnet', 'eegnet_trans_power', 'eegnet_trans_signal']

class EEGNet(torch.nn.Module):
    def __init__(self, activation, size_2d, fs=256, F1=16, D=2, F2=32, KS1=64, KS2=16):
        super(EEGNet, self).__init__()
        KS1 = fs//2
        KS2 = fs//4
        
        self.firstconv = torch.nn.Sequential(
            torch.nn.Conv2d(1, F1, kernel_size=(1,KS1), stride=(1,1), padding=(0,KS1//2), bias=False),
            torch.nn.BatchNorm2d(F1))
        self.depthwiseConv = torch.nn.Sequential(
            torch.nn.Conv2d(F1,D*F1, kernel_size=(D,1), stride=(1,1), groups=F1, bias=False),
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
        self.classify = torch.nn.Linear(in_features=(size_2d[0]-(D-1))*(size_2d[1]//4//8)*F2,out_features=1)
        
#        self.classify = torch.nn.Linear(in_features=224,out_features=3)
            
    def forward(self,x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        
        # flatten the data
        x = x.reshape((x.shape[0], -1))
        x = self.classify(x)
        
        # flatten the output
        x = x.flatten()
        
        return x
    
class EEGNet_trans_power(nn.Module):
    
    def __init__(self, in_channels, in_features):
        super(EEGNet_trans_power, self).__init__()
        self.num_dilations = in_channels-1
        self.activation = nn.ReLU()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,10), stride=(1,1), padding = (0,10//2), bias=False),
            nn.BatchNorm2d(16)
            )
        
        for i in range(self.num_dilations):
            setattr(self, 'depthconv%d'%(i), self.getDepthconv(i+1))
            setattr(self, 'separableconv%d'%(i), self.getSeparableconv(32))
        in_features = (in_features - 4)//4 + 1                                  # After depthconv
        in_features = (in_features - 4)//4 + 1                                  # After separableconv
        in_features = int(32 * (1+self.num_dilations)*(self.num_dilations)/2 * in_features)    # After reshape
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1)
            )
        
    def forward(self, x):
        
        # Convolve over features
        x = self.firstconv(x)
        
        for i in range(self.num_dilations):
            # Depthwise convolution (convolve over eeg channels)
            output = getattr(self, 'depthconv%d'%(i))(x)
            # Separable convolution 
            output = getattr(self, 'separableconv%d'%(i))(output)
            
            # (Batch, Filter, Channel, Freqs) -> (Batch, Filter*Channel*Freqs)
            output = output.reshape((x.shape[0], -1))
            if i == 0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=1)

        # Regression
        outputs = self.fc(outputs)
        outputs = outputs.flatten()
        
        return outputs
        
            
    def getSeparableconv(self, in_filters):
        return nn.Sequential(
            torch.nn.Conv2d(in_filters,in_filters, kernel_size=(1,4), stride=(1,1), groups=in_filters, padding=(0,4//2), bias=False),
            torch.nn.Conv2d(in_filters,in_filters, kernel_size=(1,1), stride=(1,1), bias=False),
            torch.nn.BatchNorm2d(32),
            self.activation,
            torch.nn.AvgPool2d(kernel_size=(1,4), stride=(1,4)),
            torch.nn.Dropout(p=0.5)
            )
                
    def getDepthconv(self, depth_dilation):
        return nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), dilation=(depth_dilation,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4)),
            nn.Dropout(p=0.5)
            ) 

class EEGNet_trans_signal(nn.Module):
    def __init__(self, in_channels, in_features, fs=256):
        super(EEGNet_trans_signal, self).__init__()
        self.num_dilations = in_channels-1
        self.activation = nn.ReLU()
        self.KS1 = fs//2
        self.KS2 = fs//4
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,self.KS1), stride=(1,1), padding = (0,self.KS1//2), bias=False),
            nn.BatchNorm2d(16)
            )
        
        for i in range(self.num_dilations):
            setattr(self, 'depthconv%d'%(i), self.getDepthconv(i+1))
            setattr(self, 'separableconv%d'%(i), self.getSeparableconv(32))
        in_features = (in_features - 4)//4 + 1                                  # After depthconv
        in_features = (in_features - 8)//8 + 1                                  # After separableconv
        in_features = int(32 * (1+self.num_dilations)*(self.num_dilations)/2 * in_features)    # After reshape
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            )
        
    def forward(self, x):
        
        # Convolve over features
        x = self.firstconv(x)
        
        for i in range(self.num_dilations):
            # Depthwise convolution (convolve over eeg channels)
            output = getattr(self, 'depthconv%d'%(i))(x)
            # Separable convolution 
            output = getattr(self, 'separableconv%d'%(i))(output)
            
            # (Batch, Filter, Channel, Freqs) -> (Batch, Filter*Channel*Freqs)
            output = output.reshape((x.shape[0], -1))
            if i == 0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=1)

        # Regression
        outputs = self.fc(outputs)
        outputs = outputs.flatten()
        
        return outputs
        
            
    def getSeparableconv(self, in_filters):
        return nn.Sequential(
            torch.nn.Conv2d(in_filters,in_filters, kernel_size=(1,self.KS2), stride=(1,1), groups=in_filters, padding=(0,self.KS2//2), bias=False),
            torch.nn.Conv2d(in_filters,in_filters, kernel_size=(1,1), stride=(1,1), bias=False),
            torch.nn.BatchNorm2d(32),
            self.activation,
            torch.nn.AvgPool2d(kernel_size=(1,4), stride=(1,4)),
            torch.nn.Dropout(p=0.5)
            )
                
    def getDepthconv(self, depth_dilation):
        return nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), dilation=(depth_dilation,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)),
            nn.Dropout(p=0.5)
            ) 
    
def eegnet_trans_signal(in_channels, in_features):
    return EEGNet_trans_signal(in_channels, in_features)
    
def eegnet_trans_power(in_channels, in_features):
    return EEGNet_trans_power(in_channels, in_features)
    
def eegnet(activation, size_2d, fs=256, F1=16, D=2, F2=32, KS1=64, KS2=16):
    model = EEGNet(activation, size_2d, fs, F1, D, F2, KS1, KS2)
    return model