#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:36:19 2020

@author: hundredball
"""

__all__ = ['average', 'combinedfc']

import torch
import torch.nn as nn

class Average(nn.Module):
    
    def __init__(self, models):
        super(Average, self).__init__()
        
        self.num_models = len(models)
        
        for i in range(self.num_models):
            setattr(self, 'model%d'%(i), models[i])
        
    def forward(self, x):
        
        # Convolution for each time step
        for i in range(self.num_models):
            if i == 0:
                output = getattr(self, 'model%d'%(i))(x)
            else:
                output += getattr(self, 'model%d'%(i))(x)
            
        output /= self.num_models
        
        return output
    
class CombinedFC(nn.Module):
    
    def __init__(self, models):
        super(CombinedFC, self).__init__()
        
        self.num_models = len(models)
        
        for i in range(self.num_models):
            setattr(self, 'model%d'%(i), models[i])
        
        self.fc = nn.Sequential(
            nn.Linear(self.num_models, 3),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.ReLU()
            )
        
        # Set all the weights absolute
        with torch.no_grad():
            self.fc[0].weight = nn.Parameter(torch.abs(self.fc[0].weight.data))
            self.fc[2].weight = nn.Parameter(torch.abs(self.fc[2].weight.data))
            
        self.fc[0].weight.requires_grad = True
        self.fc[2].weight.requires_grad = True
        
    def forward(self, x):
        
        outputs = [getattr(self, 'model%d'%(i))(x) for i in range(self.num_models)]
        
        outputs = torch.stack(outputs)

        # (features, batch) -> (batch, features)
        outputs = torch.transpose(outputs, 0, 1)
        
        outputs = self.fc(outputs)
        outputs = outputs.flatten()
        
        return outputs
        
def combinedfc(models):
    return CombinedFC(models)
    
def average(models):
    model = Average(models)
    return model
    