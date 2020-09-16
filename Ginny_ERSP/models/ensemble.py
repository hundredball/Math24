#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:36:19 2020

@author: hundredball
"""

__all__ = ['average']

import torch
import torch.nn as nn

class Average(nn.Module):
    
    def __init__(self, models):
        super(Average, self).__init__()
        
        self.num_models = len(models)
        
        for i in range(self.num_models):
            setattr(self, 'model'%(i), models[i])
        
    def forward(self, x):
        
        # Convolution for each time step
        for i in range(self.num_models):
            if i == 0:
                output = getattr(self, 'model'%(i))(x)
            else:
                output += getattr(self, 'model'%(i))(x)
            
        output /= self.num_models
        
        return output
    
def average(models):
    model = Average(models)
    return model
    