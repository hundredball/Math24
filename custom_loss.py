#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 23:07:24 2020

@author: hundredball
"""

import torch

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def gradient_penalty_loss(grad_output, y_pred, averaged_samples, gradient_penalty_weight):
    
    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=y_pred, inputs=averaged_samples,
                           grad_outputs=grad_output, create_graph=True, retain_graph=True)[0]
    
    # Compute the L2 norm
    gradient_L2_norm = torch.sqrt( 1e-8+torch.sum(gradients**2, dim=(1,2,3)))
    
    # Compute lambda * (1-||grad||)^2 for each single sample
    gradient_penalty = gradient_penalty_weight * ((1-gradient_L2_norm)**2)
    
    return torch.mean(gradient_penalty)