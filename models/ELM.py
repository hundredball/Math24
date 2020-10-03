#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:42:09 2020

@author: hundredball
"""


__all__ = ['elm']

import numpy as np
from scipy.linalg import pinv2
    
    
class ELM:
    
    def __init__(self, input_size, hidden_size):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights
        self.input_weights = np.random.normal(size=[input_size, hidden_size])
        self.biases = np.random.normal(size=[hidden_size])
    
    def fit(self, X_train, Y_train):
        self.output_weights = np.dot(pinv2(self.hidden_nodes(X_train)), Y_train)
        
    def predict(self, X):
        out = self.hidden_nodes(X)
        out = np.dot(out, self.output_weights)
        return out
    
    def hidden_nodes(self, X):
        G = np.dot(X, self.input_weights)
        G = G + self.biases
        H = self.relu(G)
        return H
    
    def relu(self, x):
        return np.maximum(x,0,x)
    
def elm(input_size, hidden_size):
    return ELM(input_size, hidden_size)