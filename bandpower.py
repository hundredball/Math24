#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:52:20 2020

@author: hundredball
"""

import numpy as np
from scipy import signal
from scipy.integrate import simps

import dataloader as dl


def get_bandpower(data):
    '''
    Calculate bandpower of theta, alpha, beta

    Parameters
    ----------
    data : numpy 3D array (i,j,k) 
        i : example
        j : channel
        k : sample

    Returns
    -------
    powers : numpy 3D array (i,j,k)
        i : example
        j : channel
        k : band

    '''
    fs = 256
    
    # Define window length
    win = 0.5*fs
    freqs, psd = signal.welch(data, fs, nperseg=win, noverlap=win/2)
    
    # Define lower and upper bounds
    low, high = [4,7,13], [7,13,30]
    
    # Find intersecting values in frequency vector
    idx = np.logical_and(freqs[:,np.newaxis] >= low, freqs[:,np.newaxis] <= high)
    idx = idx.T   # (65,3)->(3,65)
    
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1/0.5 = 2
    
    # Compute the absolute power by approximating the area under the curve
    print('Calculating the bandpower of time-series data...')
    powers = np.zeros((data.shape[0],data.shape[1],3))
    for i in range(3):
        idx_power = idx[i,:]
        powers[:,:,i] = simps(psd[:,:,idx_power], dx=freq_res)

    return powers


if __name__ == '__main__':
    
    X, Y, channels = dl.read_data([1,2,3])
    powers = get_bandpower(X)
    
    print('Absolute theta power: %.3f uV^2' % powers[525, 12, 0])
    print('Absolute alpha power: %.3f uV^2' % powers[525, 12, 1])
    print('Absolute beta power: %.3f uV^2' % powers[525, 12, 2])