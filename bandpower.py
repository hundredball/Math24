#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:52:20 2020

@author: hundredball
"""

import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.integrate import simps

import dataloader as dl

fs = 256

def get_bandpower(data, low = [4,7,13], high=[7,13,30]):
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
    assert isinstance(data, np.ndarray)
    assert hasattr(low, '__iter__') and hasattr(high, '__iter__')
    assert all((high[i]>=low[i]) for i in range(len(high)))
    
    # Define window length
    win = 0.5*fs
    freqs, psd = signal.welch(data, fs, nperseg=win, noverlap=win/2)
    
    # Find intersecting values in frequency vector
    idx = np.logical_and(freqs[:,np.newaxis] >= low, freqs[:,np.newaxis] <= high)
    idx = idx.T   # (65,3)->(3,65)
    
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1/0.5 = 2
    
    # Compute the absolute power by approximating the area under the curve
    print('Calculating the bandpower of time-series data...')
    powers = np.zeros((data.shape[0],data.shape[1],len(high)))
    for i in range(len(high)):
        idx_power = idx[i,:]
        powers[:,:,i] = simps(psd[:,:,idx_power], dx=freq_res)

    return powers

def STFT(data, SLs, low, high):
    '''
    Adopt STFT to data

    Parameters
    ----------
    data : numpy 3D array (i,j,k) 
        i : example
        j : channel
        k : sample
    SLs : numpy 1D array
        solution latency of each data
    low : int
        Lower frequency bound
    high : int
        Upper frequency bound

    Returns
    -------
    None.

    '''

    assert isinstance(data, np.ndarray)
    assert isinstance(SLs, np.ndarray)
    assert isinstance(low, int) and isinstance(high, int)
    assert (high >= low >= 0)
    
    f, t, Zxx = signal.stft(data, fs, nperseg = 512, noverlap = 512-3, axis=2)
    
    # # Interpolate to make 114 steps for 2-30 Hz
    # interp = interpolate.interp1d(f, Zxx, axis=2)
    # new_f = np.linspace(0, f[-1], 2*(f.shape[0]+4))
    # Zxx = interp(new_f)
    new_f = f
    
    # Average estimates accross time dimension
    Zxx = np.mean(abs(Zxx), axis=3)
    
    # Transform to dB power
    print('Transform to dB')
    Zxx = 10*np.log10(Zxx)
    
    # Subtract the base spectrum (trials <= 5s)
    base = np.mean(Zxx[np.where(SLs<=5)[0], :, :], axis=0)
    Zxx = Zxx - base[np.newaxis, :, :]
    
    # Transform back 
    # Zxx = 10 ** (Zxx/10)
    
    # Find intersecting values in frequency vector
    idx = np.logical_and(new_f >= low, new_f <= high)
    print('%d - %d Hz: %d steps'%(low, high, np.sum(idx)))
    
    # Select 2-30 HZ frequency components
    new_f = new_f[idx]
    Zxx = Zxx[:,:,idx]
    
    return new_f, t, Zxx

if __name__ == '__main__':
    
    X, Y_class, Y_reg, channels = dl.read_data([2], list(range(11)), 'class')
    
    # Test get_bandpower
    powers = get_bandpower(X, [2], [30])
    print(powers.shape)
    
    # print('Absolute theta power: %.3f uV^2' % powers[525, 12, 0])
    # print('Absolute alpha power: %.3f uV^2' % powers[525, 12, 1])
    # print('Absolute beta power: %.3f uV^2' % powers[525, 12, 2])
    
    # Test STFT
    freq, t, Zxx  = STFT(X, Y_reg, 2, 30)