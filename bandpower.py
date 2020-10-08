#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:52:20 2020

@author: hundredball
"""

import numpy as np
import pandas as pd
import pickle
from scipy import signal
from scipy import interpolate
from scipy.integrate import simps

import raw_dataloader as dl
import add_features

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

def STFT(data, SLs, channels, low, high, savePath=None):
    '''
    Adopt STFT to data to get ERSP, and finally save it

    Parameters
    ----------
    data : numpy 3D array (i,j,k) 
        i : example
        j : channel
        k : sample
    SLs : numpy 1D array
        solution latency of each data
    channels : numpy 1D array
        Number of channel file
    low : int
        Lower frequency bound
    high : int
        Upper frequency bound

    Returns
    -------
    new_f : numpy 1d array
        Frequency steps of ERSP
    t : numpy 1d array
        Time steps of ERSP
    Zxx : numpy 4d array (epoch, channel, freq, time)
        Event related spectral perturbation

    '''

    assert isinstance(data, np.ndarray) and data.ndim == 3
    assert isinstance(SLs, np.ndarray) and SLs.ndim == 1
    assert isinstance(channels, np.ndarray) and channels.ndim == 1
    assert isinstance(low, int) and isinstance(high, int)
    assert (high >= low >= 0)
    
    print('--- STFT ---\n')
    f, t, Zxx = signal.stft(data, fs, nperseg = 512, noverlap = 512-3, axis=2)
    
    '''
    # Interpolate to make 114 steps for 2-30 Hz
    print('Interpolate')
    interp = interpolate.interp1d(f, Zxx, axis=2)
    new_f = np.linspace(0, f[-1], 2*(f.shape[0]+4))
    Zxx = interp(new_f)
    '''
    
    new_f = f
    
    # Remove imaginary part
    Zxx = abs(Zxx)
    
    # Transform to dB power
    print('Transform to dB')
    Zxx = 10*np.log10(Zxx)
    
    # Find intersecting values in frequency vector
    idx = np.logical_and(new_f >= low, new_f <= high)
    print('%d - %d Hz: %d steps'%(low, high, np.sum(idx)))
    
    # Select 2-30 HZ frequency components
    new_f = new_f[idx]
    Zxx = Zxx[:,:,idx]
    
    # Save to pickle file
    if savePath is not None:
        dict_ERSP = {'freq':new_f, 't':t, 'ERSP':Zxx, 'SLs':SLs}
        with open(savePath, 'wb') as fp:
            pickle.dump(dict_ERSP, fp)
    
    return new_f, t, Zxx

if __name__ == '__main__':
    
    channel_limit = 21
    '''
    # Save data for all subject
    X, Y_reg, channels = dl.read_data([1,2,3], list(range(11)), channel_limit=channel_limit, rm_baseline=True)
    freq, t, Zxx = STFT(X, Y_reg, channels, 2, 30, savePath='./raw_data/ERSP_from_raw_100_channel%d.data'%(channel_limit))
    
    print('Calculate conditional entropy...')
    _ = add_features.calculate_CE(X, './raw_data/CE_sub100_channel%d.data'%(channel_limit))
    '''
    
    
    # Save data for each subject
    for i_sub in range(11):
        print('--- Subject %d ---'%(i_sub))
        
        X, Y_reg, channels = dl.read_data([1,2,3], [i_sub], channel_limit=channel_limit, rm_baseline=True)
    
        # Adopt STFT and save file
        freq, t, Zxx  = STFT(X, Y_reg, channels, 2, 30, savePath='./raw_data/ERSP_from_raw_%s_channel%d.data'%(str(i_sub), channel_limit))
    
        print('Calculate conditional entropy...')
        _ = add_features.calculate_CE(X, './raw_data/CE_sub%d_channel%d.data'%(i_sub, channel_limit))
    