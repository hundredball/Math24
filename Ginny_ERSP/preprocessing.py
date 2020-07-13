#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:28:06 2020

@author: hundredball
"""

import numpy as np
import dataloader

def standardize(ERSP, tmp):
    '''
    Average over time and subtract the base spectrum

    Parameters
    ----------
    ERSP : 4d numpy array (epoch, channel, freq_step, time_step)
        ERSP of all trials
    tmp : 2d numpy array (epoch, time_periods)
        time_periods include time points of fixation, cue, end
    Returns
    -------
    ERSP : 3d numpy array (epoch, channel, freq_step)
        ERSP of all trials
    SLs : 1d numpy array (epoch)
        solution latency of each trials

    '''
    assert isinstance(ERSP, np.ndarray) and len(ERSP.shape)==4
    assert isinstance(tmp, np.ndarray) and len(tmp.shape)==2
    
    # Average over time
    ERSP = np.mean(ERSP, axis=3)
    
    # Subtract the base spectrum (trials <= 5s)
    SLs = tmp[:, 2]
    base = np.mean(ERSP[np.where(SLs<=5)[0], :, :], axis=0)
    ERSP = ERSP - base[np.newaxis, :, :]
    
    return ERSP, SLs

def bandpower(ERSP, freqs, low, high):
    '''
    Get bandpower from ERSP

    Parameters
    ----------
    ERSP : 3d numpy array (epoch, channel, freq_step)
        ERSP of all trials
    freqs : 1d numpy array
        frequency step of ERSP
    low : int
        lower bound
    high : int
        upper bound

    Returns
    -------
    mean_ERSP : 2d numpy array (epoch, channel)
        mean ERSP

    '''
    assert isinstance(ERSP, np.ndarray) and len(ERSP.shape)==3
    assert isinstance(freqs, np.ndarray) and len(freqs.shape)==1
    assert isinstance(low, int) and isinstance(high, int)
    assert high>=low
    
    index_freq = np.logical_and(freqs>low, freqs<high)
    mean_ERSP = np.mean(ERSP[:,:,index_freq], axis=2)
    
    return mean_ERSP

def trimMean(ERSP_all, SLs, freqs):
    '''
    Generate new trials by averaging old trials

    Parameters
    ----------
    ERSP_all : 2d numpy array (epoch, channel)
        mean ERSP
    SLs : 1d numpy array (epoch)
        solution latency of each trials
    freqs : 1d numpy array (freq_step)
        frequency step of ERSP

    Returns
    -------
    new_ERSP : 2d numpy array (epoch, channel)
        new ERSP
    new_SLs : 1d numpy array
        new solution latency

    '''
    assert isinstance(ERSP_all, np.ndarray) and len(ERSP_all.shape)==3
    assert isinstance(SLs, np.ndarray) and len(SLs.shape)==1
    assert isinstance(freqs, np.ndarray) and len(freqs.shape)==1
    
    num_channel = ERSP_all.shape[1]

    # Take mean power between 10-15 Hz
    mean_ERSP = bandpower(ERSP_all, freqs, 10, 15)

    # Take mean trial power with specific SL
    windowsize = 10
    start_SL = np.arange(0, 65-(windowsize-1)+0.5, 0.5)
    end_SL = np.arange(windowsize-1, 65+0.5, 0.5)
    first_trial = True

    for i_win in range(len(start_SL)):
        index_SL = np.logical_and(SLs>=start_SL[i_win], SLs<end_SL[i_win])
        if np.sum(index_SL) != 0:
            select_ERSP = mean_ERSP[index_SL,:]
            select_SL = SLs[index_SL]
            tmp_ERSP = np.zeros((1, num_channel))
            for i_channel in range(num_channel):

                # Sort ERSP in increasing order
                sorted_index = np.argsort(select_ERSP[:,i_channel])
                sorted_ERSP = select_ERSP[sorted_index,i_channel]
                sorted_SL = select_SL[sorted_index]
                num_trial = sorted_ERSP.shape[0]

                # Take mean trial power between 25-75%
                tmp_ERSP[0, i_channel] = np.mean( sorted_ERSP[round(num_trial*0.25):round(num_trial*0.75)] )
                new_SL = np.mean(sorted_SL[round(num_trial*0.25):round(num_trial*0.75)])
                
            if first_trial:
                new_ERSP = tmp_ERSP.copy()
                new_SLs = new_SL
                first_trial = False
            else:
                new_ERSP = np.vstack((new_ERSP, tmp_ERSP))
                new_SLs = np.vstack((new_SLs, new_SL))

    new_SLs = new_SLs.flatten()
    return new_ERSP, new_SLs 

def make_target(SLs, threshold=None):
    '''
    Classify trials into slow and fast group and concatenate label and SLs

    Parameters
    ----------
    SLs : 1d numpy array
        Solution latency
    threshold : float, optional
        Threshold of solution latency. The default is None.

    Returns
    -------
    Y : 2d numpy array (epoch, type)
        First column: Classification label
        Second column: Solution latency

    '''
    assert isinstance(SLs, np.ndarray) and len(SLs.shape)==1
    
    if threshold is None:
        threshold = np.mean(SLs)
    else:
        assert isinstance(threshold, float)
    
    label = np.asarray(SLs >= threshold, float)
    Y = np.vstack((label, SLs)).T
    
    return Y

if __name__ == '__main__':
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    ERSP_all, SLs = standardize(ERSP_all, tmp_all)
    new_ERSP, new_SLs = trimMean(ERSP_all, SLs, freqs)
    
    Y = make_target(new_SLs)