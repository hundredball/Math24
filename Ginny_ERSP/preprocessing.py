#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:28:06 2020

@author: hundredball
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing as skpp
import dataloader

def standardize(ERSP, tmp, num_time=1, train_indices=None, threshold=5.0):
    '''
    Average over time and subtract the base spectrum

    Parameters
    ----------
    ERSP : 4d numpy array (epoch, channel, freq_step, time_step)
        ERSP of all trials
    tmp : 2d numpy array (epoch, time_periods) or 1d numpy array (epoch)
        2d : time_periods include time points of fixation, cue, end
        1d : solution latency
    num_time : int, optional
        Number of time steps after standardizing. The default is 1.
    train_indices : 1d numpy array
        Indices of training data. The default is None.
    threshold : float
        If 0<threshold<=1, threshold is the quantile ratio
        If threshold > 1, threshold is the solution latency
        
    Returns
    -------
    ERSP : 3d numpy array or 4d numpy array(epoch, channel, freq_step, (time))
        ERSP of all trials
    SLs : 1d numpy array (epoch)
        solution latency of each trials

    '''
    assert isinstance(ERSP, np.ndarray) and ERSP.ndim == 4
    assert isinstance(tmp, np.ndarray) and (tmp.ndim == 2 or tmp.ndim == 1)
    assert isinstance(num_time, int) and num_time >= 1
    assert isinstance(threshold, float) and threshold >= 0
    #assert ERSP.shape[3]%num_time == 0
    
    if train_indices is None:
        train_indices = np.arange(ERSP.shape[0])
    
    time_step = int(ERSP.shape[3]/num_time)
    # Average over time
    ERSP_avg = np.zeros((ERSP.shape[0],ERSP.shape[1],ERSP.shape[2],num_time))
    for i_time in range(num_time):
        ERSP_avg[:,:,:,i_time] = np.mean(ERSP[:,:,:, i_time*time_step:(i_time+1)*time_step], axis=3)

    # Subtract the base spectrum (trials <= 80%)
    if tmp.ndim == 2:
        SLs = tmp[:, 2]
    else:
        SLs = tmp
    if 0 < threshold <= 1:
        threshold = np.quantile(SLs,threshold)
    print('Base threshold: %f'%(threshold))
    if threshold == 0:
        base = np.zeros(ERSP_avg.shape[1:])
    else:
        base = np.mean(ERSP_avg[ train_indices[np.where(SLs[train_indices]<=threshold)[0]], :, :, : ], axis=0)
    ERSP_avg = ERSP_avg - base[np.newaxis, :, :, :]
        
    if num_time == 1:
        ERSP_avg = np.squeeze(ERSP_avg, axis=3)
        
    return ERSP_avg, SLs

def center(train, test):
    '''
    Center training and testing data

    Parameters
    ----------
    train : numpy 2d array (epoch, features)
        Training data
    test : numpy 2d array (epoch, features)
        Testing data

    Returns
    -------
    train : numpy 2d array (epoch, features)
        Training data after standardizing
    test : numpy 2d array (epoch, features)
        Testing data after standardizing

    '''
    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)
    
    '''
    scaler = skpp.StandardScaler().fit(train)
    
    test = scaler.transform(test)
    train = scaler.transform(train)
    '''
    
    mean_train = np.mean(train, axis=0)
    
    test = test - mean_train
    train = train - mean_train
    
    return train, test
    

def select_correlated_ERSP(ERSP, SLs, threshold_corr=0.75, train_indices = None):
    '''
    Select ERSP whose correlation with solution latency is larger than threshold
    
    Parameters
    ----------
    ERSP : 3d numpy array (epoch, channel, freq_step)
        ERSP of all trials
    SLs : 1d numpy array (epoch)
        solution latency of each trials
    threshold_corr : float
        Threshold of correlation
    train_indices : 1d numpy array
        Indices of training data. The default is None.
    
    Returns
    -------
    select_ERSP : 2d numpy array (epoch, features)
        ERSP of interest
    select_indices : 1d numpy array 
        Indices of selected feature, 0: discard, 1: select
    
    '''
    assert isinstance(ERSP, np.ndarray) and ERSP.ndim == 3
    assert isinstance(SLs, np.ndarray) and SLs.ndim == 1
    assert isinstance(threshold_corr, float) and 0<=threshold_corr<=1
    
    if train_indices is None:
        train_indices = np.arange(ERSP.shape[0])
    
    train_ERSP = ERSP[train_indices,:]
    train_SLs = SLs[train_indices]
    # Change the dimension of ERSP_all
    ERSP_corr = train_ERSP.reshape((train_ERSP.shape[0],-1)).T
    #print('Shape of ERSP_corr:', ERSP_corr.shape)
    
    # Make SLs the same shape as ERSP_all
    SLs_corr = np.tile(train_SLs, (ERSP_corr.shape[0], 1))
    #print('Shape of SLs_corr: ', SLs_corr.shape)
    
    # Calculate the correlation matrix
    corr_mat = np.corrcoef(ERSP_corr, SLs_corr)
    #print(corr_mat.shape)
    corr_ERSP_SLs = corr_mat[:ERSP_corr.shape[0], ERSP_corr.shape[0]]
    #print('Shape of corr_ERSP_SLs: ', corr_ERSP_SLs.shape)
    
    # Select interested ERSP
    abs_corr = abs(corr_ERSP_SLs)
    select_indices = np.zeros(ERSP_corr.shape[0])
    select_indices[abs_corr >= np.quantile(abs_corr, threshold_corr)] = 1
    
    select_ERSP = ERSP.reshape((ERSP.shape[0],-1))[:, select_indices==1]
    #print(select_ERSP.shape)
    #print('Select %d features'%(np.sum(select_indices)))
    
    return select_ERSP, select_indices

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
    assert isinstance(ERSP, np.ndarray) and ERSP.ndim == 3
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1
    assert isinstance(low, int) and isinstance(high, int)
    assert high>=low
    
    index_freq = np.logical_and(freqs>low, freqs<high)
    bandpower = np.sum(ERSP[:,:,index_freq], axis=2)
    
    return bandpower

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
    assert isinstance(ERSP_all, np.ndarray) and ERSP_all.ndim == 3
    assert isinstance(SLs, np.ndarray) and SLs.ndim == 1
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1
    
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
    assert isinstance(SLs, np.ndarray) and SLs.ndim == 1
    
    if threshold is None:
        threshold = np.mean(SLs)
    else:
        assert isinstance(threshold, float)
    
    label = np.asarray(SLs >= threshold, float)
    Y = np.vstack((label, SLs)).T
    
    return Y

def PCA_corr(X_train, Y_train, X_test=None, num_features=2):
    '''
    Find two PCs most correlated with SLs

    Parameters
    ----------
    X_train : 2d numpy array (epoch, features)
        training data
    Y_train : 1d numpy array (epoch)
        Solution latency
    X_test : 2d numpy array (epoch, features)
        testing data, default=None
    Returns
    -------
    X_train : 2d numpy array (epoch, PC projections)
        training data after PCA
    X_est : 2d numpy array (epoch, PC projections)
        testing data after PCA
    
    '''
    assert isinstance(X_train, np.ndarray) and X_train.ndim == 2
    assert isinstance(Y_train, np.ndarray) and Y_train.ndim == 1
    assert X_train.shape[0] == Y_train.shape[0]
    if X_test is not None:
        assert isinstance(X_test, np.ndarray) and X_test.ndim == 2
    if num_features is not None:
        assert isinstance(num_features, int) and num_features>0
    
    # PCA fit
    pca = PCA(n_components=0.9)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)

    # Find n PCs correlated most strongly with SLs
    corr_coef = np.zeros(pca.n_components_)
    for i in range(pca.n_components_):
        corr_coef[i] = abs(np.corrcoef(X_train_pca[:,i], Y_train)[0,1])

    # Sort correlation coefficients in decreasing order
    sorted_indices = np.argsort(corr_coef)[::-1]
    feature_indices = [sorted_indices[i] for i in range(num_features)]
    for i in range(num_features):
        print('%d. %.3f'%(i+1, corr_coef[feature_indices[i]]))
    
    PC_n = pca.components_[feature_indices, :]
    
    # PCA predict
    X_train = X_train - pca.mean_
    X_train = np.dot(X_train, PC_n.T)
    if X_test is not None:
        X_test = X_test - pca.mean_
        X_test = np.dot(X_test, PC_n.T)
        return X_train, X_test
    
    return X_train

def remove_trials(ERSP_all, tmp_all, threshold):
    '''
    Remove trials with solution latency larger than threshold

    Parameters
    ----------
    ERSP_all : nd numpy array (epoch, ...)
        All data
    tmp_all : nd numpy array (epoch, time_periods)
        time_periods include time points of fixation, cue, end
        or solution latency
    threshold : float or int
        Threshold of removing trials
    Returns
    -------
    ERSP_rem : 2d numpy array (epoch, features)
        Data after removing trials
    tmp_rem : 2d numpy array (epoch, time_periods)
        time_periods include time points of fixation, cue, end after removing trials
    
    '''
    assert isinstance(ERSP_all, np.ndarray)
    assert isinstance(tmp_all, np.ndarray)
    assert isinstance(threshold, int) or isinstance(threshold, float)
    assert threshold > 0
    
    # Remove trials with SLs >= threshold
    if tmp_all.ndim == 2:
        remove_indices = np.where(tmp_all[:,2]>=threshold)[0]
    elif tmp_all.ndim == 1:
        remove_indices = np.where(tmp_all>=threshold)[0]
    ERSP_rem = np.delete(ERSP_all, remove_indices, axis=0)
    tmp_rem = np.delete(tmp_all, remove_indices, axis=0)
    print('> Remove %d trials'%(tmp_all.shape[0]-tmp_rem.shape[0]))
    
    return ERSP_rem, tmp_rem

if __name__ == '__main__':
    
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    ERSP_all, SLs = standardize(ERSP_all, tmp_all)
    
    #select_ERSP, select_indices = select_correlated_ERSP(ERSP_all, SLs)
    
    #X_train = PCA_corr(select_ERSP, SLs, 5)