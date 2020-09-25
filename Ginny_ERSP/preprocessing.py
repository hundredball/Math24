#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:28:06 2020

@author: hundredball
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing as skpp
from scipy.integrate import simps
import random as rand
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

def normalize(train, test):
    '''
    Electrode-wise exponential moving standardization

    Parameters
    ----------
    train : numpy 3d array (epoch, channel, features)
        Training data
    test : numpy 2d array (epoch, channel, features)
        Testing data

    Returns
    -------
    train : numpy 2d array (epoch, features)
        Training data after normalizing
    test : numpy 2d array (epoch, features)
        Testing data after normalizing

    '''
    assert isinstance(train, np.ndarray) and train.ndim==3
    assert isinstance(test, np.ndarray) and test.ndim==3
    assert train.shape[1] == test.shape[1] and train.shape[2] == test.shape[2]
    
    print('Moving standardize the data...')
    
    mean_train, var_train = np.zeros(train.shape), np.zeros(train.shape)
    mean_test, var_test = np.zeros(test.shape), np.zeros(test.shape)
    
    # Take first mean and variance of first 256 points as initial mean and variance
    base = 128
    mean_train[:,:,:base] = np.mean(train[:,:,:base], axis=2)[:,:,np.newaxis]
    var_train[:,:,:base] = np.var(train[:,:,:base], axis=2)[:,:,np.newaxis]
    mean_test[:,:,:base] = np.mean(test[:,:,:base], axis=2)[:,:,np.newaxis]
    var_test[:,:,:base] = np.var(test[:,:,:base], axis=2)[:,:,np.newaxis]
    
    train[:,:,:base] = (train[:,:,:base]-mean_train[:,:,:base])/(var_train[:,:,:base]**0.5)
    test[:,:,:base] = (test[:,:,:base]-mean_test[:,:,:base])/(var_test[:,:,:base]**0.5)
    for i_time in range(base, train.shape[2]):
        
        mean_train[:,:,i_time] = 0.001*train[:,:,i_time]+0.999*mean_train[:,:,i_time-1]
        mean_test[:,:,i_time] = 0.001*test[:,:,i_time]+0.999*mean_test[:,:,i_time-1]
        
        var_train[:,:,i_time] = 0.001*(train[:,:,i_time]-mean_train[:,:,i_time])**2 + 0.999*var_train[:,:,i_time-1]
        var_test[:,:,i_time] = 0.001*(test[:,:,i_time]-mean_test[:,:,i_time])**2 + 0.999*var_test[:,:,i_time-1]
            
        train[:,:,i_time] = (train[:,:,i_time]-mean_train[:,:,i_time])/(var_train[:,:,i_time]**0.5)
        test[:,:,i_time] = (test[:,:,i_time]-mean_test[:,:,i_time])/(var_test[:,:,i_time]**0.5)
    
    return train, test

def center(train, test):
    '''
    Center training and testing data for each features

    Parameters
    ----------
    train : numpy 3d array (epoch, features)
        Training data
    test : numpy 2d array (epoch, features)
        Testing data

    Returns
    -------
    train : numpy 2d array (epoch, features)
        Training data after centering
    test : numpy 2d array (epoch, features)
        Testing data after centering

    '''
    
    assert isinstance(train, np.ndarray) and train.ndim==2
    assert isinstance(test, np.ndarray) and test.ndim==2
    
    print('Center the data...')
    
    mean_train = np.mean(train, axis=0)
    
    test = test - mean_train[np.newaxis,:]
    train = train - mean_train[np.newaxis,:]
    
    return train, test

def stratified_split(X, Y, n_split=3, mode=1):
    '''
    Split data by ordered solution latency

    Parameters
    ----------
    X : np.ndarray (epoch, ...)
        Data
    Y : np.ndarray (epoch)
        Solution latency
    n_split : int, optional
        Number of split clusters. The default is 3.
    mode : int, optional
        Split mode. The default is 1.
        1 : Random
        2 : Group split (0-25% | 25%-50%...)
        3 : Strided split (1th, 5th, ... | 2th, 6th, ...)

    Returns
    -------
    None.

    '''
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray) and Y.shape[0]==Y.size
    assert isinstance(n_split, int) and n_split>0
    assert isinstance(mode, int)
    
    ori_Y_ndim = Y.ndim
    if Y.ndim == 2:
        Y = Y.flatten()
    
    # Arrange trials in ascending order or random
    if mode == 1:
        print('Split trials randomly')
        random_indices = np.arange(len(X))
        rand.Random(4).shuffle(random_indices)
        X = X[random_indices, :]
        Y = Y[random_indices]
    elif mode == 2:
        print('Split trials by ordered groups')
        sorted_indices = np.argsort(Y)
        X = X[sorted_indices, :]
        Y = Y[sorted_indices]
    elif mode == 3:
        print('Split trials with strides')
        sorted_indices = np.argsort(Y)
        X = X[sorted_indices, :]
        Y = Y[sorted_indices]
        
        # Split them into list
        i = 0
        X_list, Y_list = [], []
        indices_dict = {i : [] for i in range(n_split)}
        while (i < len(Y)):
            indices_dict[i%n_split].append(i)
            i += 1
        for i_list in range(n_split):
            indices_dict[i_list] = np.array(indices_dict[i_list])
            X_list.append(X[indices_dict[i_list], :])
            Y_list.append(Y[indices_dict[i_list]])
    
    if mode in [1,2]:
        # Split them into list
        num_trials_split = len(X)//n_split
        X_list, Y_list = [], []
        for i in range(n_split):
            if i == n_split-1:
                X_list.append(X[i*num_trials_split:, :])
                Y_list.append(Y[i*num_trials_split:])
            else:
                X_list.append(X[i*num_trials_split:(i+1)*num_trials_split, :])
                Y_list.append(Y[i*num_trials_split:(i+1)*num_trials_split])

    if ori_Y_ndim == 2:
        Y_list = [Y.reshape((-1,1)) for Y in Y_list]
    
    return X_list, Y_list

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
    print('Select %d features'%(np.sum(select_indices)))
    
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
    low : list
        lower bound
    high : list
        upper bound

    Returns
    -------
    bandpower : 3d numpy array (epoch, channel, band)
        Bandpower of given bands

    '''
    assert isinstance(ERSP, np.ndarray) and ERSP.ndim == 3
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1
    assert all([low[i]<high[i] for i in range(len(low))]) and len(low) == len(high)
    
    freq_res = freqs[1]-freqs[0]
    for i in range(len(low)):
        index_freq = np.logical_and(freqs>low[i], freqs<high[i])
        bandpower_i = simps(ERSP[:,:,index_freq], dx=freq_res, axis=2).reshape((ERSP.shape[0],ERSP.shape[1],1))
        if i == 0:
            bandpower = bandpower_i
        else:
            bandpower = np.concatenate((bandpower, bandpower_i), axis=2)
    
    return bandpower

def trimData(ERSP_all, tmp_all):
    
    num_example = len(ERSP_all)
    
    SLs = tmp_all[:,2]
    sorted_indices = np.argsort(SLs)
    
    ERSP_all = ERSP_all[sorted_indices, :]
    tmp_all = tmp_all[sorted_indices, :]
    
    ERSP_all = ERSP_all[num_example//4: -num_example//4, :]
    tmp_all = tmp_all[num_example//4: -num_example//4, :]
    
    print('Remain %d trials'%(len(ERSP_all)))
    
    return ERSP_all, tmp_all

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
    print('> Remove %d trials (%.3f sec)'%(tmp_all.shape[0]-tmp_rem.shape[0], threshold))
    
    return ERSP_rem, tmp_rem

if __name__ == '__main__':
    
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    ERSP_all, SLs = standardize(ERSP_all, tmp_all)
    
    #select_ERSP, select_indices = select_correlated_ERSP(ERSP_all, SLs)
    
    #X_train = PCA_corr(select_ERSP, SLs, 5)
    
    X_list, Y_list = stratified_split(ERSP_all, SLs, n_split=4, mode=3)