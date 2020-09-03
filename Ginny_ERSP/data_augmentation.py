#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:45:06 2020

@author: hundredball
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

import dataloader
import preprocessing

def undersampling(data, target):
    '''
    Random undersampling for regression

    Parameters
    ----------
    data : 2d numpy array (epoch, features)
        All the samples from the original data
    target : nd numpy array 
        Solution latency

    Returns
    -------
    new_data : 2d numpy array (epoch, features)
        New data with balance data set
    new_target : 1d numpy array
        New solution latency

    '''
    assert isinstance(data, np.ndarray) and data.ndim==2
    assert isinstance(target, np.ndarray) and target.shape[0]==target.size
    assert data.shape[0] == target.shape[0]
    
    # Use mean value as the threshold
    threshold = np.mean(target)
    print('Undersampling threshold : %f'%(threshold))
    
    minor_index = np.where(target >= threshold)[0]
    major_index = np.where(target < threshold)[0]
    
    # Randomly select samples from the majority
    select_index = np.random.choice(major_index, len(minor_index))
    
    # Generate new data
    new_data = np.concatenate((data[select_index, :], data[minor_index, :]), axis=0)
    new_target = np.concatenate((target[select_index], target[minor_index]), axis=0)
    
    print('> After undersampling: (%d,%d)'%(np.sum(new_target>=threshold), np.sum(new_target<threshold)))
    
    return new_data, new_target

def SMOTER(data, target, threshold=None):
    '''
    SMOTE for regression

    Parameters
    ----------
    data : nd numpy array 
        All the samples from the original data
    target : nd numpy array 
        Solution latency

    Returns
    -------
    new_data : 2d numpy array (epoch, features)
        New data with balance data set
    new_target : 1d numpy array
        New solution latency

    '''
    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray) and target.shape[0]==target.size
    assert data.shape[0] == target.shape[0]
    
    # Transfor data to (epoch, features), target to (epoch)
    ori_data_ndim = data.ndim
    if ori_data_ndim != 2:
        ori_data_shape = data.shape
        data = data.reshape((data.shape[0],-1))
    if target.ndim == 2:
        target = target.flatten()
    
    # Calculate the histogram of the data
    if threshold is not None:
        assert 0<=threshold<=np.max(target)
        hist, bin_edges = np.histogram(target, bins=[0, threshold, np.max(target)])
    else:
        hist, bin_edges = np.histogram(target, bins=10)
    max_index = np.argmax(hist)
    major_indices = np.where(np.logical_and(target >= bin_edges[max_index], target <= bin_edges[max_index+1]))[0]
    num_major = len(major_indices)
    
    data_aug = data[major_indices,:]
    target_aug = target[major_indices]
    
    for i in range(len(hist)):
        if i == max_index:
            continue
        
        minor_indices = np.where(np.logical_and(target >= bin_edges[i], target <= bin_edges[i+1]))[0]
        num_minor = len(minor_indices)
        minor_data = data[minor_indices,:]
        minor_target = target[minor_indices]
        
        # number of new samples each minor samples generates
        try:
            num_new = int(np.ceil(num_major/num_minor) - 1)
        except:
            print('Number of minor equals to 0')
            continue
        new_data = np.zeros((num_new*num_minor, data.shape[1]))
        new_target = np.zeros(num_new*num_minor)
        
        # Generate synthetic samples
        if num_minor < 2:
            data_aug = np.concatenate((data_aug, minor_data), axis=0)
            target_aug = np.concatenate((target_aug, minor_target))
            continue
        elif 2 <= num_minor <= 6:
            num_neigh = num_minor
        else:
            num_neigh = 6
        neigh = NearestNeighbors(n_neighbors=num_neigh, radius=0.4)
        neigh.fit(minor_data)
        for iter_minor, index_minor in enumerate(minor_indices):
            #print(index_minor)
            #print(data[index_minor,:].shape)
            neighbors_indices = neigh.kneighbors(data[index_minor, :].reshape((1,-1)), num_neigh, return_distance=False)
            for i_new in range(num_new):
                # Remove the original minor point from the neighbors
                index_neighbor = np.random.choice(neighbors_indices.flatten()[1:])
                
                # Interpolation for predictor
                sample_vector = data[index_minor,:]
                feature_vector = minor_data[index_neighbor,:]
                diff = sample_vector - feature_vector
                new_sample = sample_vector - np.random.rand()*diff
                
                # Interpolation for target
                d1 = np.linalg.norm(new_sample-sample_vector)
                d2 = np.linalg.norm(new_sample-feature_vector)
                new_sample_target = (d1*minor_target[index_neighbor]+d2*target[index_minor]) / (d1+d2)
                
                new_data[iter_minor*num_new + i_new,:] = new_sample
                new_target[iter_minor*num_new + i_new] = new_sample_target
        
        # Add synthetic minor data and original minor data
        data_aug = np.concatenate((data_aug, new_data, minor_data), axis=0)
        target_aug = np.concatenate((target_aug, new_target, minor_target))
    
    # Transform data and target back to the original shape
    if ori_data_ndim != 2:
        data_aug = data_aug.reshape((data_aug.shape[0],)+ori_data_shape[1:])
    if target.ndim == 2:
        target_aug = target_aug.reshape((-1,1))
    
    return data_aug, target_aug

def overlapping(data, target, params):
    '''
    Augment data by overlapping window

    Parameters
    ----------
    data : numpy 3d array (epoch, channels, time)
        Time series data
    target : numpy 1d array (epoch)
        Solution latencies

    Returns
    -------
    data_aug : numpy 3d array (epoch, channels, time)
        Augmented time series data
    target_aug : numpy 1d array (epoch)
        Augmented solution latencies

    '''
    assert isinstance(data, np.ndarray) and data.ndim==3
    assert isinstance(target, np.ndarray) and target.ndim==1
    assert data.shape[0] == target.shape[0]
    
    fs = params[0]              # sampling rate
    overlap_size = params[1]    # overlapping size
    window_size = params[2]     # window size for the data
    len_time = data.shape[2]    # length of time
    assert len_time>=window_size>overlap_size>=0
    
    # multiplication of the number of original data
    mul = (len_time-window_size)//(window_size-overlap_size)+1
    
    for i in range(mul):
        if i == 0:
            data_aug = data[:,:, :window_size]
        else:
            start_point = (window_size-overlap_size)*i
            new_data = data[:,:, start_point:start_point+window_size]
            data_aug = np.concatenate((data_aug, new_data), axis=0)
            
    target_aug = np.tile(target, mul)
    
    return data_aug, target_aug

def add_noise(data, target, params):
    '''
    Augment data by adding zero-mean Gaussian noise

    Parameters
    ----------
    data : numpy nd array
        Time series data or power spectrum
    target : numpy 1d array (epoch)
        Solution latencies

    Returns
    -------
    data_aug : numpy 3d array (epoch, channels, time)
        Augmented time series data
    target_aug : numpy 1d array (epoch)
        Augmented solution latencies

    '''
    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray) and target.ndim==1
    assert data.shape[0] == target.shape[0]
    
    mul = params[0]     # Multiplication of the number the data
    std = params[1]     # Standard deviation of Gaussian distribution
    
    for i in range(mul):
        
        if i == 0:
            data_aug = data.copy()
        else:
            noise = np.random.normal(loc=0.0, scale=std, size=data.shape)
            data_aug = np.concatenate((data_aug, data+noise), axis=0)
            
    target_aug = np.tile(target, mul)
            
    return data_aug, target_aug

def add_noise_minority(data, target, params):
    '''
    Augment data by adding zero-mean Gaussian noise to minority

    Parameters
    ----------
    data : numpy nd array
        Time series data or power spectrum
    target : numpy 1d array (epoch)
        Solution latencies

    Returns
    -------
    data_aug : numpy 3d array (epoch, channels, time)
        Augmented time series data
    target_aug : numpy 1d array (epoch)
        Augmented solution latencies

    '''
    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray) and target.ndim==1
    assert data.shape[0] == target.shape[0]
    

    # Use mean value as threshold
    threshold = np.mean(target)
    
    # Divide into majority and minority
    minor_indices = np.where(target >= threshold)[0]
    major_indices = np.where(target < threshold)[0]
    
    # Add noise to minority data
    data_aug, target_aug = add_noise(data[minor_indices,:], target[minor_indices], params)
    
    # Concatenate with majority data
    data_aug = np.concatenate((data_aug, data[major_indices,:]), axis=0)
    target_aug = np.concatenate((target_aug, target[major_indices]))
    
    return data_aug, target_aug
    

def aug(data, target, method, params=None):
    
    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray) and target.ndim==1
    assert data.shape[0] == target.shape[0]
    
    print('--- Data Augmentation (%s) ---'%(method))
    
    ori_data_shape = data.shape
    ori_target_shape = target.shape
    
    if method == 'undersampling':
        data, target = undersampling(data, target)
    elif method == 'SMOTER':
        data, target = SMOTER(data, target, params)
    elif method == 'overlapping':
        data, target = overlapping(data, target, params)
    elif method == 'add_noise':
        data, target = add_noise(data, target, params)
    elif method == 'add_noise_minority':
        data, target = add_noise_minority(data, target, params)
        
    print('> After %s'%(method))
    print('Mean of all trials: %f'%(np.mean(target)))
    print('Data: %s -> %s'%(ori_data_shape, data.shape))
    print('Target: %s -> %s'%(ori_target_shape, target.shape))
        
    return data, target

if __name__ == '__main__':
    
    '''
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all)
    ERSP_all = preprocessing.bandpower(ERSP_all, freqs, 5, 10)
    
    # Test undersampling
    new_data, new_target = undersampling(ERSP_all, SLs)
    threshold = np.mean(SLs)
    num_major = np.sum(new_target < threshold)
    num_minor = np.sum(new_target >= threshold)
    assert num_major == num_minor
    
    # Test SMOTER
    new_data, new_target = SMOTER(ERSP_all, SLs)
    threshold = np.mean(SLs)
    num_major = np.sum(new_target < threshold)
    num_minor = np.sum(new_target >= threshold)
    print('Original classes: %d | %d'%(np.sum(SLs<threshold), np.sum(SLs>=threshold)))
    print('After : %d | %d'%(num_major, num_minor))
    
    '''
    
    # Test overlapping
    data = np.random.rand(100, 12, 512)
    target = np.random.rand(100)
    
    # data, target = aug(data, target, 'overlapping', (256, 64, 128))
    data, target = aug(data, target, 'add_noise', (5,0.2))