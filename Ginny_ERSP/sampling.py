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
    target : 1d numpy array 
        Solution latency

    Returns
    -------
    new_data : 2d numpy array (epoch, features)
        New data with balance data set
    new_target : 1d numpy array
        New solution latency

    '''
    assert isinstance(data, np.ndarray) and len(data.shape) == 2
    assert isinstance(target, np.ndarray) and len(target.shape) == 1
    assert data.shape[0] == target.shape[0]
    
    # Use mean value as the threshold
    threshold = np.mean(target)
    
    minor_index = np.where(target >= threshold)[0]
    major_index = np.where(target < threshold)[0]
    
    # Randomly select samples from the majority
    select_index = np.random.choice(major_index, len(minor_index))
    
    # Generate new data
    new_data = np.concatenate((data[select_index, :], data[minor_index, :]), axis=0)
    new_target = np.concatenate((target[select_index], target[minor_index]), axis=0)
    
    return new_data, new_target

def SMOTER(data, target):
    '''
    SMOTE for regression

    Parameters
    ----------
    data : 2d numpy array (epoch, features)
        All the samples from the original data
    target : 1d numpy array 
        Solution latency

    Returns
    -------
    new_data : 2d numpy array (epoch, features)
        New data with balance data set
    new_target : 1d numpy array
        New solution latency

    '''
    assert isinstance(data, np.ndarray) and len(data.shape) == 2
    assert isinstance(target, np.ndarray) and len(target.shape) == 1
    assert data.shape[0] == target.shape[0]
    
    # Use mean value as the threshold
    threshold = np.mean(target)
    
    minor_indices = np.where(target >= threshold)[0]
    major_indices = np.where(target < threshold)[0]
    minor_data = data[minor_indices,:]
    minor_target = target[minor_indices]
    
    # number of new samples each minor samples generates
    num_new = int(np.ceil(len(major_indices)/len(minor_indices)) - 1)
    new_data = np.zeros((num_new*len(minor_indices), data.shape[1]))
    new_target = np.zeros(num_new*len(minor_indices))
    
    # Generate synthetic samples
    neigh = NearestNeighbors(n_neighbors=6, radius=0.4)
    neigh.fit(minor_data)
    for iter_minor, index_minor in enumerate(minor_indices):
        neighbors_indices = neigh.kneighbors(data[index_minor, :].reshape((1,-1)), 6, return_distance=False)
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
    
    # Add original minor and major data
    new_data = np.concatenate((new_data, data), axis=0)
    new_target = np.concatenate((new_target, target))
    
    return new_data, new_target

if __name__ == '__main__':
    
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all)
    ERSP_all = preprocessing.bandpower(ERSP_all, freqs, 5, 10)
    
    '''
    # Test undersampling
    new_data, new_target = undersampling(ERSP_all, SLs)
    threshold = np.mean(SLs)
    num_major = np.sum(new_target < threshold)
    num_minor = np.sum(new_target >= threshold)
    assert num_major == num_minor
    '''
    
    # Test SMOTER
    new_data, new_target = SMOTER(ERSP_all, SLs)
    threshold = np.mean(SLs)
    num_major = np.sum(new_target < threshold)
    num_minor = np.sum(new_target >= threshold)
    print('Original classes: %d | %d'%(np.sum(SLs<threshold), np.sum(SLs>=threshold)))
    print('After : %d | %d'%(num_major, num_minor))