#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:12:53 2020

@author: hundredball
"""

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

import raw_dataloader
import preprocessing


class LST(object):
    
    def __init__(self, num_sub, base_sub):
        
        self.num_sub = num_sub
        self.base_sub = base_sub        # Target subject being projected onto
        
    def center_data_(self, data):
        '''
        Center each data respectively across channels

        Parameters
        ----------
        data : np.ndarray (epoch, channels, features)
            Original data

        Returns
        -------
        data : np.ndarray (epoch, channels, features)
            Transformed data

        '''
        assert isinstance(data, np.ndarray) and data.ndim==3
        
        for i in range(len(data)):
            data[i,:,:] = data[i,:,:] - np.mean(data[i,:,:], axis=1)[:, np.newaxis]
        
        return data
        
        
    def fit(self, train_data, train_target, train_sub):
        '''
        Calculate transformation matrix for each subject

        Parameters
        ----------
        train_data : np.ndarray (epoch, channels, features)
            Training data with all subjects
        train_target : np.ndarray
            Solution latencies of training data
        train_sub : np.ndarray
            Subject IDs of training data

        Returns
        -------
        None.

        '''
        assert isinstance(train_data, np.ndarray) and train_data.ndim==3
        assert isinstance(train_target, np.ndarray) and train_target.ndim==1
        assert isinstance(train_sub, np.ndarray) and train_sub.ndim==1
        
        self.train_data = train_data
        self.train_target = train_target
        self.train_sub = train_sub
        
        self.trans_matrix = []      # List for placing transformation matrix of each subject
        
        # Center each data respectively across channels
        train_data = self.center_data_(train_data)
        
        # Collect base data for projection target
        base_data = train_data[np.where(train_sub==self.base_sub)[0], :]
        base_target = train_target[np.where(train_sub==self.base_sub)[0]]
        
        dist_all = []
        
        for i_sub in range(self.num_sub):
            if i_sub == self.base_sub:
                self.trans_matrix.append(np.eye(train_data.shape[1]))
            else:
                
                # Collect data of i_sub
                index_sub = np.where(train_sub==i_sub)[0]
                sub_data, sub_target = train_data[index_sub,:], train_target[index_sub]
                
                # Initialize tranformation matrix for each sub_data
                P = np.zeros((len(sub_data), sub_data.shape[1], sub_data.shape[1]))
                
                dist_sub = np.zeros((10, len(sub_data)))
                
                for i_data in range(len(sub_data)):
                    # Get mean of 10 base_data with closes targets
                    dist = np.abs(base_target - sub_target[i_data])
                    dist_sub[:,i_data] = np.sort(dist)[:10]
                    mean_data = np.mean(base_data[np.argsort(dist)[:10], :], axis=0)
                    
                    # Channel-wise least square transform onto mean_data
                    model = LinearRegression(fit_intercept=False)
                    model.fit(sub_data[i_data,:].T, mean_data.T)
                    P[i_data, :] = model.coef_
                
                # Average all the P to get a transformation matrix for that subject
                self.trans_matrix.append(np.mean(P, axis=0))
                
                dist_all.append(dist_sub)
                
        return dist_all
        
    def transform(self, data, sub):
        '''
        Least square transform for input

        Parameters
        ----------
        data : np.ndarray (epoch, channel, features)
            Input data for transforming
        target : np.ndarray
            Solution latency
        sub : np.ndarray
            Subject IDs

        Returns
        -------
        None.

        '''
        assert isinstance(data, np.ndarray) and data.ndim==3
        assert isinstance(sub, np.ndarray) and sub.ndim==1
        
        # Center the data
        data = self.center_data_(data)
        
        # Transform each data (x' = Px)
        trans_data = [self.trans_matrix[sub[i_data]].dot(data[i_data, :, :]) for i_data in range(len(data))]
        trans_data = np.array(trans_data)
        
        return trans_data
    
if __name__ == '__main__':
    
    # Load data
    fileName = './raw_data/ERSP_from_raw_100_channel21.data'
    with open(fileName, 'rb') as fp:
        dict_ERSP = pickle.load(fp)
    ERSP_all, SLs_all, subs_all = dict_ERSP['ERSP'], dict_ERSP['SLs'], dict_ERSP['Sub_ID']
    
    ERSP_all, SLs_all = preprocessing.standardize(ERSP_all, SLs_all, threshold=0.0)
    
    lst_model = LST(11, 2)
        
    # Test fit, transform
    dist_all = lst_model.fit(ERSP_all, SLs_all, subs_all)
    trans_ERSP = lst_model.transform(ERSP_all, subs_all)