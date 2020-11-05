#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:12:53 2020

@author: hundredball
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import raw_dataloader
import preprocessing
import bandpower
import add_features
from sklearn.ensemble import RandomForestRegressor


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
        
    
    def fit(self, train_data, train_target, train_sub, num_closest=10):
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
        num_closest : int
            Number of closest trials for average

        Returns
        -------
        None.

        '''
        assert isinstance(train_data, np.ndarray) and train_data.ndim==3
        assert isinstance(train_target, np.ndarray) and train_target.ndim==1
        assert isinstance(train_sub, np.ndarray) and train_sub.ndim==1
        assert isinstance(num_closest, int) and num_closest>0
        
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
                
                dist_sub = np.zeros((num_closest, len(sub_data)))
                
                for i_data in range(len(sub_data)):
                    # Get mean of 10 base_data with closes targets
                    dist = np.abs(base_target - sub_target[i_data])
                    dist_sub[:,i_data] = np.sort(dist)[:num_closest]
                    mean_data = np.mean(base_data[np.argsort(dist)[:num_closest], :], axis=0)
                    
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
    
    def fit_(self, train_data, train_target, train_sub):
        '''
        Save trial of base subject in training data

        Parameters
        ----------
        train_data : np.ndarray (epoch, channel, features)
            Training data for transforming
        train_sub : np.ndarray
            Training subject IDs

        Returns
        -------
        None.

        '''
        assert isinstance(train_data, np.ndarray) and train_data.ndim==3
        assert isinstance(train_target, np.ndarray) and train_target.ndim==1
        assert isinstance(train_sub, np.ndarray) and train_sub.ndim==1
        
        base_index = np.where(train_sub==self.base_sub)[0]
        self.train_base_data = train_data[base_index,:]
        self.train_base_target = train_target[base_index]
    
    def transform_(self, data, target, sub, num_closest=3, dist_type='target'):
        '''
        Least square transform for all the subjects except base subject (all test data are from base subject)

        Parameters
        ----------
        data : np.ndarray (epoch, channel, features)
            Training data for transforming
        target : np.ndarray
            Training solution latency
        sub : np.ndarray
            Training subject IDs
        num_closest : int
            Number of closest targets

        Returns
        -------
        data : np.ndarray (epoch, channel, features)
            Training data after transforming

        '''
        assert isinstance(data, np.ndarray) and data.ndim==3
        assert isinstance(target, np.ndarray) and target.ndim==1
        assert isinstance(sub, np.ndarray) and sub.ndim==1
        assert isinstance(num_closest, int) and num_closest>0
        assert dist_type in ['target', 'correlation']
        
        if dist_type == 'correlation':
            train_base_data_corr = add_features.get_correlations(self.train_base_data)
        for i_data in range(len(data)):
            
            if sub[i_data] != self.base_sub:
                # Get mean of 3 base_data with closes targets
                if dist_type == 'target':
                    dist = np.abs(self.train_base_target - target[i_data])
                elif dist_type == 'correlation':
                    data_corr = add_features.get_correlations(data[i_data][np.newaxis,:,:])
                    dist = np.sum((train_base_data_corr-data_corr)**2, axis=1)
                mean_data = np.mean(data[np.argsort(dist)[:num_closest], :], axis=0)
                
                # Channel-wise least square transform onto mean_data
                model = LinearRegression()
                model.fit(data[i_data,:].T, mean_data.T)
                data[i_data,:] = model.predict(data[i_data,:].T).T
                
        return data
    
def plot_scatter(true, pred, fileName=None):
    plt.ioff()
    
    sort_indices = np.argsort(true)
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].plot(range(len(true)), true[sort_indices], 'r.', range(len(true)), pred[sort_indices], 'b.')
    axs[0].set_xlabel('Record number')
    axs[0].set_ylabel('Solution latency')
    axs[0].legend(('True', 'Pred'))
    
    max_value = np.max(np.hstack((true, pred)))
    axs[1].scatter(true, pred, marker='.')
    axs[1].plot(range(int(max_value)),range(int(max_value)), 'r')
    axs[1].set_xlabel('True')
    axs[1].set_ylabel('Pred')
    axs[1].set_xlim([0, max_value])
    axs[1].set_ylim([0, max_value])
    axs[1].set_title('r = %.3f'%(np.corrcoef(true, pred)[0,1]))
    
    std = mean_squared_error(true, pred)**0.5
    fig.suptitle('Standard error: %.3f'%(std))
    
    if fileName is not None:
        plt.savefig('./results/classical/%s_scatter.png'%(fileName))
        
    plt.close()
    
if __name__ == '__main__':
    
    # Load data
    X,Y,C,S = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
    
    # Leave one subject out
    dict_error = {'train_std': np.zeros((11, 5)), 'test_std': np.zeros((11, 5))}
    for i_base in range(11):
        print('----- Subject %d -----'%(i_base))
        
        lst_model = LST(11, i_base)
        indices_base, indices_other = np.where(S==i_base)[0], np.where(S!=i_base)[0]
        base_data, base_target, base_sub = X[indices_base,:], Y[indices_base], S[indices_base]
        other_data, other_target, other_sub = X[indices_other,:], Y[indices_other], S[indices_other]
        test_pred_all, test_target_all = np.zeros(len(base_data)), np.zeros(len(base_data))
        
        # K-fold cross validation (all test data are in one subject)
        kf = KFold(n_splits=5, shuffle=True, random_state=23)
        curr_test_index = 0
        for i_split, (more_index, few_index) in enumerate(kf.split(base_data)):
            # Wrap up training and testing data
            train_data, test_data = np.concatenate((base_data[more_index,:],other_data), axis=0), base_data[few_index,:]
            train_target, test_target = np.concatenate((base_target[more_index],other_target), axis=0), base_target[few_index]
            train_sub, test_sub = np.concatenate((base_sub[more_index],other_sub), axis=0), base_sub[few_index]
            
            # LST for training data
            lst_model.fit_(train_data, train_target, train_sub)
            train_data = lst_model.transform_(train_data, train_target, train_sub)
            '''
            # Extract ERSP
            _,_, train_data = bandpower.STFT(train_data, train_target, train_sub, 2, 30)
            _,_, test_data = bandpower.STFT(test_data, test_target, test_sub, 2, 30)
            train_data, _ = preprocessing.standardize(train_data, train_target, threshold=0.0)
            test_data, _ = preprocessing.standardize(test_data, test_target, threshold=0.0)
            '''
            # PCA
            train_data, test_data = train_data.reshape((train_data.shape[0],-1)), test_data.reshape((test_data.shape[0],-1))
            pca = PCA(n_components=30)
            pca.fit(train_data)
            train_data = pca.transform(train_data)
            test_data = pca.transform(test_data)
            
            # Regression
            rgr_model = RandomForestRegressor(max_depth=20, random_state=10, n_estimators=100)
            rgr_model.fit(train_data, train_target)
            train_pred = rgr_model.predict(train_data)
            test_pred = rgr_model.predict(test_data)
            
            # Record error and prediction
            train_std = mean_squared_error(train_target, train_pred)**0.5
            test_std = mean_squared_error(test_target, test_pred)**0.5
            print('Split %d    Std: (%.3f,%.3f)'%(i_split, train_std, test_std))
            dict_error['train_std'][i_base,i_split] = train_std
            dict_error['test_std'][i_base,i_split] = test_std
            
            test_pred_all[curr_test_index:curr_test_index+len(few_index)] = test_pred
            test_target_all[curr_test_index:curr_test_index+len(few_index)] = test_target
            curr_test_index += len(few_index)
            
        plot_scatter(train_target, train_pred, fileName='RF20_100_LST_sub%d_train'%(i_base))
        plot_scatter(test_target_all, test_pred_all, fileName='RF20_100_LST_sub%d'%(i_base))