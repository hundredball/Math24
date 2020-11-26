#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:38:58 2020

@author: hundredball
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

import raw_dataloader


class SourceSeparation(object):
    
    def __init__(self, num_channels, num_subs):
        
        self.num_channels = num_channels
        self.num_subs = num_subs
        
        # Load channel information
        self.channel_info = pd.read_csv('./Channel_coordinate/Channel_location_angle_%d.csv'%(num_channels))
        self.channel_info = self.channel_info.to_numpy()    # Columns: (Channel name, arc length, theta)
        
        # Calculate transformation matrix A (x = As) without attenuation factor
        A = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            for j in range(num_channels):
                if i==j:    # Diagonal line
                    A[i,j] = 1
                elif j>i:   # Upper triangle
                    r1, r2 = self.channel_info[i,1], self.channel_info[j,1]
                    t1, t2 = self.channel_info[i,2]*np.pi/180, self.channel_info[j,2]*np.pi/180
                    A[i,j] = 1/(r1**2 + r2**2 - 2*r1*r2*np.cos(t1-t2))
                else:       # Lower triangle
                    A[i,j] = A[j,i]
                    
        # Initialize transform matrix for each subject
        self.trans_mat = np.zeros((num_subs, num_channels, num_channels))
        for i in range(num_subs):
            self.trans_mat[i,:] = A
        
    def fit(self, data, sub):
        '''
        Find transform matrices for each subjects

        Parameters
        ----------
        data : np.ndarray (epoch, channel, time)
            Time signal
        sub : np.ndarray
            Subject ID

        Returns
        -------
        None.

        '''
        assert isinstance(data, np.ndarray) and data.ndim==3
        assert isinstance(sub, np.ndarray) and sub.ndim==1
        
        range_alpha = np.linspace(0.1, 1, 10)
        
        for i_sub in range(self.num_subs):
            
            index_sub = np.where(sub==i_sub)[0]     # Index of i_sub in data
            data_sub = data[index_sub,:]
            
            if len(data_sub) != 0:
                # Regression loss
                min_loss, min_alpha = 10000, 0
                loss = 0
                for alpha in range_alpha:
                    A_sub = self.get_A(alpha, i_sub)
                    inv_A_sub = np.linalg.pinv(A_sub)
                    # Calculate regression error ((x-x')**2, x^ = AA^{-1}x)
                    for x in data_sub:
                        x_hat = A_sub.dot(inv_A_sub).dot(x)
                        loss += (np.sum((x-x_hat)**2) / x.size)**0.5
                        
                    loss /= len(data_sub)
                    if loss <= min_loss:
                        min_loss, min_alpha = loss, alpha
                        
                    #print('Sub %d   Alpha %.1f   Rank: %d   Loss: %.3f'%(i_sub, alpha, np.linalg.matrix_rank(A_sub), loss))
                    
                self.trans_mat[i_sub, :] = self.get_A(min_alpha, i_sub)
            
        # Calculate inverse matrices
        self.inv_trans_mat = np.zeros(self.trans_mat.shape)
        for i in range(len(self.trans_mat)):
            self.inv_trans_mat[i,:] = np.linalg.pinv(self.trans_mat[i,:])
        
            
    def transform(self, data, sub):
        '''
        Transform from observed signal (x) to source signal (s)

        Parameters
        ----------
        data : np.ndarray (epoch, channel, time)
            Observed signal
        sub : np.ndarray
            Subject ID

        Returns
        -------
        data : np.ndarray (epoch, channel, time)
            Source signal
        '''
        assert isinstance(data, np.ndarray) and data.ndim==3
        assert isinstance(sub, np.ndarray) and sub.ndim==1
        
        for i_data in range(len(data)):
            data[i_data,:] = self.inv_trans_mat[sub[i_data],:].dot(data[i_data,:])
            
        return data
            
                
    def get_A(self, alpha, i_sub):
        '''
        Calculate transform matrix with attenuation factor

        Parameters
        ----------
        alpha : float
            Attenuation factor
        i_sub : int
            Subject ID

        Returns
        -------
        A : np.ndarray (channel, channel)
            x = As, x: Observed signal, s: Source signal
        '''
        assert isinstance(alpha, float)
        assert isinstance(i_sub, int)

        A = self.trans_mat[i_sub,:] * alpha
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                if i==j:
                    A[i,j] = 1
        
        return A        
        
def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")
    plt.subplot(3,1,3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")
    
    fig.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    
    '''
    X, Y, C, S, D = raw_dataloader.read_data([1,2,3], list(range(11)), channel_limit=21, rm_baseline = True)
    
    separator = SourceSeparation(X.shape[1], 11)
    separator.fit(X, S)
    X_hat = separator.transform(X, S)
    '''
    
    # Test ICA
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * time)+1
    s2 = np.sign(np.sin(3 * time))+1
    s3 = signal.sawtooth(2 * np.pi * time)+1
    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)
    S /= S.std(axis=0)
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
    X = np.dot(S, A.T)
    ica = FastICA(n_components=3)
    ica.fit(X)
    S_ = ica.transform(X)
    S__ = ica.components_.dot((X-ica.mean_).T).T
    print(ica.mean_.shape)
    fig = plt.figure()
    models = [X, S, S_,S__]
    names = ['mixtures', 'real sources', 'predicted sources', 'PS']
    colors = ['red', 'blue', 'orange']
    for i, (name, model) in enumerate(zip(names, models)):
        plt.subplot(5, 1, i+1)
        plt.title(name)
        for sig, color in zip (model.T, colors):
            plt.plot(sig, color=color)

    fig.tight_layout()        
    plt.show()
    

    
    