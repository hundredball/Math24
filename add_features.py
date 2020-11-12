#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:57:14 2020

@author: hundredball
"""

import numpy as np
import time
import pickle
import dataloader
import preprocessing

from pyinform.conditionalentropy import conditional_entropy
from scipy.stats import multivariate_normal
from itertools import combinations, permutations

import os,sys,inspect
import raw_dataloader

def parzen_window_est(X_train, X_test, h):
    '''
    Parzen-window estimation for Gaussian kernels

    Parameters
    ----------
    X_train : np.ndarray (epoch, features)
        Training data
    X_test : np.ndarray (epoch, features)
        Testing data
    h : float
        Window length

    Returns
    -------
    prob_test : np.ndarray
        Estimated probabilities of testing data

    '''
    assert isinstance(X_train, np.ndarray) and X_train.ndim == 2
    assert isinstance(X_test, np.ndarray) and X_test.ndim == 2
    assert isinstance(h, float) and h > 0
    assert X_train.ndim == X_test.ndim
    
    d = X_train.shape[1]    # Dimension
    N = X_train.shape[0]    # Number of training data
    
    # Calculate covariance matrix from training data
    mu = np.mean(X_train, axis=0).reshape((1,d))
    Sigma = 1/N*(X_train-mu).T.dot(X_train-mu)
    inv_Sigma = np.linalg.inv(Sigma)
    halfnorm_Sigma = np.linalg.norm(Sigma)**0.5
    
    prob_test = np.zeros(X_test.shape[0])
    # Estimate prob
    for i, test_sample in enumerate(X_test):
        
        '''
        prob = 0
        for train_sample in X_train:
            z = (test_sample-train_sample).reshape((d,1))
            prob += 1/N * np.exp(-z.T.dot(inv_Sigma).dot(z))[0,0] / ( ((2*np.pi)**(d/2)) * (h**d) * halfnorm_Sigma )
        '''
        
        z = test_sample.reshape((-1,1)) - X_train.T
        prob = 1/N * np.exp(-np.matmul(np.matmul(z.T,inv_Sigma),z)) / ( ((2*np.pi)**(d/2)) * (h**d) * halfnorm_Sigma )
        prob = np.sum(np.diag(prob))
        
        prob_test[i] = prob

    return prob_test

def get_conditional_entropy(X_train, X_test, Y_train, Y_test):
    '''
    Calculate conditional entropy H(X_test|Y_test)

    Parameters
    ----------
    X_train : np.ndarray
        Training channel data
    Y_train : np.ndarray
        Training channel data
    X_test : np.ndarray
        Testing channel data
    Y_test : np.ndarray
        Testing channel data

    Returns
    -------
    CE : float
        Conditional entropy H(X_test|Y_test)

    '''
    assert isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray) and isinstance(Y_test, np.ndarray)
    assert len(X_train) == len(Y_train)
    assert len(X_test) == len(Y_test)
    
    num_test = len(X_test)
    
    # Calculate marginal prob of Y
    prob_Y = parzen_window_est(Y_train, Y_test, 0.5)
    print('A')

    # Calculate joint prob of X adn Y
    prob_XY = np.zeros((num_test, num_test))
    XY_train = np.concatenate((X_train, Y_train), axis=1)
    for i_Y in range(num_test):
        XY_comb = np.concatenate((X_test, np.zeros((num_test,1))+Y_test[i_Y]), axis=1)
        if i_Y == 0:
            XY_test = XY_comb
        else:
            XY_test = np.concatenate((XY_test, XY_comb), axis=0)
    
    print(XY_test.shape)
        
    prob_XY = parzen_window_est(XY_train, XY_test, 0.5).reshape((num_test,num_test))
    print('B')

    # Calculate H(X|Y)
    prob_Y = np.tile(prob_Y, (num_test,1)).T    # Arrage prob_Y into num_test X num_test
    CE = np.sum( prob_XY * np.log2(prob_Y/prob_XY) )
    print('C')

    return CE


def calculate_CE(signal, savePath):
    '''
    Calculate conditional entropy between each channels

    Parameters
    ----------
    signal : np.ndarray (epoch, channel, time)
        Signal data
    savePath : str
        Path for saving the data

    Returns
    -------
    CE : np.ndarray (epoch, features)
        Conditional entropy between each channels

    '''
    assert isinstance(signal, np.ndarray) and signal.ndim==3
    assert isinstance(savePath, str)
    
    num_channels = signal.shape[1]
    channel_perm = list(permutations(range(num_channels), 2))
    CE_all = np.zeros((signal.shape[0], len(channel_perm)))
    
    for i_perm, (i,j) in enumerate(channel_perm):
        for i_sample, sample in enumerate(signal):
            # Begin from 0 and round to integer
            signal1 = np.round(signal[i_sample, i, :] - np.min(signal[i_sample, i, :]))
            signal2 = np.round(signal[i_sample, j, :] - np.min(signal[i_sample, j, :]))
            
            CE = conditional_entropy(signal1, signal2)
            CE_all[i_sample, i_perm] = CE
            
    with open(savePath, 'wb') as fp:
        pickle.dump(CE_all, fp)
    
    return CE_all
    

def get_correlations(ERSP):
    '''
    Get correlation between channels

    Parameters
    ----------
    ERSP : np.ndarray (epoch, channel, features)
        Event-related spectral potential

    Returns
    -------
    correlation_all : np.ndarray (epoch, features)
        Correlation between channels of all trials
    '''
    assert isinstance(ERSP, np.ndarray) and ERSP.ndim==3
    
    channel_comb = list(combinations(list(range(ERSP.shape[1])), 2))
    correlation_all = np.zeros((ERSP.shape[0],len(channel_comb)))
    for i_comb, (i,j) in enumerate(channel_comb):
        for i_sample, sample in enumerate(ERSP):
            correlation = np.corrcoef(sample[i,:], sample[j,:])[0,1]
            correlation_all[i_sample, i_comb] = correlation
            
    return correlation_all

def get_bandpower_ratio(bp):
    '''
    Get bandpower ratios of within and across channels

    Parameters
    ----------
    bp : np.ndarray (epoch, channel, band)
        Bandpower 

    Returns
    -------
    None.

    '''
    assert isinstance(bp, np.ndarray) and bp.ndim==3
    
    print('Calculate bandpower ratio...')
    bp = bp.reshape((bp.shape[0],-1))
    feature_comb = list(combinations(range(bp.shape[1]), 2))
    bp_ratios = np.zeros((bp.shape[0], len(feature_comb)))
    
    for i_comb, (i,j) in enumerate(feature_comb):
        bp_ratios[:,i_comb] = bp[:,i]/bp[:,j]
    
    return bp_ratios
    
    

if __name__ == '__main__':
    
    '''
    # Load data
    mu_vec = np.array([0,0])
    cov_mat = np.array([[1,0],[0,1]])
    samples = np.random.multivariate_normal(mu_vec, cov_mat, 500)
    
    start_time = time.time()
    
    # Test parzen_window_est
    for h in np.linspace(0.1,1,num=10):
        pdf_est = parzen_window_est(samples, samples, h=h)
        pdf_actual = multivariate_normal.pdf(samples, mu_vec, cov_mat)
        
        print('[%f] Mean absolute error: %.3f'%(h, np.mean(np.abs(pdf_est-pdf_actual))))
    
    # Test get_conditional_entropy
    X, Y = samples[:,0].reshape((-1,1)), samples[:,1].reshape((-1,1))
    CE_X_Y = get_conditional_entropy(X, X, Y, Y)
    print(CE_X_Y)
    CE_Y_X = get_conditional_entropy(Y, Y, X, X)
    print(CE_Y_X)
    
    print('Takes %.3f seconds'%(time.time()-start_time))
    '''
    '''
    # Load preprocessed ERSP data
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    
    ERSP_all, tmp_all = preprocessing.remove_trials(ERSP_all, tmp_all, 60)
    ERSP_all, _ = preprocessing.standardize(ERSP_all, tmp_all)
    
    correlation_all = get_correlations(ERSP_all)
    '''
    
    # Load raw data
    for i in range(11):
        X, _, Y_reg, channels = raw_dataloader.read_data([1,2,3], date=[i], pred_type='class', rm_baseline=True)
        X, Y_reg = preprocessing.remove_trials(X, Y_reg, 60)
        _ = calculate_CE(X, './raw_data/CE_sub%d'%(i))
    
    X, _, Y_reg, channels = raw_dataloader.read_data([1,2,3], date=range(11), pred_type='class', rm_baseline=True)
    X, Y_reg = preprocessing.remove_trials(X, Y_reg, 60)
    _ = calculate_CE(X, './raw_data/CE_sub100')