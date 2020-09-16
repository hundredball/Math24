#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:11:22 2020

@author: hundredball
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate results')
parser.add_argument('-n', '--folder_name', default=None, type=str, help='Name of folder in results')
parser.add_argument('-e', '--num_exp', default=10, type=int, help='Number of experiments in the folder')

def plot_error(dict_error, dirName, fileName):
    '''
    Plot the error curve of training and testing data

    Parameters
    ----------
    dict_error : dictionary
        Dictionary containing train_std, test_std, train_MAPE, test_MAPE, pred, target
    dirName : str
        Directory after results
    fileName : string, optional
        Name for file and title

    Returns
    -------
    None.

    '''
    assert isinstance(dirName, str)
    assert isinstance(fileName, str)
    
    epoch = list(range(len(dict_error['train_std'])))
    
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].plot(epoch, dict_error['train_std'], 'r-', epoch, dict_error['test_std'], 'b--')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Standard error')
    axs[0].legend(('Train', 'Test'))
    
    axs[1].plot(epoch, dict_error['train_MAPE'], 'r-', epoch, dict_error['test_MAPE'], 'b--')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAPE')
    axs[1].legend(('Train', 'Test'))
    
    plt.suptitle(fileName)
    plt.savefig('./results/%s/%s_error.png'%(dirName, fileName))
    plt.close()
    
def plot_scatter(true, pred, dirName, fileName):
    '''
    Plot the scatter plots of true target and prediction

    Parameters
    ----------
    true : iterator
        Target
    pred : iterator
        Prediction
    dirName : str
        Directory after results
    fileName : str
        File name

    Returns
    -------
    None.

    '''
    assert hasattr(true, '__iter__')
    assert hasattr(pred, '__iter__')
    assert isinstance(dirName, str)
    assert isinstance(fileName, str)
    
    sort_indices = np.argsort(true)
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].plot(range(len(true)), true[sort_indices], 'r.', range(len(true)), pred[sort_indices], 'b.')
    axs[0].set_xlabel('Record number')
    axs[0].set_ylabel('Solution latency')
    axs[0].legend(('True', 'Pred'))
    
    max_value = np.max(np.hstack((true, pred)))
    axs[1].scatter(true, pred, marker='.')
    axs[1].plot(range(int(max_value)), range(int(max_value)), 'r')
    axs[1].set_xlabel('True')
    axs[1].set_ylabel('Pred')
    axs[1].set_xlim([0, max_value])
    axs[1].set_ylim([0, max_value])
    axs[1].set_title('r = %.3f'%(np.corrcoef(true, pred)[0,1]))
    
    plt.suptitle(fileName)
    
    plt.savefig('./results/%s/%s_scatter.png'%(dirName, fileName))
    plt.close(fig)
    
if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    # Combine results from different experiments
    std_all = 0
    mape_all = 0
    for i in range(args.num_exp):
        with open('./results/%s/%s_exp%d.data'%(args.folder_name, args.folder_name, i), 'rb') as fp:
            dict_error = pickle.load(fp)
        target, pred = dict_error['target'], dict_error['pred'],
        std, mape = dict_error['test_std'][-1], dict_error['test_MAPE'][-1]
        std_all += std
        mape_all += mape
        
        if i == 0:
            target_all, pred_all = target, pred
        else:
            target_all = np.concatenate((target_all, target))
            pred_all = np.concatenate((pred_all, pred))
        
    std_all /= args.num_exp
    mape_all /= args.num_exp
    print('Average standard error of all experiments: %.3f'%(std_all))
    print('Average Mean Absolute Percentage Error of all experiments : %.3f'%(mape_all))
    
    # Plot scatter
    fileName = '%s, Std error: %.3f, MAPE: %.3f'%(args.folder_name, std_all, mape_all)
    plot_scatter(target_all, pred_all, args.folder_name, fileName)
    