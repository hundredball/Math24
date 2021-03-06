#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:11:22 2020

@author: hundredball
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from sklearn.metrics import mean_squared_error
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate results')
parser.add_argument('-n', '--folder_name', default=None, type=str, help='Name of folder in results')
parser.add_argument('-e', '--num_exp', default=10, type=int, help='Number of experiments in the folder')
parser.add_argument('--ensemble', dest='ensemble', action='store_true', help='Evaluate ensemble results')
parser.add_argument('--deepex', dest='deepex', action='store_true', help='Deep network extraction with other regression model')

def plot_error(dict_error, dirName, fileName, mode='reg'):
    '''
    Plot the error curve of training and testing data

    Parameters
    ----------
    dict_error : dictionary
        Dictionary containing train_std, val_std, test_std, train_mape, val_mape, test_mape, pred, target
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
    
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    
    if mode == 'reg':
        epoch = list(range(len(dict_error['train_std'])))
        
        axs[0].plot(epoch, dict_error['train_std'], 'r-', epoch, dict_error['val_std'], 'b--')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Standard error')
        axs[0].legend(('Train', 'Validate'))
        axs[0].set_title('Last std: (%.3f,%.3f,%.3f)'%(dict_error['train_std'][-1], dict_error['val_std'][-1],dict_error['test_std'][-1]))
        
        axs[1].plot(epoch, dict_error['train_mape'], 'r-', epoch, dict_error['val_mape'], 'b--')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('MAPE')
        axs[1].legend(('Train', 'Validate'))
        axs[1].set_title('Last MAPE: (%.3f,%.3f,%.3f)'%(dict_error['train_mape'][-1], dict_error['val_mape'][-1],dict_error['test_mape'][-1]))
    elif mode == 'class':
        epoch = list(range(len(dict_error['train_loss'])))
        
        axs[0].plot(epoch, dict_error['train_loss'], 'r-', epoch, dict_error['val_loss'], 'b--')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Cross-entropy loss')
        axs[0].legend(('Train', 'Test'))
        axs[0].set_title('Last loss: (%.3f,%.3f)'%(dict_error['train_loss'][-1], dict_error['test_loss'][-1]))
        
        axs[1].plot(epoch, dict_error['train_acc']*100, 'r-', epoch, dict_error['val_acc']*100, 'b--')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].legend(('Train', 'Test'))
        axs[1].set_title('Last accuracy: (%.1f,%.1f)'%(dict_error['train_acc'][-1]*100, dict_error['test_acc'][-1]*100))
    
    plt.suptitle(fileName)
    plt.tight_layout(pad=2.0)
    plt.savefig('./results/%s/%s_error.png'%(dirName, fileName))
    plt.close()
    
def plot_scatter(true, pred, dirName, fileName, subIDs=None):
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
    subIDs : iterator
        Subject ID, default=None

    Returns
    -------
    None.

    '''
    assert hasattr(true, '__iter__')
    assert hasattr(pred, '__iter__')
    assert len(true)==len(pred)
    assert isinstance(dirName, str)
    assert isinstance(fileName, str)
    
    colors = ['black', 'yellow', 'sienna', 'purple', 'olive', 'navy', 'lime', 'green', 'cyan', 'darkblue', 'red', 'grey']
    
    sort_indices = np.argsort(true)
    fig, axs = plt.subplots(1,2, figsize=(16,8))
    mape = np.sum( np.abs( (true-pred)/true ) ) / true.shape[0]
    if subIDs is not None and len(np.unique(subIDs))>1:
        assert len(subIDs) == len(pred)
        
        sorted_subIDs = subIDs[sort_indices]
        unique_subIDs = np.unique(sorted_subIDs)
        axs[0].plot(range(len(true)), true[sort_indices], marker='.', linestyle='', color=colors[0], label='True')
        for i_sub, subID in enumerate(unique_subIDs):
            sub_indices = np.nonzero(sorted_subIDs==subID)[0]
            axs[0].plot(sub_indices, pred[sort_indices[sub_indices]], marker='.', linestyle='', color=colors[i_sub+1], label='Pred_%d'%(subID))
        axs[0].legend()
    else:
        axs[0].plot(range(len(true)), true[sort_indices], 'r.', range(len(true)), pred[sort_indices], 'b.')
        axs[0].legend(('True', 'Pred'))
    axs[0].set_xlabel('Record number')
    axs[0].set_ylabel('Solution latency')
    axs[0].set_title('std error = %.3f, MAPE = %.3f'%(mean_squared_error(true, pred)**0.5, mape))
    
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
    
    if args.ensemble:
        index_split = 100
    else:
        index_split = 0
    
    for i in range(args.num_exp):
        if args.deepex:
            data_file = './results/%s/%s_exp%d.data'%(args.folder_name, args.folder_name, i)
        else:
            data_file = './results/%s/%s_split%d_exp%d.data'%(args.folder_name, args.folder_name, index_split, i)
        with open(data_file, 'rb') as fp:
            dict_error = pickle.load(fp)
        target, pred = dict_error['target'], dict_error['pred']
        
        if not args.deepex:
            std, mape = dict_error['test_std'][-1], dict_error['test_MAPE'][-1]
            std_all += std
            mape_all += mape
        
        if i == 0:
            target_all, pred_all = target, pred
        else:
            target_all = np.concatenate((target_all, target))
            pred_all = np.concatenate((pred_all, pred))
        
    if not args.deepex:
        std_all /= args.num_exp
        mape_all /= args.num_exp
        print('Average standard error of all experiments: %.3f'%(std_all))
        print('Average Mean Absolute Percentage Error of all experiments : %.3f'%(mape_all))
        
        fileName = '%s, Std error: %.3f, MAPE: %.3f'%(args.folder_name, std_all, mape_all)
    else:
        fileName = '%s_allFolds'%(args.folder_name)
    
    # Plot scatter
    plot_scatter(target_all, pred_all, args.folder_name, fileName)
    