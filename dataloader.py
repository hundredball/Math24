#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:33 2020

@author: hundredball
"""

import numpy as np
import pandas as pd
import scipy.io as sio

def extract_data(data, events, channels, diff_type):
    '''
    Extract baseline data and solution time

    Parameters
    ----------
    data : numpy 3D array (i, j, k)
        i : example
        j : channel
        k : sample
    events : numpy 2D array
        event time and type
    channels : numpy 2D array
        channel names
    diff_type : str
        difficulty of interest (1,2,3)

    Returns
    -------
    X : numpy 3D array (i, j, k)
        i : example
        j : channel
        k : sample
    Y : numpy 1D array
        solution latency of example

    '''
    assert isinstance(diff_type, str)
    
    sampling_rate = 256
    num_channel = channels.shape[0]
    num_epoch = sum([1 if x[1]==diff_type else 0 for x in events])
    X = np.zeros((num_epoch, num_channel, sampling_rate*2))
    Y = np.zeros(num_epoch)
    
    iter_event = 0
    for i in range(events.shape[0]):
        event = events[i]
        event_time = int(event[0])
        event_type = event[1]
        
        if event_type == diff_type:
            X[iter_event,:,:] = data[:, event_time-2*sampling_rate:event_time]
            Y[iter_event] = int(events[i+1,0] - event_time)/sampling_rate
            iter_event += 1
            
    return X, Y
    

def read_data(diff_type):
    '''
    Load data from Data_Python and transform them to input and labels

    Parameters
    ----------
    diff_type : iter
        difficulties of interest

    Returns
    -------
    X : numpy 3D array (i, j, k)
        i : example
        j : channel
        k : sample
    Y : numpy 1D array
        solution latency of example

    '''
    
    assert hasattr(diff_type, '__iter__')
    assert all((isinstance(x, int) and 1<=x<=3) for x in diff_type)
    
    # Get list of data names
    df_names = pd.read_csv('./Data_Matlab/data_list.csv')
    data_names = [x[0:6] for x in df_names.values.flatten()]
    
    # Get channel order
    df_channel = pd.read_csv('./Channel_coordinate/Channel_location_angle.csv')
    channel_order = df_channel['Channel'].to_numpy()
    
    # Iterate over each files
    X_list_diff_date = [[] for x in range(3)]
    Y_list_diff_date = [[] for x in range(3)]
    
    for fileName in data_names:
        EEG = sio.loadmat('./Data_Python/%s.mat'%(fileName))
        data = EEG['data']
        events = EEG['event']
        channels = EEG['chanloc_labels'].flatten()
        
        # Set channel order of the data
        reorder = [np.where(channels == x)[0][0] for x in channel_order]
        data = data[reorder,:]
        
        X1, Y1 = extract_data(data, events, channels, '1')
        X2, Y2 = extract_data(data, events, channels, '2')
        X3, Y3 = extract_data(data, events, channels, '3')
                
        X_list_diff_date[0].append(X1)
        Y_list_diff_date[0].append(Y1)
        X_list_diff_date[1].append(X2)
        Y_list_diff_date[1].append(Y2)
        X_list_diff_date[2].append(X3)
        Y_list_diff_date[2].append(Y3)
        
        
    # Concatenate over dates
    X_list_diff = []
    Y_list_diff = []
    for i in range(len(X_list_diff_date)):
        X_list_date = X_list_diff_date[i]
        Y_list_date = Y_list_diff_date[i]
        
        for j in range(len(X_list_date)):
            if j == 0:
                X = X_list_date[j]
                Y = Y_list_date[j]
            else:
                X = np.concatenate((X,X_list_date[j]), axis=0)
                Y = np.concatenate((Y,Y_list_date[j]))
                
        X_list_diff.append(X.copy())
        Y_list_diff.append(Y.copy())
        
    print('Event 1 X shape: ', X_list_diff[0].shape)
    print('Event 2 X shape: ', X_list_diff[1].shape)
    print('Event 3 X shape: ', X_list_diff[2].shape)
    # Concatenate over difficulties
    for i in range(len(diff_type)):
        index_list = diff_type[i]-1
        if i == 0:
            X = X_list_diff[index_list]
            Y = Y_list_diff[index_list]
        else:
            X = np.concatenate((X, X_list_diff[index_list]), axis=0)
            Y = np.concatenate((Y, Y_list_diff[index_list]))
            
    print('Combined X shape: ', X.shape)
    
    return X, Y, channel_order



if __name__ == '__main__':
    
    X, Y, channel_order = read_data([1,2,3])
    print('X shape: ', X.shape)
    print(channel_order)
    