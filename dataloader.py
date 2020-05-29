#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:33 2020

@author: hundredball
"""

import numpy as np
import pandas as pd
import scipy.io as sio

def extract_data(data, events, i_file, diff_type):
    '''
    Extract baseline data and solution time

    Parameters
    ----------
    data : numpy 2D array (j, k)
        j : channel
        k : sample
    events : numpy 2D array
        event time and type
    i_file : int
        index of the file in data_list.csv
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
    num_channel = data.shape[0]
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
            
    C = np.zeros(X.shape[0]) + i_file
    
    return X, Y, C
    

def read_data(diff_type, date = 0):
    '''
    Load data from Data_Python and transform them to input and labels

    Parameters
    ----------
    diff_type : iter
        difficulties of interest
    date : int
        chosen date

    Returns
    -------
    X : numpy 3D array (i, j, k)
        i : example
        j : channel
        k : sample
    Y : numpy 1D array
        solution latency of example
    C : numpy 1D array
        channel information of example

    '''
    
    assert hasattr(diff_type, '__iter__')
    assert isinstance(date, int)
    assert all((isinstance(x, int) and 1<=x<=3) for x in diff_type)
    
    # Get list of data names
    df_names = pd.read_csv('./Data_Matlab/data_list.csv')
    data_names = [x[0:6] for x in df_names.values.flatten()]
    data_names = [data_names[date]]
    
    # Iterate over each files
    X_list_diff_date = [[] for x in range(3)]   # Data
    Y_list_diff_date = [[] for x in range(3)]   # Labels
    C_list_diff_date = [[] for x in range(3)]   # Channel info
    
    for i_file, fileName in enumerate(data_names):
        EEG = sio.loadmat('./Data_Python/%s.mat'%(fileName))
        data = EEG['data']
        events = EEG['event']
        chanlocs = EEG['chanlocs']
        if data.shape[0]!=128:
            print('%s has only %d channels'%(fileName, data.shape[0]))
            continue
        
        # Extract channel label, theta, arc length
        num_channel = chanlocs.shape[1]
        chan_label, chan_theta, chan_arc_length = [], [], []
        
        for i in range(num_channel):
            chan_label.append(chanlocs[0,i][0][0])
            chan_theta.append(chanlocs[0,i][1][0,0])
            chan_arc_length.append(chanlocs[0,i][2][0,0])
            
        # create dataframe for them
        '''
        channels : pandas DataFrame (num_channel, 3)
            label : name of channels
            theta : 0 toward naison, positive for right hemisphere
            arc_length : arc_length of each channel, diameter of circle is 1
        '''
        d = {'label':chan_label, 'theta':chan_theta, 'arc_length':chan_arc_length}
        channels = pd.DataFrame(data=d)
        channels.to_csv('./Channel_coordinate/%s_channels.csv'%(fileName))
        
        X1, Y1, C1 = extract_data(data, events, i_file, '1')
        X2, Y2, C2 = extract_data(data, events, i_file, '2')
        X3, Y3, C3 = extract_data(data, events, i_file, '3')
                
        X_list_diff_date[0].append(X1.copy())
        Y_list_diff_date[0].append(Y1.copy())
        C_list_diff_date[0].append(C1.copy())
        
        X_list_diff_date[1].append(X2.copy())
        Y_list_diff_date[1].append(Y2.copy())
        C_list_diff_date[1].append(C2.copy())
        
        X_list_diff_date[2].append(X3.copy())
        Y_list_diff_date[2].append(Y3.copy())
        C_list_diff_date[2].append(C3.copy())
        
        
    # Concatenate over dates
    X_list_diff = []
    Y_list_diff = []
    C_list_diff = []
    for i in range(len(X_list_diff_date)):
        X_list_date = X_list_diff_date[i]
        Y_list_date = Y_list_diff_date[i]
        C_list_date = C_list_diff_date[i]
        
        for j in range(len(X_list_date)):
            if j == 0:
                X = X_list_date[j]
                Y = Y_list_date[j]
                C = C_list_date[j]
            else:
                X = np.concatenate((X,X_list_date[j]), axis=0)
                Y = np.concatenate((Y,Y_list_date[j]))
                C = np.concatenate((C,C_list_date[j]))
                
        X_list_diff.append(X.copy())
        Y_list_diff.append(Y.copy())
        C_list_diff.append(C.copy())
        
    print('Event 1 X shape: ', X_list_diff[0].shape)
    print('Event 2 X shape: ', X_list_diff[1].shape)
    print('Event 3 X shape: ', X_list_diff[2].shape)
    
    # Concatenate over difficulties
    for i in range(len(diff_type)):
        index_list = diff_type[i]-1
        if i == 0:
            X = X_list_diff[index_list]
            Y = Y_list_diff[index_list]
            C = C_list_diff[index_list]
        else:
            X = np.concatenate((X, X_list_diff[index_list]), axis=0)
            Y = np.concatenate((Y, Y_list_diff[index_list]))
            C = np.concatenate((C, C_list_diff[index_list]))
            
    print('Combined X shape: ', X.shape)
    
    # Remove trials with solution time more than 120 seconds
    chosen_trials = np.where(Y <= 120)[0]
    X = X[chosen_trials, :, :]
    Y = Y[chosen_trials]
    C = C[chosen_trials]
    
    print('After removing outliers, X shape: ', X.shape)
    
    C = C.astype('int')
    
    return X, Y, C



if __name__ == '__main__':
    
    X, Y, C = read_data([1,2,3], 0)
    print('X shape: ', X.shape)
    