#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:36:33 2020

@author: hundredball
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import os,sys,inspect

root_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def extract_data(data, events, i_file, diff_type, rm_baseline):
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
    rm_baseline : str
        Remove baseline before fixation

    Returns
    -------
    X : numpy 3D array (i, j, k)
        i : example
        j : channel
        k : sample
    Y : numpy 1D array
        solution latency of example
    C : numpy 1D array
        Indices of channel file

    '''
    assert isinstance(diff_type, str)
    assert isinstance(rm_baseline, bool)
    
    sampling_rate = 256
    time_baseline = int(sampling_rate/2)
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
            
            X[iter_event,:,:] = data[:, event_time-2*sampling_rate+1:event_time+1]
            if rm_baseline:
                baseline = np.mean(data[:, int(events[i-1,0])-time_baseline:int(events[i-1,0])], axis=1)
                X[iter_event,:,:] -= baseline[:,np.newaxis] 
            
            Y[iter_event] = int(events[i+1,0] - event_time)/sampling_rate
            iter_event += 1
            
    C = np.zeros(X.shape[0]) + i_file
    
    return X, Y, C
    
def generate_class(Y_SL):
    '''
    Generate class label by the mean of all the trials
    
    Parameters
    ----------
    Y_SL : numpy 1D array
        solution latency of each trial
    
    Return
    ------
    Y_class : numpy 1D array
        labels of each trial
    
    '''
    assert isinstance(Y_SL, np.ndarray)
    
    threshold = np.mean(Y_SL)
    print('Mean of all trials: %f'%(threshold))
    
    Y_class = Y_SL.copy()
    Y_class[Y_class<threshold] = 0
    Y_class[Y_class>=threshold] = 1
    
    return Y_class
    
    
def read_data(diff_type, date = [0], pred_type = 'reg', rm_baseline = False):
    '''
    Load data from Data_Python and transform them into input and labels

    Parameters
    ----------
    diff_type : iter
        difficulties of interest (1,2,3)
    date : iter
        chosen dates
    pred_type : str
        regression (reg) or classification (class)
    rm_baseline : bool
        Remove baseline before fixation

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
    assert hasattr(date, '__iter__')
    assert all((isinstance(x, int) and 1<=x<=3) for x in diff_type)
    assert isinstance(rm_baseline, bool)
    
    # Assign path depending on regression or classification
    if pred_type == 'reg':
        channel_limit = 128
        EEG_path = 'Data_Python'
        coord_subFileName = '_channels.csv'
    else:
        channel_limit = 12
        EEG_path = 'Data_Python_Replicate'
        coord_subFileName = '_channels_class.csv'
    
    # Get list of data names
    df_names = pd.read_csv('%s/Data_Matlab/data_list.csv'%(root_path))
    data_names = [x[0:6] for x in df_names.values.flatten()]
    data_names = [data_names[i] for i in date]
    
    # Iterate over each files
    X_list_diff_date = [[] for x in range(len(diff_type))]   # Data
    Y_list_diff_date = [[] for x in range(len(diff_type))]   # Labels
    C_list_diff_date = [[] for x in range(len(diff_type))]   # Channel info
    
    for i_file, fileName in enumerate(data_names):
        EEG = sio.loadmat('%s/%s/%s.mat'%(root_path, EEG_path, fileName))
        data = EEG['data']
        events = EEG['event']
        chanlocs = EEG['chanlocs']
        if data.shape[0]!=channel_limit:
            print('%s has only %d channels'%(fileName, data.shape[0]))
            continue
        
        # Extract channel label, theta, arc length
        num_channel = chanlocs.shape[1]
        chan_label, chan_theta, chan_arc_length = [], [], []
        
        for i in range(num_channel):
            chan_label.append(chanlocs['labels'][0,i][0])
            chan_theta.append(chanlocs['theta'][0,i][0,0])
            chan_arc_length.append(chanlocs['radius'][0,i][0,0])
            
        # create dataframe for them
        '''
        channels : pandas DataFrame (num_channel, 3)
            label : name of channels
            theta : 0 toward naison, positive for right hemisphere
            arc_length : arc_length of each channel, diameter of circle is 1
        '''
        d = {'label':chan_label, 'theta':chan_theta, 'arc_length':chan_arc_length}
        channels = pd.DataFrame(data=d)
        channels.to_csv('%s/Channel_coordinate/%s'%(root_path, fileName+coord_subFileName))
        
        
        X_sub, Y_sub, C_sub = {}, {}, {}
        
        for i_diff, diff in enumerate(diff_type):
            X_sub[diff], Y_sub[diff], C_sub[diff] = extract_data(data, events, i_file, str(diff), rm_baseline)
            
            X_list_diff_date[i_diff].append(X_sub[diff].copy())
            Y_list_diff_date[i_diff].append(Y_sub[diff].copy())
            C_list_diff_date[i_diff].append(C_sub[diff].copy())
        
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
    
    # Concatenate over difficulties
    for i, diff in enumerate(diff_type):
        
        #print('Event %s X shape: %s'%(str(diff), X_list_diff[i].shape))
            
        if i == 0:
            X = X_list_diff[i]
            Y = Y_list_diff[i]
            C = C_list_diff[i]
        else:
            X = np.concatenate((X, X_list_diff[i]), axis=0)
            Y = np.concatenate((Y, Y_list_diff[i]))
            C = np.concatenate((C, C_list_diff[i]))
            
    #print('Combined X shape: ', X.shape)
    C = C.astype('int')
    
    # Remove trials with solution time more than 120 seconds
    chosen_trials = np.where(Y <= 120)[0]
    X = X[chosen_trials, :, :]
    Y = Y[chosen_trials]
    C = C[chosen_trials]
    
    print('After removing outliers, X shape: ', X.shape)
    
    print('Arrange all the channels as the same order\n')
    
    # Order of channels
    channel_order = pd.read_csv('%s/Channel_coordinate/Channel_location_angle.csv'%(root_path))['Channel'].values
    
    # Arrange all the channels in the same order
    for i in range(X.shape[0]):
        date = C[i]
    
        # Read channel locations
        channel_info = pd.read_csv('%s/Channel_coordinate/%s_channels_class.csv'%(root_path,data_names[date]))
        channel_info = channel_info.to_numpy()
        
        # Change the order of data
        temp_X = np.array([X[i, np.where(channel_order[j]==channel_info[:,1])[0],:] for j in range(X.shape[1])])
        temp_X = temp_X.reshape((X.shape[1], -1))
        X[i,:] = temp_X
    
    # Return data, channel and Y_class, Y_reg depending on mode
    if pred_type == 'class':
        Y_class = generate_class(Y)
        return X, Y_class, Y, C
    else:
        return X, Y, C


if __name__ == '__main__':
    
    # Test read_data
    X, Y_class, Y_reg, C = read_data([1,2,3], list(range(11)), pred_type='class', rm_baseline = True)
    print('X shape: ', X.shape)
    