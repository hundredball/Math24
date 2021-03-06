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
        Indices of channel file

    '''
    assert isinstance(diff_type, str)
    assert isinstance(rm_baseline, bool)
    
    sampling_rate = 256
    time_baseline = int(sampling_rate/2)
    num_channel = data.shape[0]
    X = []
    Y = []
    
    for i in range(len(events)-1):
        # In event : [time, type]
        cur_event = events[i]
        next_event = events[i+1]
        cur_time = int(cur_event[0])
        
        if cur_event[1] == diff_type and next_event[1] == '501':
            X.append(data[:, cur_time-2*sampling_rate+1:cur_time+1])
            if rm_baseline:
                #baseline = np.mean(data[:, int(events[i-1,0])-time_baseline:int(events[i-1,0])], axis=1)
                baseline = np.mean(data[:, cur_time-int(2.5*sampling_rate):cur_time-int(2*sampling_rate)], axis=1)
                X[-1] -= baseline[:,np.newaxis] 
            
            Y.append(float((next_event[0] - cur_time)/sampling_rate))
          
    X = np.array(X)
    Y = np.array(Y)
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

def read_channel_name(channel_limit=21):
    '''
    Load channel names
    '''
    assert channel_limit in [12,21]
    
    # Order of channels
    if channel_limit == 12:
        channel_order = pd.read_csv('./Channel_coordinate/Channel_location_angle_12.csv')['Channel'].values
    elif channel_limit == 21:
        channel_order = pd.read_csv('./Channel_coordinate/Channel_location_angle_21.csv')['Channel'].values
        
    return channel_order
    
def read_data(diff_type, date = [0], channel_limit = 12, rm_baseline = False, SL_threshold=60):
    '''
    Load data from Data_Python and transform them into input and labels

    Parameters
    ----------
    diff_type : iter
        difficulties of interest (1,2,3)
    date : iter
        chosen dates
    channel_limit : int
        Number of channels of loaded dataset (12, 21)
    rm_baseline : bool
        Remove baseline before fixation
    SL_threshold : float
        Remove trials with solution latency higher than threshold
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
    S : numpy 1D array
        subject ID
    D : numpy 1D array
        difficulty level

    '''
    
    assert hasattr(diff_type, '__iter__')
    assert hasattr(date, '__iter__')
    assert all((isinstance(x, int) and 1<=x<=3) for x in diff_type)
    assert isinstance(rm_baseline, bool)
    assert channel_limit == 12 or channel_limit == 21
    assert SL_threshold>0
    
    # Assign path depending on regression or classification
    if channel_limit == 21:
        EEG_path = 'Data_Python'
        coord_subFileName = '_channels_21.csv'
    elif channel_limit == 12:
        EEG_path = 'Data_Python_Replicate'
        coord_subFileName = '_channels_12.csv'
    
    # Get list of data names
    df_names = pd.read_csv('%s/Data_Matlab/data_list.csv'%(root_path))
    data_names = [x[0:6] for x in df_names.values.flatten()]
    data_names = [data_names[i] for i in date]
    
    # Iterate over each files
    X_list_diff_date = [[] for x in range(len(diff_type))]   # Data
    Y_list_diff_date = [[] for x in range(len(diff_type))]   # Labels
    C_list_diff_date = [[] for x in range(len(diff_type))]   # Channel info
    S_list_diff_date = [[] for x in range(len(diff_type))]   # Subject ID
    D_list_diff_date = [[] for x in range(len(diff_type))]   # Difficulty level
    
    for i_file, fileName in enumerate(data_names):
        if channel_limit == 12:
            EEG = sio.loadmat('%s/%s/%s.mat'%(root_path, EEG_path, fileName))
        elif channel_limit == 21:
            EEG = sio.loadmat('%s/%s/%s_21.mat'%(root_path, EEG_path, fileName))
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
            S_list_diff_date[i_diff].append( np.ones(len(X_sub[diff]))*i_file )
            D_list_diff_date[i_diff].append( np.zeros(len(X_sub[diff]))+i_diff )
        
    # Concatenate over dates
    X_list_diff = []
    Y_list_diff = []
    C_list_diff = []
    S_list_diff = []
    D_list_diff = []
    for i in range(len(X_list_diff_date)):
        X_list_date = X_list_diff_date[i]
        Y_list_date = Y_list_diff_date[i]
        C_list_date = C_list_diff_date[i]
        S_list_date = S_list_diff_date[i]
        D_list_date = D_list_diff_date[i]
        
        for j in range(len(X_list_date)):
            if j == 0:
                X = X_list_date[j]
                Y = Y_list_date[j]
                C = C_list_date[j]
                S = S_list_date[j]
                D = D_list_date[j]
                
            else:
                X = np.concatenate((X,X_list_date[j]), axis=0)
                Y = np.concatenate((Y,Y_list_date[j]))
                C = np.concatenate((C,C_list_date[j]))
                S = np.concatenate((S,S_list_date[j]))
                D = np.concatenate((D,D_list_date[j]))
                
        X_list_diff.append(X.copy())
        Y_list_diff.append(Y.copy())
        C_list_diff.append(C.copy())
        S_list_diff.append(S.copy())
        D_list_diff.append(D.copy())
    
    # Concatenate over difficulties
    for i, diff in enumerate(diff_type):
        
        #print('Event %s X shape: %s'%(str(diff), X_list_diff[i].shape))
            
        if i == 0:
            X = X_list_diff[i]
            Y = Y_list_diff[i]
            C = C_list_diff[i]
            S = S_list_diff[i]
            D = D_list_diff[i]
        else:
            X = np.concatenate((X, X_list_diff[i]), axis=0)
            Y = np.concatenate((Y, Y_list_diff[i]))
            C = np.concatenate((C, C_list_diff[i]))
            S = np.concatenate((S, S_list_diff[i]))
            D = np.concatenate((D, D_list_diff[i]))
            
    #print('Combined X shape: ', X.shape)
    C = C.astype('int')
    S = S.astype('int')
    
    # Remove trials with solution time more than SL_threshold seconds
    chosen_trials = np.where(Y <= SL_threshold)[0]
    X = X[chosen_trials, :, :]
    Y = Y[chosen_trials]
    C = C[chosen_trials]
    S = S[chosen_trials]
    D = D[chosen_trials]
    
    print('After removing trials longer than 60s, X shape: ', X.shape)
    
    print('Arrange all the channels as the same order\n')
    
    # Order of channels
    channel_order = read_channel_name(channel_limit)
    
    # Arrange all the channels in the same order
    for i in range(X.shape[0]):
        date = C[i]
    
        # Read channel locations
        channel_info = pd.read_csv('%s/Channel_coordinate/%s'%(root_path,data_names[date]+coord_subFileName))
        channel_info = channel_info.to_numpy()
        
        # Change the order of data
        temp_X = np.array([X[i, np.where(channel_order[j]==channel_info[:,1])[0],:] for j in range(X.shape[1])])
        temp_X = temp_X.reshape((X.shape[1], -1))
        X[i,:] = temp_X
    
    return X, Y, C, S, D


if __name__ == '__main__':
    
    # Test read_data
    X, Y, C, S, D = read_data([1,2,3], list(range(11)), channel_limit=21, rm_baseline = True)
    print('X shape: ', X.shape)
    