#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:15:15 2020

@author: hundredball
"""
import numpy as np
import scipy.io as sio
import pandas as pd
import os,inspect
root_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def load_data():
    '''
    Load data from Matlab to Python

    Returns
    -------
    ERSP_all : 4d numpy array (epoch, channel, freq_step, time_step)
        ERSP of all trials
    tmp_all : 2d numpy array (epoch, time_periods)
        time_periods include time points of fixation, cue, end
    freqs : 1d numpy array
        frequency step of ERSP

    '''
    
    # Get list of data names
    df_names = pd.read_csv('%s/Data_Matlab/data_list.csv'%(root_path))
    dates = [x[0:6] for x in df_names.values.flatten()]
    
    # Concatenate over dates
    for i_date, date in enumerate(dates):
        data = sio.loadmat('%s/savedata/%s/%s_python.mat'%(root_path, date, date))
        ERSP = data['ERSP']
        tmp = data['tmp']
        if i_date == 0:
            ERSP_all = ERSP
            tmp_all = tmp
        else: 
            ERSP_all = np.concatenate((ERSP_all, ERSP), axis=0)
            tmp_all = np.concatenate((tmp_all, tmp), axis=0)
            
    freqs = data['freqs'].flatten()    # freqs of all dates are equal
    
    # ERSP_all (epoch, channel, freq_step, time_step)
    # tmp_all (epoch, time_periods)
    # freqs (freq_step)
    print('Shape of ERSP_all: ', ERSP_all.shape)
    print('Shape of tmp_all: ', tmp_all.shape)
    print('Shape of freqs: ', freqs.shape)
    
    return ERSP_all, tmp_all, freqs
    
if __name__ == '__main__':
    ERSP_all, tmp_all, freqs = load_data()