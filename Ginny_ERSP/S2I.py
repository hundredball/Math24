#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:39:12 2020

@author: hundredball
"""

import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

import dataloader
import preprocessing
import sampling

def generate_topo(ERSP, freqs):
    '''
    Generate mixed topoplot

    Parameters
    ----------
    ERSP : numpy 3d array (epoch, channel, freq)
        Event-related spectral perturbation
    freqs : numpy 1d array
        Frequency steps
    Returns
    -------
    fileNames : numpy 1d array (epoch)
        file names of mixed images

    '''
    assert isinstance(ERSP, np.ndarray) and ERSP.ndim==3
    assert isinstance(freqs, np.ndarray) and freqs.ndim==1
    
    theta = preprocessing.bandpower(ERSP, freqs, 4, 8)
    alpha = preprocessing.bandpower(ERSP, freqs, 8, 14)
    beta = preprocessing.bandpower(ERSP, freqs, 14, 30)
    bandpower = np.concatenate((theta[:,:,np.newaxis], alpha[:,:,np.newaxis], beta[:,:,np.newaxis]), axis=2)
    
    # Read channel information
    channel_info = pd.read_csv('Channel_location.csv')
    channel_info = channel_info.to_numpy()
    num_channels = channel_info.shape[0]
    num_example = bandpower.shape[0]
    
    # Change coordinate from 0 toward naison to 0 toward right ear
    channel_info[:,2] = 90-channel_info[:,2]
    
    band_name = ['theta', 'alpha', 'beta']
    cmap_name = ['Reds', 'Greens', 'Blues']
    
    # Turn interactive plotting off
    plt.ioff()
    
    # Topoplot
    fileNames = np.empty(num_example, dtype=object)
    
    start_time = time.time()
    print('[%f] Generating topoplots...'%(time.time()-start_time))
    
    for i_data in range(num_example):
        
        # Plot topo for each band
        for i_band in range(3):
            fig, ax = plt.subplots(figsize=(4,4))
            n_angles = 48
            n_radii = 100
            radius = 0.9
            radii = np.linspace(0, radius, n_radii)
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    
            # Calculate channel locations on the plot
            scale_radius = radius/0.5
            scale_arc = scale_radius*channel_info[:,1]
            plot_loc = np.zeros((num_channels, 2)) # first for x, second for y
            plot_loc[:,0] = scale_arc*np.cos(np.array(channel_info[:,2]*np.pi/180, dtype = np.float))
            plot_loc[:,1] = scale_arc*np.sin(np.array(channel_info[:,2]*np.pi/180, dtype = np.float))
    
    
            # Add couple of zeros to outline for interpolation
            add_x = np.reshape(radius*np.cos(angles), (len(angles), 1))
            add_y = np.reshape(radius*np.sin(angles), (len(angles), 1))
            add_element = np.concatenate((add_x, add_y), axis=1)
            plot_loc = np.concatenate((plot_loc, add_element), axis=0)
            channel_values = np.concatenate((bandpower[i_data,:,i_band], np.zeros(len(angles))))
    
            # Interpolate 
            angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1) 
            x = (radii * np.cos(angles)).flatten()
            y = (radii * np.sin(angles)).flatten()
            z = griddata(plot_loc, channel_values, (x, y), method = 'cubic', fill_value=0, rescale=True)
            triang = tri.Triangulation(x, y)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            tcf = ax.tricontourf(triang, z, cmap = cmap_name[i_band], levels=50)   # Reds, Greens, Blues
            #fig.colorbar(tcf)
    
            # Add nose
            radius = 0.4   # radius on the plot
            height = (radius**2-0.04**2)**0.5
            ax.plot([-0.04,0], [0.5+height,1], color='black')
            ax.plot([0,0.04], [1,0.5+height], color='black')
    
            # Add ears
            ax.plot([-0.91,-0.96], [0.05, 0.1], color='black')
            ax.plot([-0.96,-0.96], [0.1, -0.1], color='black')
            ax.plot([-0.96,-0.91], [-0.1, -0.05], color='black')
            ax.plot([0.91,0.96], [0.05, 0.1], color='black')
            ax.plot([0.96,0.96], [0.1, -0.1], color='black')
            ax.plot([0.96,0.91], [-0.1, -0.05], color='black')
    
            ax.axis('off')
            plt.savefig('./images/%d_%s.png'%(i_data, band_name[i_band]))
            
        # Combine three plots
        figure_r = plt.imread('./images/%d_%s.png'%(i_data, band_name[0]))
        figure_g = plt.imread('./images/%d_%s.png'%(i_data, band_name[1]))
        figure_b = plt.imread('./images/%d_%s.png'%(i_data, band_name[2]))
    
        figure_mix = (figure_r+figure_g+figure_b)/3
        fileName = '%d_mix'%(i_data)
        plt.imsave('./images/%s.png'%(fileName), figure_mix)
        
        fileNames[i_data] = fileName
        
    print('[%f] Finished all topoplots!'%(time.time()-start_time))
        
    return fileNames
    
def split(fileNames, SLs, test_size=0.1, random=True):
    '''
    Split training and testing set by creating csv files for referencing

    Parameters
    ----------
    fileNames : numpy 1d array (epoch)
        File names of mixed images
    SLs : numpy 1d array (epoch)
        Solution latency of those trials
    test_ratio : float, optional
        Ration of testing set. The default is 0.1.

    Returns
    -------
    None.

    '''
    assert isinstance(fileNames, np.ndarray) and fileNames.ndim == 1
    assert isinstance(SLs, np.ndarray) and SLs.ndim == 1
    assert isinstance(random, bool)
    assert (random and isinstance(test_size, float) and 0<=test_size<=1) or\
        ((not random) and isinstance(test_size, int) and 0<=test_size<=fileNames.shape[0])
    assert fileNames.shape[0] == SLs.shape[0]
    
    # Split for training and testing data
    if not random:
        X_train, X_test = fileNames[:-test_size], fileNames[-test_size:]
        Y_train, Y_test = SLs[:-test_size], SLs[-test_size:]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(fileNames, SLs, test_size=test_size, random_state=42)
    
    # Save csv for dataloader
    X_train_df = pd.DataFrame({'fileName':X_train})
    X_train_df.to_csv('./images/train_img.csv')
    
    X_test_df = pd.DataFrame({'fileName':X_test})
    X_test_df.to_csv('./images/test_img.csv')
    
    Y_train_df = pd.DataFrame({'solution_time':Y_train})
    Y_train_df.to_csv('./images/train_label.csv')
    
    Y_test_df = pd.DataFrame({'solution_time':Y_test})
    Y_test_df.to_csv('./images/test_label.csv')
    
    print('Generate files for dataset referencing')

if __name__ == '__main__':
    
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    
    '''
    # Take first 7 samples
    ERSP_part, tmp_part = ERSP_all[:7,:,:,:], tmp_all[:7,:]
    ERSP_part, SLs = preprocessing.standardize(ERSP_part, tmp_part)
    
    # Test generate_topo
    fileNames = generate_topo(ERSP_part, freqs)
    
    # Test split
    split(fileNames, SLs, 0.2)
    '''
    '''
    
    # -----Generate topoplot for all trials-----
    
    ERSP_all, tmp_all = preprocessing.remove_trials(ERSP_all, tmp_all, 25)
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all)
    
    fileNames = generate_topo(ERSP_all, freqs)
    split(fileNames, SLs, 0.1)
    '''
    
    # -----Generate topoplot after SMOTER-----
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all)
    
    # Split data
    ERSP_train, ERSP_test, SLs_train, SLs_test = train_test_split(ERSP_all, SLs, test_size=0.1, random_state=42)
    
    # SMOTER on training data
    ERSP_train = ERSP_train.reshape((ERSP_train.shape[0], -1))
    ERSP_train, SLs_train = sampling.SMOTER(ERSP_train, SLs_train)
    ERSP_train = ERSP_train.reshape((ERSP_train.shape[0], 12, -1))
    
    # Concatenate trainind and testing data
    ERSP_concat = np.concatenate((ERSP_train, ERSP_test), axis=0)
    SLs_concat = np.concatenate((SLs_train, SLs_test), axis=0)
    
    fileNames = generate_topo(ERSP_concat, freqs)
    split(fileNames, SLs_concat, len(SLs_test), random = False)
    