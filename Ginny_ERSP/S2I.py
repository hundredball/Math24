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
import argparse
import pickle

import dataloader
import preprocessing
import data_augmentation

parser = argparse.ArgumentParser(description='Signal to image')
parser.add_argument('-m', '--mode', default='normal', type=str, help='Generating method')
parser.add_argument('-d', '--data_cate', default=1, type=int, help='Category of data')
parser.add_argument('-t', '--num_time', default=1, type=int, help='Number of frames for each example')


def generate_topo(ERSP, freqs, num_time=1, train_indices = None):
    '''
    Generate mixed topoplot

    Parameters
    ----------
    ERSP : numpy 3d or 4d array (epoch, channel, freq, (time))
        Event-related spectral perturbation
    freqs : numpy 1d array
        Frequency steps
    num_time : int
        Number of time steps
    Returns
    -------
    fileNames : numpy 1d array (epoch)
        file names of mixed images

    '''
    assert isinstance(ERSP, np.ndarray)
    assert isinstance(freqs, np.ndarray) and freqs.ndim==1
    assert isinstance(num_time, int) and num_time >= 1
    assert (ERSP.ndim==3 and num_time==1) or (ERSP.ndim==4 and ERSP.shape[3]==num_time)
    if train_indices is None:
        train_indices = np.arange(ERSP.shape[0])
    else:
        assert isinstance(train_indices, np.ndarray) and train_indices.ndim==1
    
    if ERSP.ndim==3:
        ERSP_all = np.expand_dims(ERSP, axis=3)
    else:
        ERSP_all = ERSP.copy()
        
    num_example = ERSP.shape[0]
    fileNames = np.empty(num_example, dtype=object)
    dict_img = {}
    
    start_time = time.time()
    print('[%f] Generating topoplots...'%(time.time()-start_time))
    
    for i_time in range(num_time):
        
        print('[%f] Time step: %d ...'%(time.time()-start_time, i_time))
        
        ERSP = ERSP_all[:,:,:,i_time]
        theta = preprocessing.bandpower(ERSP, freqs, 4, 8)
        alpha = preprocessing.bandpower(ERSP, freqs, 8, 14)
        beta = preprocessing.bandpower(ERSP, freqs, 14, 30)
        bandpower = np.concatenate((theta[:,:,np.newaxis], alpha[:,:,np.newaxis], beta[:,:,np.newaxis]), axis=2)

        # Read channel information
        channel_info = pd.read_csv('Channel_location.csv')
        channel_info = channel_info.to_numpy()
        num_channels = channel_info.shape[0]

        # Change coordinate from 0 toward naison to 0 toward right ear
        channel_info[:,2] = 90-channel_info[:,2]

        band_name = ['theta', 'alpha', 'beta']
        cmap_name = ['Reds', 'Greens', 'Blues']

        # Turn interactive plotting off
        plt.ioff()

        # Topoplot

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
                plt.savefig('./images/%s.png'%(band_name[i_band]))
                plt.close()

            # Combine three plots
            figure_r = plt.imread('./images/%s.png'%(band_name[0]))
            figure_g = plt.imread('./images/%s.png'%(band_name[1]))
            figure_b = plt.imread('./images/%s.png'%(band_name[2]))
            
            if i_data in train_indices:
                dirName = 'train'
            else:
                dirName = 'test'

            figure_mix = (figure_r+figure_g+figure_b)/3
            fileName = '%d_mix_%d'%(i_data, i_time)
            plt.imsave('./images/%s/%s.png'%(dirName, fileName), figure_mix)
            # dict_img[fileName] = np.floor(figure_mix*255)

            # Only save fileNames for the first time step
            if i_time == 0:
                fileNames[i_data] = fileName
        
    print('[%f] Finished all topoplots!'%(time.time()-start_time))
    
    with open('./images/img.data', 'wb') as fp:
        pickle.dump(dict_img, fp)
        
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
    random : bool, optional
        The training or testing data in fileNames are random or not. The default is True.
        
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
    
def S2I_main(ERSP_all, tmp_all, freqs, indices, mode, num_time):
    '''
    Standardize data, then generate topo

    Parameters
    ----------
    ERSP_all : numpy 4d array
        Event related spectral perturbations
    tmp_all : numpy 1d or 2d array
        Periods of time or solution latency
    freqs : numpy 1d array
        Frequency steps
    indices : dict
        Indices of training and testing data
    mode : string
        Multiframe or single frame or SMOTE

    Returns
    -------
    None.

    '''
    assert isinstance(ERSP_all, np.ndarray) and ERSP_all.ndim == 4
    assert isinstance(tmp_all, np.ndarray) and (tmp_all.ndim == 1 or tmp_all.ndim == 2)
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1
    assert isinstance(indices, dict)
    assert isinstance(mode, str)
    assert isinstance(num_time, int)
    
    # Standardize the data
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all, num_time, train_indices=indices['train'])
    ERSP_dict = {kind : ERSP_all[indices[kind],:] for kind in ['train','test']}
    SLs_dict = {kind : SLs[indices[kind]] for kind in ['train','test']}
    
    # Data augmentation
    if mode == 'SMOTER':
        ERSP_dict['train'], SLs_dict['train'] = data_augmentation.aug(ERSP_dict['train'], SLs_dict['train'], 'SMOTER')
    elif mode == 'add_noise':
        ERSP_dict['train'], SLs_dict['train'] = data_augmentation.aug(ERSP_dict['train'], SLs_dict['train'], 
                                                                      'add_noise', (10,1))
    
    # Concatenate training and testing data
    ERSP_concat = np.concatenate((ERSP_dict['train'], ERSP_dict['test']), axis=0)
    SLs_concat = np.concatenate((SLs_dict['train'], SLs_dict['test']), axis=0)
    
    start_time = time.time()
    print('[%.1f] Signal to image (%s)'%(time.time()-start_time, mode))
    
    fileNames = generate_topo(ERSP_concat, freqs, num_time, np.arange(ERSP_dict['train'].shape[0]))
    split(fileNames, SLs_concat, len(SLs_dict['test']), random = False)
    
    print('[%.1f] Finish S2I'%(time.time()-start_time))

if __name__ == '__main__':
    
    args = parser.parse_args()
    if args.data_cate == 1:
        ERSP_all, tmp_all, freqs = dataloader.load_data()
    elif args.data_cate == 2:
        with open('./ERSP_from_raw.data', 'rb') as fp:
            dict_ERSP = pickle.load(fp)
        ERSP_all, tmp_all, freqs = dict_ERSP['ERSP'], dict_ERSP['SLs'], dict_ERSP['freq']
        print('Shape of ERSP_all: ', ERSP_all.shape)
    
    # Split data
    indices = {}
    indices['train'], indices['test'] = train_test_split(np.arange(ERSP_all.shape[0]), test_size=0.1, random_state=42)
    
    S2I_main(ERSP_all, tmp_all, freqs, indices, args.mode, args.num_time)
    