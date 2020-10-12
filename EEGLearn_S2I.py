#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:32:38 2020

@author: hundredball
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
np.random.seed(1234)

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

import dataloader
import preprocessing

def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
        
    #print('Before augment: ', feat_array_temp[0].shape)
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    n_samples = features.shape[0]
    #print('After augment: ', feat_array_temp[0].shape)

    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
        
    result = np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]
    
    return result

def augment_EEG(data, stdMult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.
    :param data: EEG feature data as a matrix (n_samples x n_features)
    :param stdMult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(scale=stdMult*np.std(feat), size=feat.size)
    return augData

if __name__ == '__main__':
    
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all, threshold=0.0)
    num_channels = ERSP_all.shape[1]
    
    ERSP_all, SLs = preprocessing.remove_trials(ERSP_all, SLs, 60.0)
    num_example = ERSP_all.shape[0]
    
    # Concatenate theta, alpha, beta
    low, high = [4,7,13], [7,13,30]
    for i in range(len(low)):
        bp_i = preprocessing.bandpower(ERSP_all, freqs, [low[i]], [high[i]]).reshape((num_example,-1))
        if i == 0:
            bp_all = bp_i
        else:
            bp_all = np.concatenate((bp_all, bp_i), axis=1)
    
    print(bp_all.shape)
    
    # Read channel information
    channel_info = pd.read_csv('./Channel_coordinate/Channel_location_angle_%d.csv'%(num_channels))
    channel_info = channel_info.to_numpy()
    
    # Change coordinate from 0 toward naison to 0 toward right ear
    channel_info[:,2] = 90-channel_info[:,2]
    
    # Calculate channel locations on the plot
    radius = 1.0                # Radius after scaling
    scale_radius = radius/0.5   # Radius from channel_info is 0.5
    scale_arc = scale_radius*channel_info[:,1]
    plot_loc = np.zeros((num_channels, 2)) # first for x, second for y
    plot_loc[:,0] = scale_arc*np.cos(np.array(channel_info[:,2]*np.pi/180, dtype = np.float))
    plot_loc[:,1] = scale_arc*np.sin(np.array(channel_info[:,2]*np.pi/180, dtype = np.float))
    
    imgs = gen_images(plot_loc, bp_all, 224)
    
    # Save by pickle
    savePath='./EEGLearn_imgs/data1.data'
    result = {'data':imgs, 'target':SLs}
    if savePath:
        with open(savePath, 'wb') as fp:
            pickle.dump(result, fp)
