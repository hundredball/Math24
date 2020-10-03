#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:46:22 2020

@author: hundredball
"""


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms

from sklearn import linear_model
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import elm
import numpy as np
import argparse
import pickle
import os

import network_dataloader as ndl
import evaluate_result
import preprocessing
import models

parser = argparse.ArgumentParser(description='Deep learning for extracting features, regression model for predicting SL')
parser.add_argument('-m', '--rgr_model', default='LR', help='Regression model')
parser.add_argument('-f', '--image_folder', default='images', help='Folder of images')
parser.add_argument('-n', '--append_name', default='', type=str, help='Appended name after the file name')
parser.add_argument('-d', '--data_cate', default=1, type=int, help='Data category (1: PreData, 2: RawData)')

parser.add_argument('--scale', action='store_true', help='Standardize output of extract layer')
parser.add_argument('--n_components', dest='n_components', default=0.9, type=float, help='Number of component for PCA')
parser.add_argument('--add_CE', action='store_true', help='Add conditional entropy after feature extraction')

def main():
    
    global args, device
    args = parser.parse_args()
    
    dirName = 'extract_regression'
    fileName = '%s_data%d_%s'%(args.rgr_model, args.data_cate, args.append_name)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(dirName)):
        os.makedirs('./results/%s'%(dirName))
        
    # Select GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    print('Extraction model: VGG16')
    print('Regression model: %s'%(args.rgr_model))

    net = tv_models.vgg16(pretrained=True).to(device=device)
    net.classifier[6] = Identity()
    
    # Get dataset
    batchSize = 64
    input_size = 224
    # Load Data
    data_transforms = {
            'train': transforms.Compose([
                    ndl.Rescale(input_size),
                    ndl.ToTensor()]), 
            'test': transforms.Compose([
                    ndl.Rescale(input_size),
                    ndl.ToTensor()])
            }
    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training and testing datasets
    image_datasets = {x: ndl.TopoplotLoader(args.image_folder, x, transform=data_transforms[x]) for x in ['train', 'test']}

    # Create training and testing dataloaders
    dataloaders_dict = {'train': Data.DataLoader(image_datasets['train'], batch_size=batchSize, shuffle=False, num_workers=4),
                        'test':  Data.DataLoader(image_datasets['test'], batch_size=batchSize, shuffle=False, num_workers=4)}
    
    # Extract features by VGG16
    net.eval() # Disable batchnorm, dropout
    X_train, Y_train = extract_layer(dataloaders_dict['train'], net)
    X_test, Y_test = extract_layer(dataloaders_dict['test'], net)
    
    # Standardize data before PCA
    if args.scale:
        X_train, X_test = preprocessing.scale(X_train, X_test)
    
    '''
    # Apply PCA to reduce dimension
    if args.n_components > 1:
        args.n_components = int(args.n_components)
    pca = PCA(n_components=args.n_components, svd_solver='full')
    pca.fit(X_train)
    
    X_train = pca.transform(X_train)
    X_test= pca.transform(X_test)
    print('(X) Number of features after PCA: %d'%(X_train.shape[1]))
    print('(X) Explained variance ratio: %.3f'%(np.sum(pca.explained_variance_ratio_)))
    '''
    
    # Add conditional entropy
    if args.add_CE and args.data_cate==2:
        print('Add conditional entropy as additional features...')
        
        with open('./raw_data/CE_sub100_channel21_exp0_train.data', 'rb') as fp:
            CE_train = pickle.load(fp)
        with open('./raw_data/CE_sub100_channel21_exp0_test.data', 'rb') as fp:
            CE_test = pickle.load(fp)
            
        # Scale CE
        CE_train, CE_test = preprocessing.scale(CE_train, CE_test)
        
        # Apply PCA
        pca = PCA(n_components=30, svd_solver='full')
        pca.fit(CE_train)
        CE_train = pca.transform(CE_train)
        CE_test = pca.transform(CE_test)
        
        print('(CE) Number of features after PCA: %d'%(CE_train.shape[1]))
        print('(CE) Explained variance ratio: %.3f'%(np.sum(pca.explained_variance_ratio_)))
        
        # Concatentate with X
        X_train = np.concatenate((X_train, CE_train), axis=1)
        X_test = np.concatenate((X_test, CE_test), axis=1)
    
    # Regression to predict solution latency
    X_train_Reg = X_train
    X_test_Reg = X_test
    if args.rgr_model == 'LR':
        rgr = linear_model.LinearRegression()
    elif args.rgr_model == 'Ridge':
        rgr = linear_model.Ridge(alpha=1)
    elif args.rgr_model == 'GPR':
        kernel = RBF(10, (1e-2,1e2)) + ConstantKernel(10, (1e-2,1e2))
        rgr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    elif args.rgr_model == 'ELMK':
        # Concatenate data for extreme learning machine
        train_data = np.concatenate((Y_train[:,np.newaxis], X_train_Reg), axis=1)
        test_data = np.concatenate((Y_test[:,np.newaxis], X_test_Reg), axis=1)
        
        elmk = elm.ELMKernel()
        elmk.search_param(train_data, cv="kfold", of="rmse", eval=10)
        pred_train = elmk.train(train_data).predicted_targets
        pred_test = elmk.test(test_data).predicted_targets
    elif args.rgr_model == 'ELMR':
        rgr = models.elm(X_train_Reg.shape[1], 500)
        
    if args.rgr_model not in ['ELMK']:
        rgr.fit(X_train_Reg, Y_train)
        pred_train = rgr.predict(X_train_Reg)
        pred_test = rgr.predict(X_test_Reg)
    
    evaluate_result.plot_scatter(Y_test, pred_test, dirName, fileName)
    
    print('Train std: %.3f'%(mean_squared_error(Y_train, pred_train)**0.5))
    print('Test std: %.3f'%(mean_squared_error(Y_test, pred_test)**0.5))
    
    return

def extract_layer(dataloader, model):
    '''
    Get the outputs of all the data in dataloader

    Parameters
    ----------
    dataloader : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    X : 
        Outputs of all the data
    Y : 
        Corresponding targets

    '''
    
    for i_samples, samples in enumerate(dataloader):
        imgs = samples['image'].to(device=device)
        labels = samples['label'].to(device=device)
    
        if i_samples == 0:
            with torch.no_grad():
                X = model(imgs).cpu().numpy()
            Y = labels.cpu().numpy()
        else:
            with torch.no_grad():
                X = np.concatenate((X, model(imgs).cpu().numpy()), axis=0)
            Y = np.concatenate((Y, labels.cpu().numpy()), axis=0)
    
    return X, Y
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

if __name__ == '__main__':
    main()
    
    