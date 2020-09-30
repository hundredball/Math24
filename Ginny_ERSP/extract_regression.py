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
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn import linear_model
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import numpy as np
import argparse
import pickle
import os

import network_dataloader as ndl
import evaluate_result

parser = argparse.ArgumentParser(description='Deep learning for extracting features, regression model for predicting SL')
parser.add_argument('-m', '--rgr_model', default='LR', help='Regression model')
parser.add_argument('-f', '--image_folder', default='images', help='Folder of images')

def main():
    
    dirName = 'extract_regression'
    fileName = '%s'%(args.rgr_model)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(dirName)):
        os.makedirs('./results/%s'%(dirName))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = models.vgg16(pretrained=True).to(device=device)
    net.classifier[6] = Identity()
    
    # Get dataset
    batchSize = 46
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
    dataloaders_dict = {'train': Data.DataLoader(image_datasets['train'], batch_size=batchSize, shuffle=True, num_workers=4),
                        'test':  Data.DataLoader(image_datasets['test'], batch_size=batchSize, shuffle=False, num_workers=4)}
    
    # Extract features by VGG16
    net.eval() # Disable batchnorm, dropout

    # Training data
    for i_samples, samples in enumerate(dataloaders_dict['train']):
        imgs = samples['image'].to(device=device)
        labels = samples['label'].to(device=device)
    
        if i_samples == 0:
            with torch.no_grad():
                X_train = net(imgs).cpu().numpy()
            Y_train = labels.cpu().numpy()
        else:
            with torch.no_grad():
                X_train = np.concatenate((X_train, net(imgs).cpu().numpy()), axis=0)
            Y_train = np.concatenate((Y_train, labels.cpu().numpy()), axis=0)
    
    print('X_train shape : ', X_train.shape)
    print('Y_train shape : ', Y_train.shape)
    
    # Testing data
    for samples in dataloaders_dict['test']:
        imgs = samples['image'].to(device=device)
        labels = samples['label'].to(device=device)
        
        with torch.no_grad():
            X_test = net(imgs).cpu().numpy()
        Y_test = labels.cpu().numpy()
        
    print('X_test shape : ', X_test.shape)
    print('Y_test shape : ', Y_test.shape)
    
    # Apply PCA to reduce dimension
    pca = PCA(n_components=0.9, svd_solver='full')
    pca.fit(X_train)
    
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test)
    
    print('X_train_PCA shape : ', X_train_PCA.shape)
    print('X_test_PCA shape : ', X_test_PCA.shape)
    
    # Regression to predict solution latency
    X_train_Reg = X_train_PCA
    X_test_Reg = X_test_PCA
    if args.rgr_model == 'LR':
        rgr = linear_model.LinearRegression()
    elif args.rgr_model == 'Ridge':
        rgr = linear_model.Ridge(alpha=1)
    elif args.rgr_model == 'GPR':
        kernel = RBF(10, (1e-2,1e2)) + ConstantKernel(10, (1e-2,1e2))
        rgr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    rgr.fit(X_train_Reg, Y_train)
    
    pred_train = rgr.predict(X_train_Reg)
    pred_test = rgr.predict(X_test_Reg)
    
    evaluate_result.plot_scatter(Y_test, pred_test, dirName, fileName)
    
    print('Train std: %.3f'%(mean_squared_error(Y_train, pred_train)**0.5))
    print('Test std: %.3f'%(mean_squared_error(Y_test, pred_test)**0.5))
    
    return


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    main()
    
    