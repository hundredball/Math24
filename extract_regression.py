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
from scale_target import TargetScaler

parser = argparse.ArgumentParser(description='Deep learning for extracting features, regression model for predicting SL')
parser.add_argument('-m', '--rgr_model', default='LR', help='Regression model')
parser.add_argument('-e', '--ext_model', default='vgg16', help='Feature extraction model')
parser.add_argument('-f', '--image_folder', default='images', help='Folder of images')
parser.add_argument('-n', '--append_name', default='', type=str, help='Appended name after the file name')
parser.add_argument('-d', '--data_cate', default=1, type=int, help='Data category (1: PreData, 2: RawData)')
parser.add_argument('-i', '--input_type', default='img', type=str, help='img: S2I, EEG_img: EEGLearn_S2I')

parser.add_argument('--scale', default=None, type=str, help='standard, minmax')
parser.add_argument('--n_components', dest='n_components', default=0.9, type=float, help='Number of component for PCA')
parser.add_argument('--add_CE', action='store_true', help='Add conditional entropy after feature extraction')
parser.add_argument('--scale_target', default=0, type=int, help='0: no, 1: normal, 2: quantization')
parser.add_argument('--num_fold', type=int, default=1, help='Number of experiments (for cross validation)')
parser.add_argument('--subject_ID', default=100, type=int, help='Subject ID, 100 for all subjects')

def main(index_exp=0):
    
    
    dirName = '%s_%s_data%d_%s'%(args.ext_model, args.rgr_model, args.data_cate, args.append_name)
    fileName = '%s_exp%d'%(dirName, index_exp)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(dirName)):
        os.makedirs('./results/%s'%(dirName))
    
    print('Extraction model: %s'%(args.ext_model))
    print('Regression model: %s'%(args.rgr_model))
    
    if args.ext_model == 'vgg16':
        net = tv_models.vgg16(pretrained=True).to(device=device)
        set_parameter_requires_grad(net, True)
        net.classifier[6] = Identity()
    elif args.ext_model == 'resnet50':
        net = tv_models.resnet50(pretrained=True).to(device=device)
        set_parameter_requires_grad(net, True)
        net.fc = Identity()
    
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
    image_datasets = {x: ndl.TopoplotLoader(args.image_folder, x, transform=data_transforms[x], index_exp=index_exp) for x in ['train', 'test']}

    # Create training and testing dataloaders
    dataloaders_dict = {'train': Data.DataLoader(image_datasets['train'], batch_size=batchSize, shuffle=False, num_workers=4),
                        'test':  Data.DataLoader(image_datasets['test'], batch_size=batchSize, shuffle=False, num_workers=4)}
    
    # Extract features by VGG16
    net.eval() # Disable batchnorm, dropout
    X_train, Y_train = extract_layer(dataloaders_dict['train'], net)
    X_test, Y_test = extract_layer(dataloaders_dict['test'], net)
    
    # Standardize data before PCA
    if args.scale:
        X_train, X_test = preprocessing.scale(X_train, X_test, mode=args.scale)
    
    # Apply PCA to reduce dimension
    if args.n_components > 1:
        args.n_components = int(args.n_components)
    pca = PCA(n_components=args.n_components, svd_solver='full')
    pca.fit(X_train)
    
    X_train = pca.transform(X_train)
    X_test= pca.transform(X_test)
    print('(X) Number of features after PCA: %d'%(X_train.shape[1]))
    print('(X) Explained variance ratio: %.3f'%(np.sum(pca.explained_variance_ratio_)))
    
    
    # Add conditional entropy
    if args.add_CE and args.data_cate==2:
        print('Add conditional entropy as additional features...')
        
        with open('./raw_data/CE_sub%d_channel21_exp%d_train.data'%(args.subject_ID, index_exp), 'rb') as fp:
            CE_train = pickle.load(fp)
        with open('./raw_data/CE_sub%d_channel21_exp%d_test.data'%(args.subject_ID, index_exp), 'rb') as fp:
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
        rgr = elm.ELMKernel()
    elif args.rgr_model == 'ELMR':
        params = ["sigmoid", 1, 500, False]
        rgr= elm.ELMRandom(params)
        
    if args.rgr_model not in ['ELMK', 'ELMR']:
        rgr.fit(X_train_Reg, Y_train)
        pred_train = rgr.predict(X_train_Reg)
        pred_test = rgr.predict(X_test_Reg)
    else:
        # Scale target into -1~1
        if args.scale_target == 2:
                    
            scaler = TargetScaler(num_step=10)
            scaler.fit(Y_train)
            Y_train, Y_test = scaler.transform(Y_train), scaler.transform(Y_test)
        elif args.scale_target == 1:
            
            Y_train, Y_test = (Y_train-30)/30, (Y_test-30)/30
            
        
        # Concatenate data for extreme learning machine
        train_data = np.concatenate((Y_train[:,np.newaxis], X_train), axis=1)
        test_data = np.concatenate((Y_test[:,np.newaxis], X_test), axis=1)
        
        rgr.search_param(train_data, cv="kfold", of="rmse", eval=10)
        
        pred_train = rgr.train(train_data).predicted_targets
        pred_test = rgr.test(test_data).predicted_targets
        
        # Scale target back to 0~60
        if args.scale_target == 2:
            
            [Y_train, Y_test, pred_train, pred_test] = [scaler.transform(x, mode='inverse') for x in \
                                           [Y_train, Y_test, pred_train, pred_test]]
        elif args.scale_target == 1:
            
            [Y_train, Y_test, pred_train, pred_test] = [x*30+30 for x in \
                                           [Y_train, Y_test, pred_train, pred_test]]
    
    evaluate_result.plot_scatter(Y_test, pred_test, dirName, fileName)
    
    print('Train std: %.3f'%(mean_squared_error(Y_train, pred_train)**0.5))
    print('Test std: %.3f'%(mean_squared_error(Y_test, pred_test)**0.5))
    
    # Save targets and predictions
    dict_target = {}
    dict_target['target'], dict_target['pred'] = Y_test, pred_test
    with open('./results/%s/%s.data'%(dirName, fileName), 'wb') as fp:
        pickle.dump(dict_target, fp)
    
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
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    global args, device
    args = parser.parse_args()
    
    # Select GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    for i_exp in range(args.num_fold):
        print('------ Experiment %d ------'%(i_exp))
        main(i_exp)
    
    