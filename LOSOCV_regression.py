#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:55:32 2020

@author: hundredball
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as Data

import raw_dataloader
import preprocessing
import bandpower
import LSTransform
import evaluate_result
import add_features
import source_separation
import data_augmentation
import models as models

parser = argparse.ArgumentParser(description='Leave one subject out or cross validation regression')
parser.add_argument('-v', '--cv_mode', default='LOSO', help='Cross validation mode (LOSO, CV)')
parser.add_argument('-i', '--input_type', default='signal', help='Input type (signal, ERSP, bp_ratio)')
parser.add_argument('-m', '--model_name', default='eegnet', help='Model name for regression (RF, LR, RR, eegnet, pcafc, pcafcown, icarnn, icarnnown)')
parser.add_argument('-c', '--num_closest', type=int, default=3, help='Number of closest trials for LST')
parser.add_argument('-d', '--dist_type', type=str, default='target', help='Type of distance for LST (target,correlation)')
parser.add_argument('-a', '--augmentation', type=str, default=None, help='Data augmentation method (SMOTER)')

parser.add_argument('-e', '--num_epoch', type=int, default=100, help='Number of epoch for deep regression, default=100')
parser.add_argument('--SF', type=int, default=-1, help='Stop freezing feature extraction layer until x epochs, default=-1')
parser.add_argument('--lr_rate', default=0.001, type=float, help='Learning rate, default=0.001')
parser.add_argument('--dirName', default='', help='Directory name of folder in results')
parser.add_argument('--SCF', type=int, default=None, help='Number of selected features correlated with SLs')

parser.add_argument('--PCA', action='store_true', help='PCA for eeg data')
parser.add_argument('--LST', action='store_true', help='Use Least-Square Transform')
parser.add_argument('--SS', action='store_true', help='Source separation')
parser.add_argument('--add_sub_diff', action = 'store_true', help='Concatenate with one-hot subject ID and difficulty level')

def classical_regression(train_data, val_data, test_data, train_target):
      
    # Regression
    if args.model_name == 'LR':
        rgr_model = LinearRegression()
    elif args.model_name == 'RF':
        rgr_model = RandomForestRegressor(max_depth=10, random_state=10, n_estimators=100)
    elif args.model_name == 'RR':
        rgr_model = Ridge(alpha=10.0)
    rgr_model.fit(train_data, train_target)
    train_pred = rgr_model.predict(train_data)
    val_pred = rgr_model.predict(val_data)
    test_pred = rgr_model.predict(test_data)
    
    w = rgr_model.coef_
    inter = rgr_model.intercept_
    print('L2 norm of w: ', w.T.dot(w))
    print('Intercept: ', inter)
    
    return train_pred, val_pred, test_pred
        
def deep_regression(train_data, val_data, test_data, train_target, val_target, test_target, 
                    train_sub, val_sub, test_sub, i_base, i_split):
    
    # --- Wrap up dataloader ---
    batch_size = 64
    
    if args.model_name == 'eegnet':
        num_channel = train_data.shape[1]
        num_feature = train_data.shape[2]
        [train_data, val_data, test_data] = [X.reshape((X.shape[0], 1, num_channel, num_feature)) \
                                           for X in [train_data, val_data, test_data]]
    elif args.model_name in ['pcafc','pcafcown']:
        [train_data, val_data, test_data] = [X.reshape((X.shape[0], -1)) \
                                           for X in [train_data, val_data, test_data]]
        #mean = np.mean(train_data, axis=0)
        #train_data, test_data = train_data - mean, test_data-mean
    elif args.model_name == 'icarnn':
        num_channel = train_data.shape[1]
        num_sample = train_data.shape[2]
        
    (train_dataTS, train_targetTS, train_subTS, val_dataTS, val_targetTS, val_subTS, test_dataTS, test_targetTS, test_subTS) = map(torch.from_numpy, (train_data, train_target, train_sub, val_data, val_target, val_sub, test_data, test_target, test_sub))
    [train_dataset, val_dataset, test_dataset] = map(Data.TensorDataset, 
                                                     [train_dataTS.float(),val_dataTS.float(),test_dataTS.float()],
                                                     [train_targetTS.float(),val_targetTS.float(),test_targetTS.float()],
                                                     [train_subTS, val_subTS, test_subTS])

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size)
    
    # --- Create model --- 
    if args.model_name == 'eegnet':
        shape_train = train_data.shape
        model = models.__dict__['eegnet'](nn.ReLU(), (shape_train[2], shape_train[3]), shape_train[3], D=3)
    elif args.model_name == 'pcafc':
        model = models.__dict__['pcafc'](train_data, train_target, num_components=30, mode='reg', C=1)
    elif args.model_name == 'pcafcown':
        model = models.__dict__['pcafcown'](train_data, train_sub, train_target, num_components=30, mode='reg', C=1)
    elif args.model_name == 'icarnn':
        assert args.input_type == 'signal'
        model = models.__dict__['icarnn'](train_data, train_target, hidden_size=32, output_size=1)
    elif args.model_name == 'icarnnown':
        model = models.__dict__['icarnnown'](train_data, train_sub, train_target, hidden_size=32, output_size=1)
    
    # Run on GPU
    if args.model_name in ['icarnnown','pcafcown']:
        optimizer = []
        for i in range(len(model)):
            model[i] = model[i].to(device=device)
            if torch.cuda.device_count() > 1:
                model[i] = nn.DataParallel(model[i])
            optimizer.append(torch.optim.Adam(model[i].parameters(), lr=args.lr_rate))
    else:
        model = model.to(device=device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_rate,momentum=0.9)
        
    lr_step = []
    active_flag = False    # flag for freezing layer
    criterion = nn.MSELoss().to(device=device)
    
    # --- Train model ---
    dict_error = {x:np.zeros(args.num_epoch) for x in ['train_std', 'val_std', 'test_std', 'train_mape', 'val_mape', 'test_mape']}
    for epoch in range(args.num_epoch):
        
        # Learning rate decay
        if epoch in lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                
        # Stop freezing ICA layer, PCA layer
        if epoch == args.SF and args.model_name in ['icarnn', 'pcafc']:
            model.stopFreezingFE()
            active_flag = True
        elif epoch == args.SF and args.model_name in ['icarnnown', 'pcafcown']:
            for model_sub in model:
                model_sub.stopFreezingFE()
            active_flag = True
        
        _, dict_error['train_std'][epoch], dict_error['train_mape'][epoch] = \
            train(train_loader, model, criterion, optimizer, epoch, active_flag)
            
        _, val_true, val_pred, val_subIDs, dict_error['val_std'][epoch], dict_error['val_mape'][epoch] = \
            validate(val_loader, model, criterion, mode='Val')
        
        _, true, pred, subIDs, dict_error['test_std'][epoch], dict_error['test_mape'][epoch] = \
            validate(test_loader, model, criterion, mode='Test')
            
    if args.cv_mode == 'LOSO':
        fileName = args.dirName + '_base%d_split%d'%(i_base, i_split)
    elif args.cv_mode == 'CV':
        fileName = args.dirName + '_exp%d'%(i_split)
    evaluate_result.plot_error(dict_error, args.dirName, fileName)
    evaluate_result.plot_scatter(val_true, val_pred, args.dirName, fileName+'_val', val_subIDs)
    evaluate_result.plot_scatter(true, pred, args.dirName, fileName+'_test', subIDs)
        
    return dict_error['train_std'][-1], dict_error['val_std'][-1], dict_error['test_std'][-1], \
        dict_error['train_mape'][-1], dict_error['val_mape'][-1], dict_error['test_mape'][-1]
        
def train(train_loader, model, criterion, optimizer, epoch, active_flag):
    losses = AverageMeter()
    std_errors = AverageMeter()
    MAPEs = AverageMeter()

    for i, sample in enumerate(train_loader):
        
        input, target, subIDs = sample[0], sample[1], sample[2]
        input = input.to(device=device)
        target = target.to(device=device)

        if args.model_name in ['icarnnown','pcafcown']:
            
            # Separate input and target for each subject
            for subID in torch.unique(subIDs):
                input_sub = input[subIDs==subID,:]
                target_sub = target[subIDs==subID]
                model_sub, optimizer_sub = model[subID], optimizer[subID]
                
                # switch to train mode
                model_sub.train()
                
                if args.model_name == 'icarnnown':
                    h0 = model_sub.initHidden(len(target_sub), device)
                    output = model_sub(input_sub, h0)
                else:
                    output = model_sub(input_sub)
                output = output.flatten()
                loss = criterion(output, target_sub)
                losses.update(loss.data.item(), input_sub.size(0))
                
                # compute gradient and do SGD step
                optimizer_sub.zero_grad()
                loss.backward()
                optimizer_sub.step()
                
                if args.model_name == 'icarnnown' and active_flag:
                    # Orthonormalize the unmixing matrix
                    model_sub.orthUnmixing()

                std_error = StandardError(output, target_sub)
                mape = MAPE(output, target_sub)
                std_errors.update(std_error.data.item(), input_sub.size(0))
                MAPEs.update(mape.data.item(), input_sub.size(0))
        else:
            # switch to train mode
            model.train()
            
            # compute output
            if args.model_name == 'icarnn':
                h0 = model.initHidden(len(target), device)
                output = model(input, h0)
            else:
                output = model(input)
            output = output.flatten()
            loss = criterion(output, target)
            losses.update(loss.data.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.model_name == 'icarnn' and active_flag:
                # Orthonormalize the unmixing matrix
                model.orthUnmixing()

            std_error = StandardError(output, target)
            mape = MAPE(output, target)
            std_errors.update(std_error.data.item(), input.size(0))
            MAPEs.update(mape.data.item(), input.size(0))

        if i % 5 == 0:
            if args.model_name in ['icarnnown','pcafcown']:
                curr_lr = optimizer[0].param_groups[0]['lr']
            else:
                curr_lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {4}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, args.num_epoch, i, len(train_loader), curr_lr, loss=std_errors))
            
        del input
        del target
        torch.cuda.empty_cache()

    return losses.avg, std_errors.avg, MAPEs.avg

def validate(val_loader, model, criterion, mode='Test'):
    losses = AverageMeter()
    std_errors = AverageMeter()
    MAPEs = AverageMeter()
    
    for i, sample in enumerate(val_loader):
        
        input, target, subIDs = sample[0], sample[1], sample[2]
        input = input.to(device=device)
        target = target.to(device=device)
        
        if args.model_name in ['icarnnown', 'pcafcown']:
            # Separate input and target for each subject
            output = torch.zeros(target.shape, device=device)
            for subID in torch.unique(subIDs):
                indices_sub = torch.nonzero(subIDs==subID).flatten()
                input_sub = input[indices_sub,:]
                target_sub = target[indices_sub]
                model_sub = model[subID]
                
                # switch to evaluate mode
                model_sub.eval()
                
                with torch.no_grad():
                    if args.model_name == 'icarnnown':
                        h0 = model_sub.initHidden(len(target_sub), device)
                        output_sub = model_sub(input_sub, h0)
                    else: 
                        output_sub = model_sub(input_sub)
                    output_sub = output_sub.flatten()
                loss = criterion(output_sub, target_sub)
                losses.update(loss.data.item(), input_sub.size(0))

                std_error = StandardError(output_sub, target_sub)
                mape = MAPE(output_sub, target_sub)
                std_errors.update(std_error.data.item(), input_sub.size(0))
                MAPEs.update(mape.data.item(), input_sub.size(0))
                
                # Put output_sub on the output
                output[indices_sub] = output_sub
        else:
            # switch to evaluate mode
            model.eval()
            
            # compute output
            with torch.no_grad():
                if args.model_name == 'icarnn':
                    h0 = model.initHidden(len(target), device)
                    output = model(input, h0)
                else:
                    output = model(input)
                output = output.flatten()
            loss = criterion(output, target)

            std_error = StandardError(output, target)
            mape = MAPE(output, target)
            losses.update(loss.data.item(), input.size(0))
            std_errors.update(std_error.data.item(), input.size(0))
            MAPEs.update(mape.data.item(), input.size(0))

        if i % 1 == 0:
            print('{0}: [{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   mode, i, len(val_loader), loss=std_errors))
            
        # Record target and prediction
        target = target.flatten()
        output = output.flatten()
        subIDs = subIDs.flatten()
        if i == 0:
            true = target.cpu().numpy()
            pred = output.cpu().numpy()
            subject_IDs = subIDs.cpu().numpy()
        else:
            true = np.concatenate((true, target.cpu().numpy()))
            pred = np.concatenate((pred, output.cpu().numpy()))
            subject_IDs = np.concatenate((subject_IDs, subIDs.cpu().numpy()))
            
        del input
        del target
    torch.cuda.empty_cache()

    return losses.avg, true, pred, subject_IDs, std_errors.avg, MAPEs.avg

def StandardError(pred, target):
    return torch.sqrt(torch.sum(torch.pow(target-pred,2))/target.size()[0])

def MAPE(pred, target):
    '''
    Mean Absolute Percentaget Error

    '''
    return torch.sum( torch.abs( torch.div(target-pred,target) ) ) / target.size()[0]

def mean_absolute_percentage_error(y_true, y_pred): 
    assert np.all(y_true!=0)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def onehot_encode(data, num_class):
    '''
    One hot encoding

    Parameters
    ----------
    data : np.ndarray
        Integers ranging from 0 to (num_class-1)
    num_class : int
        Number of classes

    Returns
    -------
    onehot_data : np.ndarray
        Data after onehot encoding

    '''
    assert isinstance(data, np.ndarray) and data.ndim==1
    assert isinstance(num_class, int) and num_class>0
    
    onehot_data = np.zeros((len(data), num_class))
    
    for i in range(len(data)):
        onehot_data[i, int(data[i])] = 1
        
    return onehot_data

def avg_list(A):
    assert isinstance(A, list)
    
    value_A = [x.avg for x in A]
    avg = sum(value_A)/len(A)
    
    return avg

def LOSO(X, Y, S, D, classical):
    # Leave one subject out
    dict_error = {x:[AverageMeter() for i in range(11)] for x in ['train_std', 'val_std', 'test_std', \
                                                                  'train_mape', 'val_mape', 'test_mape']}
    log_all = []
    start_time = time.time()
    for i_base in range(11):
        print('----- [%.1f] Subject %d -----'%(time.time()-start_time, i_base))
        
        lst_model = LSTransform.LST(11, i_base)
        indices_base, indices_other = np.where(S==i_base)[0], np.where(S!=i_base)[0]
        base_data, base_target, base_sub, base_diff = X[indices_base,:], Y[indices_base], S[indices_base], D[indices_base]
        other_data, other_target, other_sub, other_diff = X[indices_other,:], Y[indices_other], S[indices_other], D[indices_other]
        test_pred_all, test_target_all = np.array([]), np.array([])
        
        # K-fold cross validation (all test data are in one subject)
        kf = KFold(n_splits=5, shuffle=True, random_state=23)
        for i_split, (more_index, few_index) in enumerate(kf.split(base_data)):
            print('--- [%.1f] Split %d ---'%(time.time()-start_time, i_split))
            # Wrap up training and testing data
            train_index, test_index = few_index, more_index
            train_data, test_data = np.concatenate((base_data[train_index,:],other_data), axis=0), base_data[test_index,:]
            train_target, test_target = np.concatenate((base_target[train_index],other_target), axis=0), base_target[test_index]
            train_sub, test_sub = np.concatenate((base_sub[train_index],other_sub), axis=0), base_sub[test_index]
            train_diff, test_diff = np.concatenate((base_diff[train_index],other_diff), axis=0), base_diff[test_index]
            
            # Split training data into training and validation data
            train_data, val_data, train_sub, val_sub, train_diff, val_diff, train_target, val_target = \
                train_test_split(train_data, train_sub, train_diff, train_target, test_size=1/9, random_state=32)
            
            print('Number of (train, val, test): (%d,%d,%d)'%(len(train_data), len(val_data), len(test_data)))
            
            if args.LST:
                # LST for training data
                lst_model.fit_(train_data, train_target, train_sub)
                train_data = lst_model.transform_(train_data, train_target, train_sub, args.num_closest, args.dist_type)
                val_data = lst_model.transform_(val_data, val_target, val_sub, args.num_closest, args.dist_type)
            
            if args.SS:     # Source separation
                print('Apply source separation for time signal...')
                SS_model = source_separation.SourceSeparation(train_data.shape[1], 11)
                SS_model.fit(train_data, train_sub)
                train_data = SS_model.transform(train_data, train_sub)
                val_data = SS_model.transform(val_data, val_sub)
                test_data = SS_model.transform(test_data, test_sub)
            
            # Flatten the data
            if classical:
                [train_data, val_data, test_data] = [x.reshape((x.shape[0],-1)) for x in [train_data, val_data, test_data]]
            
            # Select ERSP correlated with SLs
            if args.SCF:
                train_data, test_data, select_indices = preprocessing.select_correlated_features(train_data, \
                                                                              train_target, test_data, num_features=args.SCF)
                val_data = val_data[:, select_indices==1]
            
            # Data augmentation
            if args.augmentation == 'SMOTER':
                train_data, train_target = data_augmentation.aug(train_data, train_target, method=args.augmentation)

            # PCA
            if args.PCA:
                
                pca = PCA(n_components=200)
                pca.fit(train_data)
                train_data = pca.transform(train_data)
                val_data = pca.transform(val_data)
                test_data = pca.transform(test_data)
                
                #train_data, test_data = preprocessing.PCA_corr(train_data, train_target, test_data, num_features=10)
            
            # Add subject ID and difficulty level as features
            if args.add_sub_diff:
                # Onehot encode subject ID and difficulty level
                train_sub = onehot_encode(train_sub, 11)
                val_sub = onehot_encode(val_sub, 11)
                test_sub = onehot_encode(test_sub, 11)
                train_diff = onehot_encode(train_diff, 3)
                val_diff = onehot_encode(val_diff, 3)
                test_diff = onehot_encode(test_diff, 3)
                
                # Standardize data
                _, test_data = preprocessing.scale(train_data, test_data, mode='minmax')
                train_data, val_data = preprocessing.scale(train_data, val_data, mode='minmax')

                # Concatenate subject and difficulty
                train_data = np.concatenate((train_data, train_sub, train_diff), axis=1)
                val_data = np.concatenate((val_data, val_sub, val_diff), axis=1)
                test_data = np.concatenate((test_data, test_sub, test_diff), axis=1)
            
            # Regression
            if classical:
                
                train_pred, val_pred, test_pred = classical_regression(train_data, val_data, test_data, train_target)
            
                # Record error and prediction
                train_std = mean_squared_error(train_target, train_pred)**0.5
                val_std = mean_squared_error(val_target, val_pred)**0.5
                test_std = mean_squared_error(test_target, test_pred)**0.5
                train_mape = mean_absolute_percentage_error(train_target, train_pred)
                val_mape = mean_absolute_percentage_error(val_target, val_pred)
                
                
                test_mape = mean_absolute_percentage_error(test_target, test_pred)
                print('Split %d    Std: (%.1f,%.1f,%.1f), MAPE: (%.1f,%.1f,%.1f)'%(i_split, 
                                                         train_std, val_std, test_std, train_mape, val_mape, test_mape))
                
                # test_pred_all[curr_test_index:curr_test_index+len(test_index)] = test_pred
                # test_target_all[curr_test_index:curr_test_index+len(test_index)] = test_target
                test_pred_all = np.concatenate((test_pred_all, test_pred))
                test_target_all = np.concatenate((test_target_all, test_target))
            else:
                train_std, val_std, test_std, train_mape, val_mape, test_mape = \
                    deep_regression(train_data, val_data, test_data, train_target, val_target, test_target, train_sub,
                                    val_sub, test_sub, i_base, i_split)
                    
            dict_error['train_std'][i_base].update(train_std, len(train_data))
            dict_error['val_std'][i_base].update(val_std, len(val_data))
            dict_error['test_std'][i_base].update(test_std, len(test_data))
            dict_error['train_mape'][i_base].update(train_mape, len(train_data))
            dict_error['val_mape'][i_base].update(val_mape, len(val_data))
            dict_error['test_mape'][i_base].update(test_mape, len(test_data))
        
        log_sub = 'Sub%2d\t\tStd: (%.1f,%.1f,%.1f), MAPE: (%.1f,%.1f,%.1f)\n'%(i_base, dict_error['train_std'][i_base].avg, dict_error['val_std'][i_base].avg, dict_error['test_std'][i_base].avg, 
                                                                          dict_error['train_mape'][i_base].avg, dict_error['val_mape'][i_base].avg, dict_error['test_mape'][i_base].avg)
        print(log_sub)
        log_all.append(log_sub)
            
        if classical:
            evaluate_result.plot_scatter(train_target, train_pred, dirName=args.dirName, fileName='%s_sub%d_train'%(args.dirName,i_base))
            #evaluate_result.plot_scatter(test_target_all, test_pred_all, dirName=args.dirName, fileName='%s_sub%d'%(args.dirName,i_base))
            
    log_total = 'Total\t\tStd: (%.1f,%.1f,%.1f), MAPE: (%.1f,%.1f,%.1f)\n'%(avg_list(dict_error['train_std']), avg_list(dict_error['val_std']), avg_list(dict_error['test_std']),
                                                                  avg_list(dict_error['train_mape']), avg_list(dict_error['val_mape']), avg_list(dict_error['test_mape']))
    print(log_total)
    log_all.append(log_total)
    
    return log_all, dict_error

def CV(X, Y, S, D, classical):
    n_splits = 10
    
    # Cross validation (mixed subjects), 10 splits
    dict_error = {x:[AverageMeter() for i in range(n_splits)] for x in ['train_std', 'val_std', 'test_std', \
                                                                  'train_mape', 'val_mape', 'test_mape']}
    log_all = []
    start_time = time.time()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=23)
    for i_exp, (train_index, test_index) in enumerate(kf.split(X)):
        print('----- [%.1f] Exp %d -----'%(time.time()-start_time, i_exp))
        
        # Wrap up training and testing data
        train_data, test_data = X[train_index,:], X[test_index,:]
        train_target, test_target = Y[train_index], Y[test_index]
        train_sub, test_sub = S[train_index], S[test_index]
        train_diff, test_diff = D[train_index], D[test_index]
        
        # Split training data into training and validation data
        train_data, val_data, train_sub, val_sub, train_diff, val_diff, train_target, val_target = \
            train_test_split(train_data, train_sub, train_diff, train_target, test_size=1/9, random_state=32)
        print('Number of (train, val, test): (%d,%d,%d)'%(len(train_data), len(val_data), len(test_data)))
            
            
        # Flatten the data
        if classical:
            [train_data, val_data, test_data] = [x.reshape((x.shape[0],-1)) for x in [train_data, val_data, test_data]]

        # Select ERSP correlated with SLs
        if args.SCF:
            train_data, test_data, select_indices = preprocessing.select_correlated_features(train_data, \
                                                                          train_target, test_data, num_features=args.SCF)
            val_data = val_data[:, select_indices==1]

        # Data augmentation
        if args.augmentation == 'SMOTER':
            train_data, train_target = data_augmentation.aug(train_data, train_target, method=args.augmentation)

        # PCA
        if args.PCA:

            pca = PCA(n_components=200)
            pca.fit(train_data)
            train_data = pca.transform(train_data)
            val_data = pca.transform(val_data)
            test_data = pca.transform(test_data)

            #train_data, test_data = preprocessing.PCA_corr(train_data, train_target, test_data, num_features=10)

        # Add subject ID and difficulty level as features
        if args.add_sub_diff:
            # Onehot encode subject ID and difficulty level
            train_sub = onehot_encode(train_sub, 11)
            val_sub = onehot_encode(val_sub, 11)
            test_sub = onehot_encode(test_sub, 11)
            train_diff = onehot_encode(train_diff, 3)
            val_diff = onehot_encode(val_diff, 3)
            test_diff = onehot_encode(test_diff, 3)

            # Standardize data
            _, test_data = preprocessing.scale(train_data, test_data, mode='minmax')
            train_data, val_data = preprocessing.scale(train_data, val_data, mode='minmax')

            # Concatenate subject and difficulty
            train_data = np.concatenate((train_data, train_sub, train_diff), axis=1)
            val_data = np.concatenate((val_data, val_sub, val_diff), axis=1)
            test_data = np.concatenate((test_data, test_sub, test_diff), axis=1)

        # Regression
        if classical:

            train_pred, val_pred, test_pred = classical_regression(train_data, val_data, test_data, train_target)

            # Record error and prediction
            train_std = mean_squared_error(train_target, train_pred)**0.5
            val_std = mean_squared_error(val_target, val_pred)**0.5
            test_std = mean_squared_error(test_target, test_pred)**0.5
            train_mape = mean_absolute_percentage_error(train_target, train_pred)
            val_mape = mean_absolute_percentage_error(val_target, val_pred)


            test_mape = mean_absolute_percentage_error(test_target, test_pred)
            print('Split %d    Std: (%.1f,%.1f,%.1f), MAPE: (%.1f,%.1f,%.1f)'%(i_split, 
                                                     train_std, val_std, test_std, train_mape, val_mape, test_mape))

            # test_pred_all[curr_test_index:curr_test_index+len(test_index)] = test_pred
            # test_target_all[curr_test_index:curr_test_index+len(test_index)] = test_target
            test_pred_all = np.concatenate((test_pred_all, test_pred))
            test_target_all = np.concatenate((test_target_all, test_target))
        else:
            train_std, val_std, test_std, train_mape, val_mape, test_mape = \
                deep_regression(train_data, val_data, test_data, train_target, val_target, test_target, train_sub, 
                                val_sub, test_sub, -1, i_exp)

        dict_error['train_std'][i_exp].update(train_std, len(train_data))
        dict_error['val_std'][i_exp].update(val_std, len(val_data))
        dict_error['test_std'][i_exp].update(test_std, len(test_data))
        dict_error['train_mape'][i_exp].update(train_mape, len(train_data))
        dict_error['val_mape'][i_exp].update(val_mape, len(val_data))
        dict_error['test_mape'][i_exp].update(test_mape, len(test_data))
        
        log_sub = 'Exp%d\t\tStd: (%.1f,%.1f,%.1f), MAPE: (%.1f,%.1f,%.1f)\n'%(i_exp, dict_error['train_std'][i_exp].avg, dict_error['val_std'][i_exp].avg, dict_error['test_std'][i_exp].avg, 
                                                                          dict_error['train_mape'][i_exp].avg, dict_error['val_mape'][i_exp].avg, dict_error['test_mape'][i_exp].avg)
        print(log_sub)
        log_all.append(log_sub)
            
        if classical:
            evaluate_result.plot_scatter(train_target, train_pred, dirName=args.dirName, fileName='%s_sub%d_train'%(args.dirName,i_exp))
            
    log_total = 'Total\t\tStd: (%.1f,%.1f,%.1f), MAPE: (%.1f,%.1f,%.1f)\n'%(avg_list(dict_error['train_std']), avg_list(dict_error['val_std']), avg_list(dict_error['test_std']),
                                                                  avg_list(dict_error['train_mape']), avg_list(dict_error['val_mape']), avg_list(dict_error['test_mape']))
    print(log_total)
    log_all.append(log_total)
    
    return log_all, dict_error
        
if __name__ == '__main__':
    global device, args
    
    args = parser.parse_args()
    if args.model_name in ['RR','LR','RF']:
        classical = True
    else:
        classical = False
    print('Use model %s\n'%(args.model_name))
    
    if not classical:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
    
    # Load data
    if args.input_type == 'signal':
        X,Y,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
    elif args.input_type == 'ERSP':
        with open('./raw_data/ERSP_from_raw_100_channel21.data', 'rb') as fp:
            dict_ERSP = pickle.load(fp)
        ERSP, Y, S, D = dict_ERSP['ERSP'], dict_ERSP['SLs'], dict_ERSP['Sub_ID'], dict_ERSP['D']
        X, Y = preprocessing.standardize(ERSP, Y, threshold=0.0)
    elif args.input_type == 'bp_ratio':
        X,Y,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
        low, high = [4,7,13], [7,13,30]
        X = bandpower.get_bandpower(X, low=low, high=high)
        X = add_features.get_bandpower_ratio(X)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(args.dirName)):
        os.makedirs('./results/%s'%(args.dirName))
        
    # LOSO or CV
    if args.cv_mode == 'LOSO':
        log_all, dict_error = LOSO(X, Y, S, D, classical)
    elif args.cv_mode == 'CV':
        log_all, dict_error = CV(X, Y, S, D, classical)
    
    # Save dict_error
    with open('./results/%s/%s_error.data'%(args.dirName,args.dirName), 'wb') as fp:
        pickle.dump(dict_error, fp)
        
    # Write log file
    log_file = open('./results/%s/%s_log.txt'%(args.dirName,args.dirName), 'w')
    for L in log_all:
        log_file.writelines(L)
    log_file.close()