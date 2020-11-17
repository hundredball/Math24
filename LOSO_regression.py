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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

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
from sklearn.ensemble import RandomForestRegressor
import models as models

parser = argparse.ArgumentParser(description='Leave one subject out regression')
parser.add_argument('-i', '--input_type', default='signal', help='Input type (signal, ERSP, bp_ratio)')
parser.add_argument('-m', '--model_name', default='eegnet', help='Model name for deep learning (eegnet, pcafc)')
parser.add_argument('-c', '--num_closest', type=int, default=3, help='Number of closest trials for LST')
parser.add_argument('-d', '--dist_type', type=str, default='target', help='Type of distance for LST (target,correlation)')

parser.add_argument('-e', '--num_epoch', type=int, default=100, help='Number of epoch for deep regression')
parser.add_argument('--lr_rate', default=0.001, type=float, help='Learning rate')
parser.add_argument('--dirName', default='LST_regression', help='Name of folder in results')

parser.add_argument('--no_LST', action='store_true', help='Do not use Least-Square Transform')
parser.add_argument('--SS', action='store_true', help='Source separation')
parser.add_argument('--classical', action='store_true', help='Use classical regression or not')
parser.add_argument('--add_sub_diff', action = 'store_true', help='Concatenate with one-hot subject ID and difficulty level')

def classical_regression(train_data, test_data, train_sub, test_sub, train_diff, test_diff, train_target):
    
    # Flatten the data
    train_data, test_data = train_data.reshape((train_data.shape[0],-1)), test_data.reshape((test_data.shape[0],-1))
    
    # PCA
    '''
    pca = PCA(n_components=30)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    '''
    train_data, test_data = preprocessing.PCA_corr(train_data, train_target, test_data, num_features=10)
    
    if args.add_sub_diff:
        # Standardize data
        train_data, test_data = preprocessing.scale(train_data, test_data, mode='minmax')
        
        # Concatenate subject and difficulty
        train_data = np.concatenate((train_data, train_sub, train_diff), axis=1)
        test_data = np.concatenate((test_data, test_sub, test_diff), axis=1)
    
    # Regression
    #rgr_model = LinearRegression()
    rgr_model = RandomForestRegressor(max_depth=10, random_state=10, n_estimators=100)
    rgr_model.fit(train_data, train_target)
    train_pred = rgr_model.predict(train_data)
    test_pred = rgr_model.predict(test_data)
    
    return train_pred, test_pred
        
def deep_regression(train_data, test_data, train_target, test_target, i_base, i_split):
    
    # --- Wrap up dataloader ---
    batch_size = 64
    num_channel = train_data.shape[1]
    num_feature = train_data.shape[2]
    
    if args.model_name == 'eegnet':
        [train_data, test_data] = [X.reshape((X.shape[0], 1, num_channel, num_feature)) \
                                           for X in [train_data, test_data]]
    elif args.model_name == 'pcafc':
        [train_data, test_data] = [X.reshape((X.shape[0], -1)) \
                                           for X in [train_data, test_data]]
        mean = np.mean(train_data, axis=0)
        train_data, test_data = train_data - mean, test_data-mean
        
    (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
    [train_dataset,test_dataset] = map(\
            Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size)
    
    # --- Create model --- 
    if args.model_name == 'eegnet':
        shape_train = train_data.shape
        model = models.__dict__['eegnet'](nn.ReLU(), (shape_train[2], shape_train[3]), shape_train[3], D=3)
    elif args.model_name == 'pcafc':
        model = models.__dict__['pcafc'](train_data, train_target, num_components=30)
    
    # Run on GPU
    model = model.to(device=device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    lr_step = [40,70]
    criterion = nn.MSELoss().to(device=device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_rate,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    
    # --- Train model ---
    dict_error = {x:np.zeros(args.num_epoch) for x in ['train_std', 'test_std', 'train_mape', 'test_mape']}
    for epoch in range(args.num_epoch):
        
        # Learning rate decay
        if epoch in lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        _, dict_error['train_std'][epoch], dict_error['train_mape'][epoch] = \
            train(train_loader, model, criterion, optimizer, epoch)
            
        _, true, pred, dict_error['test_std'][epoch], dict_error['test_mape'][epoch] = \
            validate(test_loader, model, criterion)
            
    fileName = args.dirName + '_base%d_split%d'%(i_base, i_split)
    evaluate_result.plot_error(dict_error, args.dirName, fileName)
    evaluate_result.plot_scatter(true, pred, args.dirName, fileName)
        
    return dict_error['train_std'][-1], dict_error['test_std'][-1], \
        dict_error['train_mape'][-1], dict_error['test_mape'][-1]
        
def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    std_errors = AverageMeter()
    MAPEs = AverageMeter()

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        
        input, target = sample[0], sample[1]

        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        output = output.flatten()
        loss = criterion(output, target_var)
        losses.update(loss.data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        std_error = StandardError(output, target_var)
        mape = MAPE(output, target_var)
        std_errors.update(std_error.data.item(), input.size(0))
        MAPEs.update(mape.data.item(), input.size(0))

        if i % 5 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {4}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, args.num_epoch, i, len(train_loader), curr_lr, loss=std_errors))
            
        del input
        del target
        torch.cuda.empty_cache()

    return losses.avg, std_errors.avg, MAPEs.avg

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    std_errors = AverageMeter()
    MAPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    for i, sample in enumerate(val_loader):
        
        input, target = sample[0], sample[1]
        
        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        with torch.no_grad():
            output = model(input_var)
            output = output.flatten()
        loss = criterion(output, target_var)
            
        std_error = StandardError(output, target)
        mape = MAPE(output, target)
        losses.update(loss.data.item(), input.size(0))
        std_errors.update(std_error.data.item(), input.size(0))
        MAPEs.update(mape.data.item(), input.size(0))

        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), loss=std_errors))
            
        # Record target and prediction
        target = target.flatten()
        output = output.flatten()
        if i == 0:
            true = target.cpu().numpy()
            pred = output.cpu().numpy()
        else:
            true = np.concatenate((true, target.cpu().numpy()))
            pred = np.concatenate((pred, output.cpu().numpy()))
            
        del input
        del target
    torch.cuda.empty_cache()

    return losses.avg, true, pred, std_errors.avg, MAPEs.avg

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
        
if __name__ == '__main__':
    global device, args
    
    args = parser.parse_args()
    
    if not args.classical:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
    
    # Load data
    if args.input_type == 'signal':
        X,Y,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
        X = np.random.rand(X.shape[0], X.shape[1], X.shape[2])
    elif args.input_type == 'ERSP':
        with open('./raw_data/ERSP_from_raw_100_channel21.data', 'rb') as fp:
            dict_ERSP = pickle.load(fp)
        ERSP, Y, S, D = dict_ERSP['ERSP'], dict_ERSP['SLs'], dict_ERSP['Sub_ID'], dict_ERSP['D']
        X, Y = preprocessing.standardize(ERSP, Y, threshold=0.0)
    elif args.input_type == 'bp_ratio':
        X,Y,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
        low, high = [4], [30]
        X = bandpower.get_bandpower(X, low=low, high=high)
        X = add_features.get_bandpower_ratio(X)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(args.dirName)):
        os.makedirs('./results/%s'%(args.dirName))
    
    # Leave one subject out
    dict_error = {x:[AverageMeter() for i in range(11)] for x in ['train_std', 'test_std', 'train_mape', 'test_mape']}
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
            print('Number of train: %d, Number of test: %d'%(len(train_data), len(test_data)))
            
            if not args.no_LST:
                # LST for training data
                lst_model.fit_(train_data, train_target, train_sub)
                train_data = lst_model.transform_(train_data, train_target, train_sub, args.num_closest, args.dist_type)
            
            if args.SS:     # Source separation
                print('Apply source separation for time signal...')
                SS_model = source_separation.SourceSeparation(train_data.shape[1], 11)
                SS_model.fit(train_data, train_sub)
                train_data = SS_model.transform(train_data, train_sub)
                test_data = SS_model.transform(test_data, test_sub)
            
            # Regression
            if args.classical:
                
                # Onehot encode subject ID and difficulty level
                train_sub = onehot_encode(train_sub, 11)
                test_sub = onehot_encode(test_sub, 11)
                train_diff = onehot_encode(train_diff, 3)
                test_diff = onehot_encode(test_diff, 3)
                
                train_pred, test_pred = classical_regression(train_data, test_data, train_sub, test_sub, train_diff, test_diff, train_target)
            
                # Record error and prediction
                train_std = mean_squared_error(train_target, train_pred)**0.5
                test_std = mean_squared_error(test_target, test_pred)**0.5
                train_mape = mean_absolute_percentage_error(train_target, train_pred)
                test_mape = mean_absolute_percentage_error(test_target, test_pred)
                print('Split %d    Std: (%.1f,%.1f), MAPE: (%.1f,%.1f)'%(i_split, 
                                                                         train_std, test_std, train_mape, test_mape))
                
                # test_pred_all[curr_test_index:curr_test_index+len(test_index)] = test_pred
                # test_target_all[curr_test_index:curr_test_index+len(test_index)] = test_target
                test_pred_all = np.concatenate((test_pred_all, test_pred))
                test_target_all = np.concatenate((test_target_all, test_target))
            else:
                train_std, test_std, train_mape, test_mape = \
                    deep_regression(train_data, test_data, train_target, test_target, i_base, i_split)
                    
            dict_error['train_std'][i_base].update(train_std)
            dict_error['test_std'][i_base].update(test_std)
            dict_error['train_mape'][i_base].update(train_mape)
            dict_error['test_mape'][i_base].update(test_mape)
        
        log_sub = 'Sub%2d\t\tStd: (%.1f,%.1f), MAPE: (%.1f,%.1f)\n'%(i_base, dict_error['train_std'][i_base].avg, dict_error['test_std'][i_base].avg, 
                                                                          dict_error['train_mape'][i_base].avg, dict_error['test_mape'][i_base].avg)
        print(log_sub)
        log_all.append(log_sub)
            
        if args.classical:
            evaluate_result.plot_scatter(train_target, train_pred, dirName=args.dirName, fileName='%s_sub%d_train'%(args.dirName,i_base))
            #evaluate_result.plot_scatter(test_target_all, test_pred_all, dirName=args.dirName, fileName='%s_sub%d'%(args.dirName,i_base))
            
    log_total = 'Total\t\tStd: (%.1f,%.1f), MAPE: (%.1f,%.1f)\n'%(avg_list(dict_error['train_std']), avg_list(dict_error['test_std']),
                                                                  avg_list(dict_error['train_mape']),avg_list(dict_error['test_mape']))
    print(log_total)
    log_all.append(log_total)
    
    # Save dict_error
    with open('./results/%s/%s_error.data'%(args.dirName,args.dirName), 'wb') as fp:
        pickle.dump(dict_error, fp)
        
    # Write log file
    log_file = open('./results/%s/%s_log.txt'%(args.dirName,args.dirName), 'w')
    for L in log_all:
        log_file.writelines(L)
    log_file.close()