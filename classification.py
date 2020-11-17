#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:13:02 2020

@author: hundredball
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

import elm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as Data

import raw_dataloader
import preprocessing
import add_features
import bandpower
import source_separation
import models as models
import evaluate_result

parser = argparse.ArgumentParser(description='Classification')

parser.add_argument('-m', '--clf_model', default='LDA', help='Classfier name (LDA, SVM, ELMK, ELMR, pcafc, pcafc_sd)')
parser.add_argument('-i', '--input_type', default='ERSP', help='Input type (signal, ERSP, bp_ratio)')

parser.add_argument('--dirName', default='classification', help='Name of folder in results')
parser.add_argument('--lr_rate', default=0.001, type=float, help='Learning rate for deep learning')
parser.add_argument('--num_epoch', default=100, type=int, help='Number of epoch for deep learning')

parser.add_argument('--PCA', action='store_true', help='PCA_corr before classification')
parser.add_argument('--SCF', action='store_true', help='Select features correlated with solution latency')
parser.add_argument('--add_sub_diff', action = 'store_true', help='Concatenate label with one-hot subject ID and difficulty level')
parser.add_argument('--SS', action='store_true', help='Source separation (for signal)')

def classify(clf_model, train_data, train_label, test_data):
    
    clf_model.fit(train_data, train_label)
    train_pred = clf_model.predict(train_data)
    test_pred = clf_model.predict(test_data)

    return train_pred, test_pred

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

def deep_classify(train_data, test_data, train_label, test_label, num_signal_features, i_exp):
    
    # --- Wrap up dataloader ---
    batch_size = 64
        
    (train_dataTS, train_labelTS, test_dataTS, test_labelTS) = map(
                torch.from_numpy, (train_data, train_label, test_data, test_label))
    [train_dataset,test_dataset] = map(\
            Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_labelTS.float(),test_labelTS.float()])

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    if args.clf_model == 'pcafc':
        model = models.pcafc(train_data, train_label, num_components=30, mode='class', C=0.1)
    elif args.clf_model == 'pcafc_sd':
        assert args.add_sub_diff
        model = models.pcafc_sd(train_data, train_label, num_signal_features, num_components=30, mode='class', C=0.1)
    
    # Run on GPU
    model = model.to(device=device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    lr_step = [40,70]
    criterion = nn.BCELoss().to(device=device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_rate,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    
    # --- Train model ---
    dict_error = {x:np.zeros(args.num_epoch) for x in ['train_loss', 'test_loss', 'train_acc', 'test_acc']}
    for epoch in range(args.num_epoch):
        
        # Learning rate decay
        if epoch in lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        dict_error['train_loss'][epoch], dict_error['train_acc'][epoch] = \
            train(train_loader, model, criterion, optimizer, epoch)
            
        dict_error['test_loss'][epoch], dict_error['test_acc'][epoch] = \
            validate(test_loader, model, criterion)
            
    fileName = args.dirName + '_exp%d'%(i_exp)
    evaluate_result.plot_error(dict_error, args.dirName, fileName, mode='class')
        
    return dict_error['train_acc'][-1], dict_error['test_acc'][-1]
        
def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc = AverageMeter()

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
        
        output_label = output.data
        output_label[output_label>=0.5] = 1
        output_label[output_label<0.5] = 0
        acc.update(int(torch.sum(target==output_label))/target.size(0), input.size(0))

        if i % 5 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {4}\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                   epoch, args.num_epoch, i, len(train_loader), curr_lr, loss=losses, acc=acc))
            
        del input
        del target
        torch.cuda.empty_cache()

    return losses.avg, acc.avg

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

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
        losses.update(loss.data.item(), input.size(0))
        
        output_label = output.data
        output_label[output_label>=0.5] = 1
        output_label[output_label<0.5] = 0
        acc.update(float(torch.sum(target==output_label))/target.size(0), input.size(0))

        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                   i, len(val_loader), loss=losses, acc=acc))
            
            
        del input
        del target
    torch.cuda.empty_cache()

    return losses.avg, acc.avg

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

if __name__ == '__main__':
    
    global args, device
    
    args = parser.parse_args()
    
    if args.clf_model in ['pcafc', 'pcafc_sd']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(args.dirName)):
        os.makedirs('./results/%s'%(args.dirName))
    
    # Load data
    if args.input_type == 'signal':
        data, SLs,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
        #data = np.random.rand(data.shape[0], data.shape[1], data.shape[2])
    elif args.input_type == 'ERSP':
        with open('./raw_data/ERSP_from_raw_100_channel21.data', 'rb') as fp:
            dict_ERSP = pickle.load(fp)
        data, SLs, S, D = dict_ERSP['ERSP'], dict_ERSP['SLs'], dict_ERSP['Sub_ID'], dict_ERSP['D']
        data, SLs = preprocessing.standardize(data, SLs, threshold=0.0)
    elif args.input_type == 'bp_ratio':
        data,SLs,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
        low, high = [4,7,13], [7,13,30]
        data = bandpower.get_bandpower(data, low=low, high=high)
        data = add_features.get_bandpower_ratio(data)
    elif args.input_type == 'bandpower':
        data,SLs,_,S,D = raw_dataloader.read_data([1,2,3], range(11), channel_limit=21, rm_baseline=True)
        low, high = [4,7,13], [7,13,30]
        data = bandpower.get_bandpower(data, low=low, high=high)
    print('Shape of data: ', data.shape)
    
    # Divide into slow and fast group
    labels_SLs = preprocessing.make_target(SLs, threshold=np.median(SLs))
    
    # KFold cross validation
    log_all = []
    dict_error = {x:AverageMeter() for x in ['train_acc', 'test_acc']}
    kf = KFold(n_splits=10, shuffle=True, random_state=23)
    for i_exp, (train_index, test_index) in enumerate(kf.split(data)):
        
        # Split data
        train_data, test_data = data[train_index,:], data[test_index,:]
        train_label, test_label = labels_SLs[train_index,0], labels_SLs[test_index,0]
        train_SLs, test_SLs = labels_SLs[train_index,1], labels_SLs[test_index,1]
        train_sub, test_sub = S[train_index], S[test_index]
        train_diff, test_diff = D[train_index], D[test_index]
        
        # Source separation
        if args.SS:
            print('Apply source separation for time signal...')
            SS_model = source_separation.SourceSeparation(21, 11)
            SS_model.fit(train_data, train_sub)
            train_data = SS_model.transform(train_data, train_sub)
            test_data = SS_model.transform(test_data, test_sub)
        
        # Dimensional reduction
        train_data, test_data = train_data.reshape((train_data.shape[0],-1)), test_data.reshape((test_data.shape[0],-1))
        if args.PCA:
            train_data, test_data = preprocessing.PCA_corr(train_data, train_SLs, X_test=test_data, num_features=10)
        if args.SCF:
            train_data, test_data, _ = preprocessing.select_correlated_features(train_data, train_SLs, test_data, num_features=10)
        
        num_signal_features = train_data.shape[1]
        if args.add_sub_diff:
            # Standardize data
            #train_data, test_data = preprocessing.scale(train_data, test_data, mode='minmax')
            
            # One-hot encode sub and diff
            train_sub, test_sub = onehot_encode(train_sub, 11), onehot_encode(test_sub, 11)
            train_diff, test_diff = onehot_encode(train_diff, 3), onehot_encode(test_diff, 3)
            
            # Concatenate subject and difficulty
            train_data = np.concatenate((train_data, train_sub, train_diff), axis=1)
            test_data = np.concatenate((test_data, test_sub, test_diff), axis=1)
            
        # Classification
        if args.clf_model == 'LDA':
            clf_model = LDA()
        elif args.clf_model == 'SVM':
            clf_model = SVC(C=1)
        elif args.clf_model == 'ELMK':
            clf_model = elm.ELMKernel()
        elif args.clf_model == 'ELMR':
            clf_model = elm.ELMRandom()
            
            
        if args.clf_model in ['ELMK','ELMR']:
            train_elm_data = np.concatenate((train_label[:,np.newaxis], train_data), axis=1)
            test_elm_data = np.concatenate((test_label[:,np.newaxis], test_data), axis=1)
            clf_model.search_param(train_elm_data, cv="kfold", of="accuracy", eval=10)
            train_acc = clf_model.train(train_elm_data).get_accuracy()
            test_acc = clf_model.test(test_elm_data).get_accuracy()
        elif args.clf_model in ['pcafc', 'pcafc_sd']:
            train_acc, test_acc = deep_classify(train_data, test_data, train_label, test_label, num_signal_features, i_exp)
        else:
            train_pred, test_pred = classify(clf_model, train_data, train_label, test_data)
            # Calculate error
            train_acc = np.sum(train_pred==train_label) / len(train_pred)
            test_acc = np.sum(test_pred==test_label) / len(test_pred)
            
            
        dict_error['train_acc'].update(train_acc)
        dict_error['test_acc'].update(test_acc)
        
        result_exp = 'Exp %d\t\tAccuracy: (%.3f,%.3f)\n'%(i_exp, train_acc, test_acc)
        print(result_exp)
        log_all.append(result_exp)
    
    result_all = 'Total\t\tAccuracy: (%.3f,%.3f)\n'%(dict_error['train_acc'].avg, dict_error['test_acc'].avg)
    print(result_all)
    log_all.append(result_all)
    
     # Write log file
    log_file = open('./results/%s/%s_log.txt'%(args.dirName,args.dirName), 'w')
    for L in log_all:
        log_file.writelines(L)
    log_file.close()
        