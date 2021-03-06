import argparse
import shutil
import time
import faulthandler
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as Data
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import dataloader
import preprocessing
import data_augmentation
import network_dataloader as ndl
from scale_target import TargetScaler
from evaluate_result import plot_error, plot_scatter

import os,sys,inspect
import raw_dataloader
import models as models

parser = argparse.ArgumentParser(description='Deep learning model training')
parser.add_argument('-m', '--model_name', default='vgg16', help='Model for predicting solution latency')
parser.add_argument('-i', '--input_type', default='image', help='Input type of the model')
parser.add_argument('-e', '--num_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('-n', '--file_name', default = '', help='filename after model_name')
parser.add_argument('-d', '--data_cate', default=1, type=int, help='Category of data')
parser.add_argument('-t', '--num_time', default=1, type=int, help='Number of frame for each example')
parser.add_argument('-a', '--augmentation', default=None, type=str, help='Way of data augmentation')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('-l', '--loss_type', default='L2', type=str, help='Loss type')
parser.add_argument('-f', '--image_folder', default='images', type=str, help='Image folder (for image)')

parser.add_argument('--num_fold', default=1, type=int, help='Number of fold for cross validation')
parser.add_argument('--num_split', default=1, type=int, help='Number of split cluster for ensemble methods')
parser.add_argument('--split_mode', default=1, type=int, help='Mode for spliting training data of ensemble methods (signal and power)')
parser.add_argument('--normalize', dest='normalize', action='store_true', help='Electrode-wise exponential moving standardization for signal')
parser.add_argument('--scale_image', dest='scale_image', action='store_true', help='Scale image based on their original values')
parser.add_argument('--scale_data', dest='scale_data', action='store_true', help='Standardize data before feeding into net')
parser.add_argument('--scale_target', dest='scale_target', action='store_true', help='Scale the target quantizationally')
parser.add_argument('--str_sampling', dest='str_sampling', action='store_true', help='Stratified sampling for dataloader')
parser.add_argument('--center', dest='center_flag', action='store_true', help='Center data before feeding into the net')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate the trained model')
parser.add_argument('--lr_rate', default=0.001, type=float, help='Learning rate')
parser.add_argument('--post_scale', dest='post_scale', action='store_true', help='Scale target between 0-1')
parser.add_argument('--index_sub', default=None, type=int, help='Subject number')
parser.add_argument('--add_CE', dest='add_CE', action='store_true', help='Add conditional entropy of signal between channels')

parser.add_argument('--ensemble', default='', type=str, help='Path to models for ensemble learning')
parser.add_argument('--pre_model_name', default='convfc', type=str, help='Pre models for ensemble learning')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none), only for hold-out method')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start_exp', default=0, type=int, help='manual experiment number (useful on restarts)')
parser.add_argument('--start_split', default=0, type=int, help='manual split cluster (useful on restarts)')

def main(index_exp, index_split):
    
    faulthandler.enable()
    torch.cuda.empty_cache()
    
    best_error = 100
    lr_step = [40, 70, 120]
    multiframe = ['convlstm', 'convfc']
    dirName = '%s_data%d_%s_%s_%s'%(args.model_name, args.data_cate, args.augmentation, args.loss_type, args.file_name)
    fileName = '%s_split%d_exp%d'%(dirName, index_split, index_exp)
    
    # Create folder for results of this model
    if not os.path.exists('./results/%s'%(dirName)):
        os.makedirs('./results/%s'%(dirName))
    
    # ------------- Wrap up dataloader -----------------
    if args.input_type == 'signal':
        X, Y_reg, C = raw_dataloader.read_data([1,2,3], list(range(11)), channel_limit=21, rm_baseline=True)
        num_channel = X.shape[1]
        num_feature = X.shape[2]     # Number of time sample
        
        # Remove trials
        X, Y_reg = preprocessing.remove_trials(X, Y_reg, threshold=60)
        
        # Split data for cross validation
        if args.num_fold == 1:
            train_data, test_data, train_target, test_target = train_test_split(X, Y_reg, test_size=0.1, random_state=23)
            # Random state 15: training error becomes lower, testing error becomes higher
        else:
            kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=23)
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                if i == index_exp:
                    train_data, train_target = X[train_index, :], Y_reg[train_index]
                    test_data, test_target = X[test_index, :], Y_reg[test_index]
                    
        # Split data for ensemble methods
        if not args.ensemble:
            if args.num_split > 1:
                data_list, target_list = preprocessing.stratified_split(train_data, train_target, n_split=args.num_split, mode=args.split_mode)
                train_data, train_target = data_list[index_split], target_list[index_split]
                '''
                kf = KFold(n_splits=args.num_split, shuffle=True, random_state=32)
                for i, (other_index, split_index) in enumerate(kf.split(train_data)):
                    if i == index_split:
                        train_data, train_target = train_data[split_index, :], train_target[split_index]
                '''
        # Normalize the data
        if args.normalize:
            train_data, test_data = preprocessing.normalize(train_data, test_data)
        
                    
        # Data augmentation
        if args.augmentation == 'overlapping':
            train_data, train_target = data_augmentation.aug(train_data, train_target, args.augmentation,
                                                             (256, 64, 128))
            test_data, test_target = data_augmentation.aug(test_data, test_target, args.augmentation,
                                                             (256, 64, 128))
        elif args.augmentation == 'add_noise':
            train_data, train_target = data_augmentation.aug(train_data, train_target, args.augmentation,
                                                             (30, 1))
        elif args.augmentation == 'add_noise_minority':
            train_data, train_target = data_augmentation.aug(train_data, train_target, args.augmentation,
                                                             (30, 1))
        elif args.augmentation == 'SMOTER':
            train_data, train_target = data_augmentation.aug(train_data, train_target, args.augmentation)
            
        # scale data
        if args.scale_data:
            train_data, test_data = train_data.reshape((train_data.shape[0],-1)), test_data.reshape((test_data.shape[0],-1))
            train_data, test_data = preprocessing.scale(train_data, test_data)
            train_data = train_data.reshape((train_data.shape[0],num_channel, -1))
            test_data = test_data.reshape((test_data.shape[0],num_channel, -1))
            
        if args.model_name in ['eegnet', 'eegnet_trans_signal']:
            # (sample, channel, time) -> (sample, channel_NN, channel_EEG, time)
            [train_data, test_data] = [X.reshape((X.shape[0], 1, num_channel, num_feature)) \
                                       for X in [train_data, test_data]]
        
        
        (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
        [train_dataset,test_dataset] = map(\
                Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

        if not args.str_sampling:
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size)
        
        model_param = [train_data.shape]
        
    elif args.input_type == 'power':
        if args.data_cate == 1:
            ERSP_all, tmp_all, freqs = dataloader.load_data()
        elif args.data_cate == 2:
            data_file = './raw_data/ERSP_from_raw_%d_channel21.data'%(args.index_sub)
            with open(data_file, 'rb') as fp:
                dict_ERSP = pickle.load(fp)
            ERSP_all, tmp_all = dict_ERSP['ERSP'], dict_ERSP['SLs']
        num_channel = ERSP_all.shape[1]
        num_freq = ERSP_all.shape[2]
            
        # Remove trials
        ERSP_all, tmp_all = preprocessing.remove_trials(ERSP_all, tmp_all, threshold=60)
        
        # Split data for cross validation
        if args.num_fold == 1:
            train_data, test_data, train_target, test_target = train_test_split(ERSP_all, tmp_all[:,2], test_size=0.1, random_state=23)
        else:
            kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=23)
            for i, (train_index, test_index) in enumerate(kf.split(ERSP_all)):
                if i == index_exp:
                    train_data, test_data = ERSP_all[train_index, :], ERSP_all[test_index, :]
                    if args.data_cate == 2:
                        train_target, test_target = tmp_all[train_index], tmp_all[test_index]
                    else:
                        train_target, test_target = tmp_all[train_index, 2], tmp_all[test_index, 2]
                        
                    if args.add_CE:
                        assert args.data_cate == 2
                        with open('./raw_data/CE_sub%d'%(args.index_sub), 'rb') as fp:
                            CE = pickle.load(fp)
                        CE_train, CE_test = CE[train_index,:], CE[test_index,:]
                        # PCA for CE
                        pca = PCA(n_components=10)
                        pca.fit(CE_train)
                        CE_train, CE_test = pca.transform(CE_train), pca.transform(CE_test)
                        
                    
        # Split data for ensemble methods
        if not args.ensemble:
            if args.num_split > 1:
                data_list, target_list = preprocessing.stratified_split(train_data, train_target, n_split=args.num_split, mode=args.split_mode)
                train_data, train_target = data_list[index_split], target_list[index_split]
                '''
                kf = KFold(n_splits=args.num_split, shuffle=True, random_state=32)
                for i, (other_index, split_index) in enumerate(kf.split(np.arange(len(train_data)))):
                    if i == index_split:
                        train_data, train_target = train_data[split_index, :], train_target[split_index]
                '''
                    
        # Concatenate train and test for standardizinsg
        data = np.concatenate((train_data, test_data), axis=0)
        target = np.concatenate((train_target, test_target))
                    
        # Standardize data
        num_train = len(train_data)
        data, target = preprocessing.standardize(data, target, train_indices = np.arange(num_train), threshold=0.0)
        data = data.reshape((data.shape[0], -1))
        
        # Scale target between 0 and 1
        if args.post_scale:
            print('Scale the target between 0-1')
            target = target/60
        
        # Split data
        train_data, test_data = data[:num_train, :], data[num_train:, :]
        train_target, test_target = target[:num_train], target[num_train:]
        
        # Data augmentation
        if args.augmentation == 'SMOTER':
            train_data, train_target = data_augmentation.aug(train_data, train_target, args.augmentation)
        
        # center data
        if args.center_flag:
            train_data, test_data = preprocessing.center(train_data, test_data)
            
        # scale data
        if args.scale_data:
            train_data, test_data = preprocessing.scale(train_data, test_data)
            
        # Add conditional entropy
        if args.add_CE:
            train_data = np.concatenate((train_data, CE_train), axis=1)
            test_data = np.concatenate((test_data, CE_train), axis=1)
            
        if args.model_name == 'eegnet_trans_power':
            # (sample, channel, freq) -> (sample, channel_NN, channel_EEG, freq)
            [train_data, test_data] = [X.reshape((X.shape[0], 1, num_channel, num_freq)) \
                                       for X in [train_data, test_data]]
        
        (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
        [train_dataset,test_dataset] = map(\
                Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

        if not args.str_sampling:
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size)
        
        model_param = [train_data.shape]
        
    elif args.input_type == 'image':
        
        if args.ensemble:
            input_model_name = args.pre_model_name
        else:
            input_model_name = args.model_name
        
        assert (input_model_name in multiframe) == (args.num_time>1)
        
        # Let input size be 224x224 if the model is vgg16
        if input_model_name in ['vgg16', 'resnet50']:
            input_size = 224
        else:
            input_size = 64
            
        # Load Data
        data_transforms = {
                'train': transforms.Compose([
                        ndl.Rescale(input_size, args.num_time),
                        ndl.ToTensor(args.num_time)]), 
                'test': transforms.Compose([
                        ndl.Rescale(input_size, args.num_time),
                        ndl.ToTensor(args.num_time)])
                }

        print("Initializing Datasets and Dataloaders...")

        # Create training and testing datasets
        # image_datasets = {x: ndl.TopoplotLoader(args.image_folder, x, args.num_time, data_transforms[x],
        #                 scale=args.scale_image, index_exp=index_exp, index_split=index_split) for x in ['train', 'test']}
        [train_dataset,test_dataset] = [ndl.TopoplotLoader(args.image_folder, x, args.num_time, data_transforms[x],
                        scale=args.scale_image, index_exp=index_exp, index_split=index_split) for x in ['train', 'test']]

        # Create training and testing dataloaders
        # if not args.str_sampling:
        #     train_loader = Data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4)
        # test_loader = Data.DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=4)
        if not args.str_sampling:
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        model_param = [input_size]
        
    elif args.input_type == 'EEGLearn_img':
        
        # Load data
        with open('./EEGLearn_imgs/data1.data', 'rb') as fp:
            dict_data = pickle.load(fp)
        data, target = dict_data['data'], dict_data['target']
        input_size = data.shape[2]
        
        # Split data for cross validation
        if args.num_fold == 1:
            train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1, random_state=23)
            # Random state 15: training error becomes lower, testing error becomes higher
        else:
            kf = KFold(n_splits=args.num_fold, shuffle=True, random_state=23)
            for i, (train_index, test_index) in enumerate(kf.split(data)):
                if i == index_exp:
                    train_data, train_target = data[train_index, :], target[train_index]
                    test_data, test_target = data[test_index, :], target[test_index]
        
        (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
        [train_dataset,test_dataset] = map(\
                Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

        if not args.str_sampling:
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size)
        
        
    # ------------ Create model ---------------
    if args.input_type in ['image','EEGLearn_img']:
        model_param = [input_size]
    else:
        model_param = [train_data.shape]
    
    if not args.ensemble:
        model = read_model(args.model_name, model_param)
    else:
        pre_models = []
        for i in range(args.num_split):
            pre_model = read_model(args.pre_model_name, model_param)
            pre_model.load_state_dict( torch.load('%s/last_model_exp%d_split%d.pt'%(args.ensemble, index_exp, i)) )
            set_parameter_requires_grad(pre_model, True)
            pre_models.append(pre_model)
            
        model = models.__dict__[args.model_name](pre_models)
        
    print('Use model %s'%(args.model_name))
        
    # Run on GPU
    model = model.to(device=device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'L2':
        criterion = nn.MSELoss().to(device=device)
    elif args.loss_type == 'L1':
        criterion = nn.L1Loss().to(device=device)
    elif args.loss_type == 'L4':
        criterion = L4Loss
    elif args.loss_type == 'MyLoss':
        criterion = MyLoss
    print('Use %s loss'%(args.loss_type))
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_rate,momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    
    # Record loss and accuracy of each epoch
    dict_error = {'train_std': list(range(args.num_epoch)), 'test_std': list(range(args.num_epoch)),
                  'train_mape': list(range(args.num_epoch)), 'test_mape': list(range(args.num_epoch))}
    
    # optionally evaluate the trained model
    if args.evaluate:
        if args.resume:
            if os.path.isfile(args.resume):
                model.load_state_dict(torch.load(args.resume))
        
        _, target, pred, _, _ = validate(test_loader, model, criterion)
        plot_scatter(target, pred, dirName, fileName)
        return 0
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            dict_error['train_std'][:args.start_epoch] = checkpoint['dict_error']['train_std']
            dict_error['test_std'][:args.start_epoch] = checkpoint['dict_error']['test_std']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # ------------- Train model ------------------

    for epoch in range(args.start_epoch, args.num_epoch):
        # Create dataloader if using stratified sampler
        if args.str_sampling:
            sampler = SubsetRandomSampler(get_indices_RSS(train_target, int(0.5*len(train_target))))
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, \
                                           sampler=sampler, num_workers=4)
            
        # Learning rate decay
        if epoch in lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # train for one epoch
        _, dict_error['train_std'][epoch], dict_error['train_mape'][epoch] = \
            train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        _, _, _, std_error, dict_error['test_mape'][epoch] = validate(test_loader, model, criterion)
        dict_error['test_std'][epoch] = std_error

        # remember best standard error and save checkpoint
        is_best = std_error < best_error
        best_error = min(std_error, best_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_error': best_error,
            'optimizer': optimizer.state_dict(),
            'dict_error': dict_error
        }, is_best)
        
        # Save best model
        if is_best:
            torch.save(model.state_dict(), './results/%s/best_model_exp%d_split%d.pt'%(dirName, index_exp, index_split))
        if epoch == args.num_epoch-1:
            torch.save(model.state_dict(), './results/%s/last_model_exp%d_split%d.pt'%(dirName, index_exp, index_split))
    # Plot error curve
    plot_error(dict_error, dirName, fileName)
    
    # Plot scatter plots
    _, target, pred, _, _ = validate(test_loader, model, criterion)
    plot_scatter(target, pred, dirName, fileName)
    dict_error['target'], dict_error['pred'] = target, pred
    
    # Plot histogram
    import matplotlib.pyplot as plt
    plt.hist(target, label = 'True')
    plt.hist(pred, label = 'Pred')
    plt.legend(loc='upper right')
    plt.savefig('./results/hist.png')
    
    # Save error over epochs
    with open('./results/%s/%s.data'%(dirName, fileName), 'wb') as fp:
        pickle.dump(dict_error, fp)
    

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    std_errors = AverageMeter()
    MAPEs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        
        if args.input_type in ['signal', 'power', 'EEGLearn_img']:
            input, target = sample[0], sample[1]
        elif args.input_type in ['image']:
            input, target = sample['image'], sample['label']
            #target = target.view(-1,1)
        
        # measure data loading time
        data_time.update(time.time() - end)

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
        
        # Record error
        if args.post_scale:
            output, target_var = output*60, target_var*60
            
        std_error = StandardError(output, target_var)
        mape = MAPE(output, target_var)
        std_errors.update(std_error.data.item(), input.size(0))
        MAPEs.update(mape.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {4}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, args.num_epoch, i, len(train_loader), curr_lr,
                   batch_time=batch_time, data_time=data_time, loss=std_errors))
            
        del input
        del target
        torch.cuda.empty_cache()

    return losses.avg, std_errors.avg, MAPEs.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    std_errors = AverageMeter()
    MAPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    for i, sample in enumerate(val_loader):
        
        if args.input_type in ['signal', 'power', 'EEGLearn_img']:
            input, target = sample[0], sample[1]
        elif args.input_type in ['image']:
            input, target = sample['image'], sample['label']
            #target = target.view(-1,1)
        
        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        with torch.no_grad():
            output = model(input_var)
            output = output.flatten()
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        if args.post_scale:
            output, target = output*60, target*60
            
        std_error = StandardError(output, target)
        mape = MAPE(output, target)
        losses.update(loss.data.item(), input.size(0))
        std_errors.update(std_error.data.item(), input.size(0))
        MAPEs.update(mape.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=std_errors))
            
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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

def L4Loss(pred, target):
    return torch.sum(torch.pow(target-pred, 4))/target.size()[0]

def MyLoss(pred, target):
    return torch.sum(torch.div(target, pred) * torch.abs(target-pred))/target.size()[0]

def StandardError(pred, target):
    return torch.sqrt(torch.sum(torch.pow(target-pred,2))/target.size()[0])

def MAPE(pred, target):
    '''
    Mean Absolute Percentaget Error

    '''
    return torch.sum( torch.abs( torch.div(target-pred,target) ) ) / target.size()[0]

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def read_model(modelName, model_param):
    
    if args.input_type in ['image', 'EEGLearn_img']:
        input_size = model_param[0]
    else:
        shape_train = model_param[0]
    
    
    if modelName == 'vgg16':
        model = tv_models.vgg16(pretrained=True)
        set_parameter_requires_grad(model, True)
        
        model.classifier[0] = nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features)
        model.classifier[1] = nn.ReLU()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, model.classifier[3].out_features)
        model.classifier[4] = nn.ReLU()
        
        model.classifier[6] = nn.Linear(model.classifier[6].in_features,1)
    elif modelName == 'resnet50':
        model = tv_models.resnet50(pretrained=True)
        set_parameter_requires_grad(model, True)
        model.fc = nn.Linear(2048, 1)
        
    elif modelName == 'convnet':
        model = models.__dict__[modelName](input_size)
    elif modelName == 'convlstm':
        model = models.__dict__[modelName](input_size, 64, 32, args.num_time)
    elif modelName == 'convfc':
        model = models.__dict__[modelName](input_size, 32, args.num_time)
    elif modelName == 'eegnet':
        model = models.__dict__[modelName](nn.ReLU(), (shape_train[2], shape_train[3]), shape_train[3], D=3)
    elif modelName in ['eegnet_trans_power', 'eegnet_trans_signal']:
        model = models.__dict__[modelName](shape_train[2], shape_train[3])
    else:
        model = models.__dict__[modelName](shape_train[1])
        
        
    return model

def get_indices_RSS(targets, num_samples):
    '''
    Get indices for RandomSubsetSampler

    Parameters
    ----------
    targets : np.ndarray
        Solution latency
    num_samples : int
        Number of samples to draw

    Returns
    -------
    indices : np.ndarray
        Chosen indices

    '''
    assert isinstance(targets, np.ndarray)
    assert isinstance(num_samples, int) and num_samples>0
    
    # Calculate weights according to inverse distribution
    hist, bin_edges = np.histogram(targets, bins=10)
    zero_range = np.where(hist==0)[0]
    weights = 1/hist
    
    # Choose samples according to probs
    probs = np.zeros(len(targets))
    for i in range(len(targets)):
        for j in range(len(hist)):
            if j in zero_range:
                continue
            if bin_edges[j]<targets[i]<=bin_edges[j+1]:
                probs[i] = weights[j]
    probs = probs/np.sum(probs)
    
    indices = np.arange(len(targets))
    indices = np.random.choice(indices, size=num_samples, p=probs, replace=True)
    
    return indices

if __name__ == '__main__':
    global device, args
    args = parser.parse_args()
    
    # select gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    for i_exp in range(args.start_exp, args.num_fold):
        # Cross validation
        print('--- Experiment %d ---'%(i_exp))
        if args.ensemble:
            print('*** Ensemble ***')
            main(i_exp, 100)
        else:
            for i_split in range(args.start_split, args.num_split):
                print('*** Split %d ***'%(i_split))
                main(i_exp, i_split)