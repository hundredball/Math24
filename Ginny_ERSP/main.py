import argparse
import shutil
import time
import faulthandler
import signal
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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

import dataloader
import preprocessing
import data_augmentation
import network_dataloader as ndl

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import raw_dataloader
'''
from mynet import mynet
from RCNN import *
from eegnet import 
'''
import models as models

best_std = 100
# select gpus
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description='Math24 project')
parser.add_argument('-m', '--model_name', default='vgg16', help='Model for predicting solution latency')
parser.add_argument('-i', '--input_type', default='image', help='Input type of the model')
parser.add_argument('-e', '--num_epoch', default=100, type=int, help='Number of epochs')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-n', '--file_name', default = '', help='filename after model_name')
parser.add_argument('-d', '--data_cate', default=1, type=int, help='Category of data')
parser.add_argument('-t', '--num_time', default=1, type=int, help='Number of frame for each example')
parser.add_argument('-a', '--augmentation', default=None, type=str, help='Way of data augmentation')
parser.add_argument('-s', '--scale_flag', default=False, type=bool, help='Standardize data before feeding into the net')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')


def main():
    global best_std, device, num_epoch, args
    
    faulthandler.enable()
    
    args = parser.parse_args()
    lr_step = [40, 70, 120]
    multiframe = ['convlstm', 'convfc']
    
    torch.cuda.empty_cache()
    
    # ------------- Wrap up dataloader -----------------
    if args.input_type == 'signal':
        X, Y_class, Y_reg, C = raw_dataloader.read_data([1,2,3], list(range(11)), pred_type='class')
        
        # Split data
        train_data, test_data, train_target, test_target = train_test_split(X, Y_reg, test_size=0.1, random_state=23)
        # Random state 15: training error becomes lower, testing error becomes higher
        
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
            
        if args.model_name == 'eegnet':
            len_time = train_data.shape[2]
            # (sample, channel, time) -> (sample, channel_NN, channel_EEG, time)
            [train_data, test_data] = [X.reshape((X.shape[0], 1, X.shape[1], X.shape[2])) \
                                       for X in [train_data, test_data]]
        
        (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
        [train_dataset,test_dataset] = map(\
                Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size)
        
    elif args.input_type == 'power':
        if args.data_cate == 1:
            ERSP_all, tmp_all, freqs = dataloader.load_data()
        elif args.data_cate == 2:
            with open('./ERSP_from_raw.data', 'rb') as fp:
                dict_ERSP = pickle.load(fp)
            ERSP_all, tmp_all = dict_ERSP['ERSP'], dict_ERSP['SLs']
        
        # Set split indices
        indices = {}
        indices['train'], indices['test'] = train_test_split(np.arange(ERSP_all.shape[0]), test_size=0.1, random_state=40)
        
        # Standardize data
        ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all, train_indices = indices['train'])
        ERSP_all = ERSP_all.reshape((ERSP_all.shape[0], -1))
        
        # Split data
        train_data, test_data = tuple([ ERSP_all[indices[kind],:] for kind in ['train','test'] ])
        train_target, test_target = tuple([ SLs[indices[kind]].reshape((-1,1)) for kind in ['train','test'] ])
        
        # Scale data
        if args.scale_flag:
            train_data, test_data = preprocessing.scale(train_data, test_data)
        
        (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
        [train_dataset,test_dataset] = map(\
                Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size)
        
    elif args.input_type == 'image':
        
        assert (args.model_name in multiframe) == (args.num_time>1)
        
        # Let input size be 224x224 if the model is vgg16
        if args.model_name == 'vgg16':
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
        image_datasets = {x: ndl.TopoplotLoader('images', x, args.num_time, data_transforms[x]) for x in ['train', 'test']}

        # Create training and testing dataloaders
        train_loader = Data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = Data.DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        
    # ------------ Create model ---------------
    
    if args.model_name == 'mynet':
        model = models.__dict__[args.model_name](train_data.shape[1])
    elif args.model_name == 'vgg16':
        model = tv_models.vgg16(pretrained=True)
        set_parameter_requires_grad(model, True)
        
        model.classifier[0] = nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features)
        model.classifier[1] = nn.Sigmoid()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, model.classifier[3].out_features)
        model.classifier[4] = nn.Sigmoid()
        
        model.classifier[6] = nn.Linear(model.classifier[6].in_features,1)
    elif args.model_name == 'convnet':
        model = models.__dict__[args.model_name](input_size)
    elif args.model_name == 'convlstm':
        model = models.__dict__[args.model_name](input_size, 64, 32, args.num_time)
    elif args.model_name == 'convfc':
        model = models.__dict__[args.model_name](input_size, 32, args.num_time)
    elif args.model_name == 'eegnet':
        model = models.__dict__[args.model_name](nn.ReLU(), (train_data.shape[2], train_data.shape[3]), len_time, D=3)
        
    print('Use model %s'%(args.model_name))
        
    # Run on GPU
    model = model.to(device=device)
    '''
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    '''

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().to(device=device)
    #optimizer = torch.optim.SGD(model.parameters(), 0.001,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Record loss and accuracy of each epoch
    dict_std = {'train': list(range(args.num_epoch)), 'test': list(range(args.num_epoch))}
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_std = checkpoint['best_std']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            dict_std = checkpoint['dict_std']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # ------------- Train model ------------------
    
    for epoch in range(args.start_epoch, args.num_epoch):
        
        # Learning rate decay
        if epoch in lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # train for one epoch
        dict_std['train'][epoch] = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        std, _, _ = validate(test_loader, model, criterion)
        dict_std['test'][epoch] = std

        # remember best standard error and save checkpoint
        is_best = std < best_std
        best_std = min(std, best_std)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_std': best_std,
            'optimizer': optimizer.state_dict(),
            'dict_std': dict_std
        }, is_best)
        
        # Save best model
        if is_best:
            torch.save(model.state_dict(), './results/best_%s_%s.pt'%(args.model_name, args.file_name))
    
    # Save error over epochs
    with open('./results/%s_%s.data'%(args.model_name, args.file_name), 'wb') as fp:
        pickle.dump(dict_std, fp)
        
    fileName = '%s_%s'%(args.model_name, args.file_name)
    # Plot error curve
    plot_std(dict_std['train'], dict_std['test'], fileName)
    
    # Plot scatter plots
    _, target, pred = validate(test_loader, model, criterion)
    plot_scatter(target, pred, fileName)
    

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        
        if args.input_type in ['signal','power']:
            input, target = sample[0], sample[1]
        elif args.input_type == 'image':
            input, target = sample['image'], sample['label']
            target = target.view(-1,1)
        
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                   batch_time=batch_time, data_time=data_time, loss=losses))
            
        del input
        del target
        torch.cuda.empty_cache()
    

    return (losses.avg)**0.5

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    for i, sample in enumerate(val_loader):
        
        if args.input_type in ['signal','power']:
            input, target = sample[0], sample[1]
        elif args.input_type == 'image':
            input, target = sample['image'], sample['label']
            target = target.view(-1,1)
        
        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))
            
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

    return (losses.avg)**0.5, true, pred


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


def cal_r2(output, target):
    
    output = output.cpu().numpy()
    target = target.cpu()
    r2 = r2_score(target, output)
    
    return r2

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def plot_std(train, test, fileName):
    '''
    Plot the standard error curve of training and testing data

    Parameters
    ----------
    train : iterator
        Training standard error
    test : itertor
        Testing standard error
    fileName : string, optional
        Name for file and title

    Returns
    -------
    None.

    '''
    assert hasattr(train, '__iter__')
    assert hasattr(test, '__iter__')
    
    epoch = list(range(len(train)))
    plt.plot(epoch, train, 'r-', epoch, test, 'b--')
    plt.xlabel('Epoch')
    plt.ylabel('Standard error')
    plt.title('%s : (%.3f,%.3f)'%(fileName, min(train), min(test)))
    plt.legend(('Train', 'Test'))
    
    plt.savefig('./results/%s_error.png'%(fileName))
    
def plot_scatter(true, pred, fileName):
    '''
    Plot the scatter plots of true target and prediction

    Parameters
    ----------
    true : iterator
        Target
    pred : iterator
        Prediction
    fileName : str
        File name

    Returns
    -------
    None.

    '''
    assert hasattr(true, '__iter__')
    assert hasattr(pred, '__iter__')
    assert isinstance(fileName, str)
    
    sort_indices = np.argsort(true)
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].plot(range(len(true)), true[sort_indices], 'r.', range(len(true)), pred[sort_indices], 'b.')
    axs[0].set_xlabel('Record number')
    axs[0].set_ylabel('Solution latency')
    axs[0].legend(('True', 'Pred'))
    
    max_value = np.max(np.hstack((true, pred)))
    axs[1].scatter(true, pred, marker='.')
    axs[1].plot(range(int(max_value)), range(int(max_value)), 'r')
    axs[1].set_xlabel('True')
    axs[1].set_ylabel('Pred')
    axs[1].set_xlim([0, max_value])
    axs[1].set_ylim([0, max_value])
    
    plt.suptitle(fileName)
    
    plt.savefig('./results/%s_scatter.png'%(fileName))

if __name__ == '__main__':
    main()