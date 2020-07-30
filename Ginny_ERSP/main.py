import argparse
import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as Data

import dataloader
import preprocessing
import sampling
from mynet import mynet

best_r2 = 0
# select gpus
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_epoch = 100
def main():
    global best_r2, device, num_epoch
    
    # wrap up training and testing data
    ERSP_all, tmp_all, freqs = dataloader.load_data()
    # ERSP_all, SLs = preprocessing.remove_trials(ERSP_all, tmp_all, 25)  # Remove trials
    ERSP_all, SLs = preprocessing.standardize(ERSP_all, tmp_all)
    ERSP_all = ERSP_all.reshape((ERSP_all.shape[0], -1))
    
    train_data, test_data, train_target, test_target = train_test_split(ERSP_all, SLs.reshape((-1,1)), test_size=0.1)
    
    '''
    # Undersampling
    train_data, train_target = sampling.undersampling(train_data, train_target)
    '''
    
    # SMOTER
    train_data, train_target = sampling.SMOTER(train_data, train_target)
    
    
    (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
            torch.from_numpy, (train_data, train_target, test_data, test_target))
    '''
    [train_dataTS, train_targetTS, test_dataTS, test_targetTS] = [x.to(device=device) for x in [train_dataTS, train_targetTS, test_dataTS, test_targetTS]]
    '''
    [train_dataset,test_dataset] = map(\
            Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])
    
    batchSize = 32
    train_loader = Data.DataLoader(train_dataset, batch_size=batchSize)
    test_loader = Data.DataLoader(test_dataset, batch_size=batchSize)

    # create model
    model = mynet(ERSP_all.shape[1]).to(device=device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9)
    
    # Record loss and accuracy of each epoch
    
    train_std = list(range(num_epoch))
    test_std = list(range(num_epoch))
    train_r2 = list(range(num_epoch))
    test_r2 = list(range(num_epoch))
    
    for epoch in range(num_epoch):
        
        # train for one epoch
        train_r2[epoch], train_std[epoch] = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        r2, test_std[epoch] = validate(test_loader, model, criterion)
        test_r2[epoch] = r2

        # remember best r2 and save checkpoint
        is_best = r2 > best_r2
        best_r2 = max(r2, best_r2)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_r2': best_r2,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        
        # Save best model
        if is_best:
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Save error over epochs
    dict_error = {'train_std': train_std, 'train_r2': train_r2, 'test_std': test_std, 'test_r2': test_r2}
    with open('error.data', 'wb') as fp:
        pickle.dump(dict_error, fp)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        r2 = cal_r2(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        top1.update(r2, input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'r2@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, num_epoch, i, len(train_loader), curr_lr,
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

    print(' * Training r2@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, (losses.avg)**0.5

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device=device)
        target = target.to(device=device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        r2 = cal_r2(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        top1.update(r2, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'r2@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Testing r2@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, (losses.avg)**0.5


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

def plot_over_epoch(train, test, titleName=None, fileName=None):
    
    epochs = list(range(len(train)))
    plt.plot(epochs, train, 'r-', epochs, test, 'b--')
    plt.legend(('Training', 'Testing'))
    plt.title(titleName)
    plt.xlabel('Epoch')
    plt.ylabel(titleName.split(' ')[0])
    plt.savefig('./images/%s'%(fileName))

if __name__ == '__main__':
    main()

    # (a) Final testing error: 0.8727
    # (b) Final testing error: 0.6414