#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:46:52 2020

@author: hundredball
"""
import argparse
import functools
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
from torch.autograd import Variable

from sklearn.model_selection import KFold
from scipy.signal import decimate

import raw_dataloader as rdl
import preprocessing
import custom_loss
from models import tcGAN

parser = argparse.ArgumentParser(description='Target conditioned GAN')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('-e', '--num_epoch', default=5, type=int, help='Number of epoch')

def main(index_exp):
    
    # ----- Wrap up dataloader -----
    # Load data
    X, Y, _ = rdl.read_data([1,2,3], list(range(11)), channel_limit=12, rm_baseline=True)
    
    # Remove trials
    X, Y = preprocessing.remove_trials(X, Y, threshold=60)
    
    # Downsample to 64 Hz
    X = decimate(X, 4, axis=2)
    print('> After downsampling, shape of X: ', X.shape)
    
    # Split data for cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=23)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if i == index_exp:
            train_data, train_target = X[train_index, :], Y[train_index]
            test_data, test_target = X[test_index, :], Y[test_index]
    
    # (sample, channel, time) -> (sample, 1, channel, time)
    [train_data, test_data] = [x.reshape((x.shape[0],1,x.shape[1],x.shape[2])) \
                               for x in [train_data,test_data]]
        
    (train_dataTS, train_targetTS, test_dataTS, test_targetTS) = map(
                torch.from_numpy, (train_data, train_target, test_data, test_target))
    [train_dataset,test_dataset] = map(\
            Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_targetTS.float(),test_targetTS.float()])

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size)

    # ----- Create model -----
    gen = tcGAN.generator().to(device=device)
    dis = tcGAN.discregressor().to(device=device)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        dis = nn.DataParallel(dis)
        
    # ----- Train model -----
    trainer = Trainer(gen, dis, train_loader, test_loader)
    trainer.train(args.num_epoch)
        
class Trainer:

    def __init__(self, G, D, train_loader, test_loader):

        self.G = G
        self.D = D
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_ratio = 2     # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
        self.GP_Weight = 10         # gradient_penalty As per Improved WGAN paper
        self.dict_error = {'train_std':[], 'test_std':[], 'train_MAPE':[], 'test_MAPE':[], 
                           'train_D_loss':[], 'train_G_loss':[]}
        self.R_Weight = 0.001         # weight for regression loss

        self.batch_size = args.batch_size

    def _noise_sample(self, bs):
        '''
        Generate fake data and target as input of generator

        '''

        fake_target = 60*torch.rand((bs,1), requires_grad=True).to(device=device)
        noise = torch.randn((bs,120), requires_grad=True).to(device=device)
        z = torch.cat([noise, fake_target], 1).view(noise.shape[0], -1)

        return z, fake_target

    def train(self, num_epoch):
        
        D_losses = AverageMeter()
        G_losses = AverageMeter()
        R_losses = AverageMeter()

        criterionW = custom_loss.wasserstein_loss    # Wasserstein loss for 
        criterionR = nn.MSELoss().cuda()             # MSE loss for target

        optimG = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.99))
        optimD = torch.optim.Adam(self.D.parameters(), lr=0.001, betas=(0.5, 0.99))
        
        since = time.time()
        
        for epoch in range(num_epoch):
            D_losses.reset()
            G_losses.reset()
            R_losses.reset()
            
            for iters, batch_data in enumerate(self.train_loader, 0):
                optimD.zero_grad()
                
                real_x, real_target = batch_data
                real_x = real_x.to(device=device)
                real_target = real_target.to(device=device)                     # Solution latency
                
                bs = real_x.size(0)
                real_x.requires_grad_()
                real_target.requires_grad_()
                positive_label = torch.ones(bs,1).to(device=device)            # Validity
                negative_label = -1*torch.ones(bs,1).to(device=device)          # Validity

                # ========== Discregressor ==========
                # --- real part ---
                # Calculate loss for real data
                output_real = self.D(real_x)
                output_label_real, output_target_real = output_real[:,0], output_real[:,1]
                lossW_real = criterionW(output_label_real, positive_label)
                lossR_real = criterionR(output_target_real, real_target)
                
                #correct_real = np.sum((probs_real.view(bs,-1).detach().cpu().numpy()>0.5) == label.detach().cpu().numpy())
                
                # --- fake part ---
                # Generate fake input for discriminator
                z, fake_target = self._noise_sample(bs)
                fake_x = self.G(z)
                
                # Calculate loss for real data
                output_fake = self.D(fake_x.detach())
                output_label_fake, output_target_fake = output_fake[:,0], output_fake[:,1]
                lossW_fake = criterionW(output_label_fake, negative_label)
                lossR_fake = criterionR(output_target_fake, fake_target)

                # --- averaged part ---
                # Generate averaged input for discriminator
                weights = torch.rand((bs, 1, 1, 1), requires_grad=True).to(device=device)
                averaged_x = weights*real_x + (1-weights)*fake_x.detach()
                grad_output = torch.ones((bs,1), requires_grad=True).to(device=device)
                
                # Calculate gradient penalty loss
                output_avg = self.D(averaged_x)
                output_label_avg = output_avg[:,0].reshape((bs,1))
                loss_gp = custom_loss.gradient_penalty_loss(grad_output, output_label_avg,
                        averaged_x, self.GP_Weight)
                
                lossW_fake_D = lossW_fake
                '''
                print('lossW_fake: ', lossW_fake)
                print('lossW_real: ', lossW_real)
                print('loss_gp: ', loss_gp)
                
                print('lossR_fake: ', lossR_fake)
                print('lossR_real: ', lossR_real)
                '''
                
                # Update weights in discregressor
                D_loss = lossW_real + lossW_fake + self.R_Weight*lossR_real + loss_gp + self.R_Weight*lossR_fake
                D_loss.backward()
                
                optimD.step()
                D_losses.update(D_loss.data.item(), bs)
                R_losses.update(lossR_real.data.item(), bs)
                
                # ========== Generator ========== (D: training_ratio, G: 1)
                if iters%self.training_ratio == 0:
                    optimG.zero_grad()
                    
                    # Generate fake input
                    z, fake_target = self._noise_sample(bs)
                    fake_x = self.G(z)
                    
                    # Calculate loss
                    output_fake = self.D(fake_x)
                    output_label_fake, output_target_fake = output_fake[:,0], output_fake[:,1]
                    lossW_fake = criterionW(positive_label, output_label_fake)
                    lossR_fake = criterionR(fake_target, output_target_fake)
                    
                    G_loss = lossW_fake + self.R_Weight*lossR_fake
                    G_loss.backward()
                    optimG.step()
                    G_losses.update(G_loss.data.item(), bs)
                    
                    if iters%(self.training_ratio*2) == 0:
                        '''
                        print('lossW_fake: ', lossW_fake)
                        print('lossW_real: ', lossW_real)
                        print('loss_gp: ', loss_gp)
                        
                        print('lossW_fake: ', lossW_fake)
                        print('lossW_fake_D', lossW_fake_D)
                        print('lossW_real: ', lossW_real)
                        '''
                        
                        print('[%.1f s] Epoch [%d/%d][%d/%d]    D_loss: %.3f (%.3f), G_loss: %.3f (%.3f), R_loss: %.3f (%.3f)'\
                              %(time.time()-since, epoch, num_epoch, iters, len(self.train_loader),
                                D_loss, D_losses.avg, G_loss, G_losses.avg, lossR_real, R_losses.avg))
                            
            self.dict_error['train_D_loss'].append(D_losses.avg)
            self.dict_error['train_G_loss'].append(G_losses.avg)
            self.dict_error['train_std'].append(R_losses.avg)

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
    global device, args
    args = parser.parse_args()
    
    # Select GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    for i in range(1):
        print('--- Experiment %d ---'%(i))
        main(i)
