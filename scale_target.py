#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:02:55 2020

@author: hundredball
"""

import numpy as np

class TargetScaler(object):
    '''
    Scale the target to -1~1 quantizationally
    '''
    def __init__(self, num_step=10):
        '''
        num_step : Number of steps for quantization

        '''
        assert isinstance(num_step, int) and num_step>0
        
        self.num_step = num_step
        # Set boundaries -1, -1+step_size, -1+2*step_size, ..., 1
        step_size = 2/num_step
        self.bound_step = np.array([-1+i*step_size for i in range(num_step+1)])
        self.bound_step[-1] = 1
        print('Boundaries of transformed target: ', self.bound_step)
        
    def fit(self, train_target):
        '''
        Calculate boundary of target

        Parameters
        ----------
        train_target : np.ndarray
            Targets of training data

        Returns
        -------
        None.

        '''
        assert isinstance(train_target, np.ndarray) and train_target.shape[0] == train_target.size
        
        # Find boundaries of the target
        step_size = 1/self.num_step
        self.bound_target = np.quantile(train_target, np.array([i*step_size for i in range(self.num_step+1)]))
        self.bound_target[0], self.bound_target[-1] = 0, 60
        print('Boundaries of original target: ', self.bound_target)
        
    def transform(self, target, mode='normal'):
        '''
        Transform target (0~60) <-> (-1~1)

        Parameters
        ----------
        target : np.ndarray
            Original targets
        mode : str
            normal (0~60) -> (-1~1) | inverse (-1~1) -> (0~60)

        Returns
        -------
        target_trans : np.ndarray
            Transformed targets

        '''
        assert isinstance(target, np.ndarray) and target.shape[0] == target.size
        assert mode == 'normal' or 'inverse'
        
        if mode == 'normal':
            indices_step = self.get_index_step(target, self.bound_target)
        else:
            indices_step = self.get_index_step(target, self.bound_step)
        target_trans = np.zeros(target.shape)
        
        # Transform the target
        for i_sample in range(len(target)):
            index_step = indices_step[i_sample]
            if mode == 'normal':
                pre_range = [self.bound_target[index_step], self.bound_target[index_step+1]]
                post_range = [self.bound_step[index_step], self.bound_step[index_step+1]]
            else:
                post_range = [self.bound_target[index_step], self.bound_target[index_step+1]]
                pre_range = [self.bound_step[index_step], self.bound_step[index_step+1]]
                
            target_trans[i_sample] = self.scale(float(target[i_sample]), pre_range, post_range)
            
        return target_trans
            
    def scale(self, x, pre_range, post_range):
        '''
        Transform x in pre_range into data in post_range

        Parameters
        ----------
        x : float
            Target
        pre_range : list
            a ~ b
        post_range : list
            c ~ d

        Returns
        -------
        result : float
            Transformed target

        '''
        assert isinstance(x, float)
        assert isinstance(pre_range, list) and len(pre_range)==2
        assert isinstance(post_range, list) and len(post_range)==2
        
        a, b = pre_range[0], pre_range[1]
        c, d = post_range[0], post_range[1]
        
        return (x-a)*(d-c)/(b-a) + c
        
        
    def get_index_step(self, target, bound):
        '''
        Get index of step in boundaries for each targets

        Parameters
        ----------
        target : np.ndarray
            Targets (0~60 or -1~1)
        bound : np.ndarray
            bound_target when targets (0~60), bound_step when targets (-1~1)
        Returns
        -------
        indices_step : np.ndarray
            Indices of step in boundaries for each targets

        '''
        assert isinstance(target, np.ndarray) and target.shape[0] == target.size
        assert isinstance(bound, np.ndarray)
        
        indices_step = np.zeros(len(target))
        for i_sample in range(len(target)):
            for i_bound in range(len(bound)-1):
                if bound[i_bound] < target[i_sample] <= bound[i_bound+1]:
                    indices_step[i_sample] = i_bound
                    break
        
        # Convert to int type
        indices_step = np.asarray(indices_step, dtype=int)
    
        return indices_step
        
if __name__ == '__main__':
    scaler = TargetScaler(10)
    target = np.random.rand(20)*60
    scaler.fit(target)
    target_trans = scaler.transform(target, mode='normal')
    target_trans_trans = scaler.transform(target_trans, mode='inverse')
    
    print('Target: ', target)
    print('Transformed target: ', target_trans)
    print('Inverse transformed target: ', target_trans_trans)
    
    print('Traget - inverse transformed target', target-target_trans_trans)
