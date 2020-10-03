#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:17:07 2020

@author: hundredball
"""


import torch 

import models as models

if __name__ == '__main__':
    num_exp = 10
    num_split = 4
    
    for i_exp in range(num_exp):
        print('--- Exp %d ---'%(i_exp))
        
        # Create model
        pre_models = []
        
        for i_split in range(num_split):
            pre_model = models.__dict__['simplefc'](114*12)
            pre_models.append(pre_model)
            
        model = models.__dict__['combinedfc'](pre_models)
        
        # Load model
        model.load_state_dict( torch.load('./results/combinedfc_data1_None_L2_strided_split_simplefc/last_model_exp%d_split100.pt'%(i_exp)) )
        
        print('Hidden weight: ', model.hidden[0].weight)
        print('fc weight: ', model.fc[0].weight)