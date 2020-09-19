import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils import data
import torch
import pickle

def getData(root, mode, index_exp, index_split):
    if mode == 'train':
        img = pd.read_csv('./%s/exp%d/train%d_img.csv'%(root, index_exp, index_split))
        label = pd.read_csv('./%s/exp%d/train%d_label.csv'%(root, index_exp, index_split))
        return img['fileName'].values, label['solution_time'].values
    else:
        img = pd.read_csv('./%s/exp%d/test_img.csv'%(root, index_exp))
        label = pd.read_csv('./%s/exp%d/test_label.csv'%(root, index_exp))
        return img['fileName'].values, label['solution_time'].values


class TopoplotLoader(data.Dataset):
    def __init__(self, root, mode, num_time=1, transform=None, scale=False, index_exp=0, index_split=0):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)
            num_time : Number of time steps

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(root, mode, index_exp, index_split)
        self.mode = mode
        self.num_time = num_time
        self.transform = transform
        self.scale=scale
        self.index_exp = index_exp
        self.index_split = index_split
        '''
        with open('./images/img.data', 'rb') as fp:
            self.dict_img = pickle.load(fp)
        '''
        
        if self.scale:
            print('Load scaler...')
            with open('%s/scaler.data'%(self.root), 'rb') as fp:
                self.dict_scaler = pickle.load(fp)
        
        print("> Found %d images, %d examples" % (len(self.img_name)*self.num_time, len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [T, H, W, C] to [T, C, H, W]
                         
            step4. Return processed image and label
        """
        for i_time in range(self.num_time):
            
            fileName = self.img_name[index][:-1] + str(i_time)
            
            path = '%s/exp%d/%s.png'%(self.root, self.index_exp, fileName)
            label = self.label[index]
            img = io.imread(path)
            
            '''
            label = self.label[index]
            img = self.dict_img[fileName]
            '''

            # set to pixel value to 0~1
            img = img/255

            # Choose RGB
            img = img[:,:,:3]
            
            # Scale back
            if self.scale:
                for i_channel in range(3):
                    img[:,:,i_channel] = img[:,:,i_channel] * self.dict_scaler[fileName]['scale'][i_channel]\
                        + self.dict_scaler[fileName]['min_'][i_channel]
            
            if i_time == 0:
                imgs = np.zeros((self.num_time, img.shape[0], img.shape[1], img.shape[2]))
            
            imgs[i_time,:,:,:] = img.copy()
            
        if self.num_time == 1:
            imgs = np.squeeze(imgs, 0)
        
        sample = {'image': imgs, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class Rescale(object):
    
    def __init__(self, output_size, num_time=1):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.num_time = num_time
        
    def __call__(self, sample):
        
        if self.num_time == 1:
            image = transform.resize(sample['image'], (self.output_size, self.output_size))
        else:
            imgs = np.zeros((self.num_time, self.output_size, self.output_size, sample['image'].shape[3]))
            for i_time in range(self.num_time):
                
                img = transform.resize(sample['image'][i_time,:], (self.output_size, self.output_size))
                imgs[i_time,:] = img
                
            image = imgs
        
        return {'image': image, 'label': sample['label']}

class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __init__(self, num_time=1):
        self.num_time = num_time
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if self.num_time == 1:
            image = image.transpose((2,0,1))
        else:
            image = image.transpose((0,3,1,2))
            
        return {'image': torch.from_numpy(image).float(), 'label': torch.tensor(label).float()}
    
