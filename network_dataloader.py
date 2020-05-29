import pandas as pd
from torch.utils import data
import numpy as np
from skimage import io, transform
import torch

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('./images/train_img.csv')
        label = pd.read_csv('./images/train_label.csv')
        return img['fileName'].values, label['solution_time'].values
    else:
        img = pd.read_csv('./images/test_img.csv')
        label = pd.read_csv('./images/test_label.csv')
        return img['fileName'].values, label['solution_time'].values


class TopoplotLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.img_shape = np.zeros(3, dtype=int)
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

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
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + '/' + self.img_name[index] + '.png'
        label = self.label[index]
        img = io.imread(path)
        
        # set to pixel value to 0~1
        img = img/255
        
        # Choose first three channels of color
        img = img[:,:,:3]
        
        sample = {'image': img, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class Rescale(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image = transform.resize(sample['image'], (self.output_size, self.output_size))
        
        return {'image': image, 'label': sample['label']}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, sample):
        image = sample['image']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        
        image = image[top: top+new_h, left: left+new_w]
        
        return {'image': image, 'label': sample['label']}

class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image).float(), 'label': torch.tensor(label).float()}
    
