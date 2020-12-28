import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA, FastICA

__all__ = ['mynet', 'simplefc', 'deepfc', 'pcafc', 'pcafcown', 'pcafc_sd']


class MyNet(nn.Module):

    def __init__(self, in_features):
        super(MyNet, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=1, groups=in_features)
        self.batchnorm = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            )

    def forward(self, x):
        
        # Depthwise separable convolution
        x = x.reshape((x.shape[0], -1, 1))
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        # Linear
        x = x.reshape((x.shape[0],-1))
        x = self.fc(x)
        x = x.flatten()

        return x

class SimpleFC(nn.Module):
    
    def __init__(self, in_features):
        super(SimpleFC, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            )
        '''
        # Set all the weights absolute
        with torch.no_grad():
            self.fc[0].weight = nn.Parameter(torch.abs(self.fc[0].weight.data))
            self.fc[2].weight = nn.Parameter(torch.abs(self.fc[2].weight.data))
            
        self.fc[0].weight.requires_grad = True
        self.fc[2].weight.requires_grad = True
        '''
        
        # Set all the weights constant
        nn.init.xavier_normal_(self.fc[0].weight, gain=0.1)
        nn.init.xavier_normal_(self.fc[2].weight, gain=0.1)
        
    def forward(self, x):
        
        x = self.fc(x)
        #x=  60*x
        x = x.flatten()
        
        return x
    
class DeepFC(nn.Module):
    
    def __init__(self, in_features):
        super(DeepFC, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.Sigmoid(),
            nn.Linear(in_features//2, in_features//4),
            nn.Sigmoid(),
            nn.Linear(in_features//4, in_features//8),
            nn.Sigmoid(),
            nn.Linear(in_features//8, in_features//16),
            nn.Sigmoid(),
            nn.Linear(in_features//16, 1),
            nn.ReLU()
            )
        
        
        # Set all the weights absolute
        for i in range(len(self.fc)):
            if i%2 == 0:
                with torch.no_grad():
                    self.fc[i].weight = nn.Parameter(torch.abs(self.fc[i].weight.data))
                self.fc[i].weight.requires_grad = True
        
        
    def forward(self, x):
        
        x = self.fc(x)
        x = x.flatten()
        
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class PCAFC(nn.Module):
    
    def __init__(self, in_features, weight1, weight2, bias1, mode):
        assert weight1.shape[1]==weight2.shape[0]
        super(PCAFC, self).__init__()
        
        
        num_hidden = weight1.shape[1]
        '''
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        '''
        self.fc1 = nn.Linear(in_features, num_hidden, bias=True)
        self.fc2 = nn.Linear(num_hidden, 1, bias=False)
        
        # Initialize fc1 by PCA matrix and fc2 by pseudoinverse
        with torch.no_grad():
            # Weight in Linear: (out_features, in_features)
            self.fc1.weight = nn.Parameter(torch.FloatTensor(weight1.T))
            #self.fc2.weight = nn.Parameter(torch.FloatTensor(weight2.T))
            self.fc1.bias = nn.Parameter(torch.FloatTensor(bias1))
        self.fc1.weight.requires_grad = False
        #self.fc2.weight.requires_grad = True
        self.fc1.bias.requires_grad=False
        
        if mode == 'class':
            #self.act = nn.Identity()
            self.act = nn.Sigmoid()
            self.out = nn.Sigmoid()
        elif mode == 'reg':
            self.act = nn.Sigmoid()
            self.out = Identity()
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.out(x)
        
        return x
    
    def stopFreezingFE(self):
        
        print('>>>Stop freezing PCA layer')
        for name, item in self._modules.items():
            if name == 'fc1':
                for p in item.parameters():
                    p.requires_grad = True

def pcafc(train_data, train_target, num_components=30, mode='reg', C=1):
    '''
    Build PCAFC network

    Parameters
    ----------
    train_data : np.ndarray (epoch, features)
        Two dimensional training data
    train_target : np.ndarray
        Training targets
    num_components : int
        Number of components for PCA, which is equivalent as number of units in hidden layer
    mode : str
        Classification or regression (class, reg)

    Returns
    -------
    None.

    '''
    assert isinstance(train_data, np.ndarray) and train_data.ndim==2
    assert isinstance(train_target, np.ndarray) and train_target.ndim==1
    assert isinstance(num_components, int) and num_components>0
    assert mode=='class' or mode=='reg'
    
    # Apply PCA for train_data
    pca = PCA(n_components=num_components)
    pca.fit(train_data)
    weight1 = pca.components_.T     # (train_data.shape[1]) x (num_components)
    bias1 = -weight1.T.dot(np.mean(train_data,0))   # (num_components)
    z = sigmoid(train_data.dot(weight1))
    
    # Get Least square solution
    if mode == 'reg':
        weight2 = np.linalg.pinv(z).dot(train_target)[:,np.newaxis]
    elif mode == 'class':
        # C is the user-defined parameter and provides a trade-off between the distance of 
        # the separating margin and training error
        train_target[train_target==0] = -1      # Let target be (1,-1) for the binary ELM
        weight2 = z.T.dot(np.linalg.inv( np.eye(z.shape[0])/C+z.dot(z.T) )).dot(train_target)[:,np.newaxis]
    
    return PCAFC(train_data.shape[1], weight1, weight2, bias1, mode)

def pcafcown(train_data, train_sub, train_target, num_components=30, mode='reg', C=1):
    '''
    Build PCAFC networks, each subjects has their own model

    Parameters
    ----------
    train_data : np.ndarray (epoch, features)
        Two dimensional training data
    train_target : np.ndarray
        Training targets
    train_sub : np.ndarray
        Training subject ID
    num_components : int
        Number of components for PCA, which is equivalent as number of units in hidden layer
    mode : str
        Classification or regression (class, reg)

    Returns
    -------
    None.

    '''
    assert isinstance(train_data, np.ndarray) and train_data.ndim==2
    assert isinstance(train_sub, np.ndarray) and train_sub.ndim==1
    assert isinstance(train_target, np.ndarray) and train_target.ndim==1
    assert isinstance(num_components, int) and num_components>0
    assert mode=='class' or mode=='reg'
    
    # Separate each subject
    subIDs = np.unique(train_sub)
    print('Create %d PCAFC'%(len(subIDs)))
    models = []
    
    for subID in subIDs:
        sub_data = train_data[train_sub==subID,:]
        sub_target = train_target[train_sub==subID]
        print('Sub %d: %d'%(subID, len(sub_data)))
        
        models.append(pcafc(sub_data, sub_target, num_components=num_components, mode=mode, C=C))
    
    return models

    
class PCAFC_SD(nn.Module):
    
    def __init__(self, num_signal_features, num_sd_features, weight1, weight2, bias1, mode):
        assert weight1.shape[0]==num_signal_features
        assert weight1.shape[1]+num_sd_features == weight2.shape[0]
        super(PCAFC_SD, self).__init__()
        self.num_signal_features = num_signal_features
        
        num_hidden = weight1.shape[1]
        self.fc1 = nn.Linear(num_signal_features, num_hidden, bias=True)
        self.fc2 = nn.Linear(num_hidden+num_sd_features, 1, bias=False)
        
        # Initialize fc1 by PCA matrix and fc2 by pseudoinverse
        with torch.no_grad():
            # Weight in Linear: (out_features, in_features)
            self.fc1.weight = nn.Parameter(torch.FloatTensor(weight1.T))
            self.fc2.weight = nn.Parameter(torch.FloatTensor(weight2.T))
            self.fc1.bias = nn.Parameter(torch.FloatTensor(bias1))
        self.fc1.weight.requires_grad = True
        self.fc2.weight.requires_grad = True
        self.fc1.bias.requires_grad=True
        
        if mode == 'class':
            #self.act = nn.Identity()
            self.act = nn.Sigmoid()
            self.out = nn.Sigmoid()
        elif mode == 'reg':
            self.act = nn.Sigmoid()
            self.out = Identity()
        
    def forward(self, data):
        
        # Split data into signal and sd parts
        signal, sd = data[:,:self.num_signal_features], data[:,self.num_signal_features:]
        
        # PCA for data
        x = self.fc1(signal)
        x = self.act(x)
        
        # Concatenate x with sd
        x = torch.cat((x, sd), 1)
        
        # Second fc
        x = self.fc2(x)
        x = self.out(x)
        
        return x
    
def pcafc_sd(train_data, train_target, num_signal_features, num_components=30, mode='reg', C=1):
    '''
    Build PCAFC with subject ID and difficulty level

    Parameters
    ----------
    train_data : np.ndarray (epoch, features)
        Two dimensional training data
    train_target : np.ndarray
        Training targets
    num_components : int
        Number of components for PCA, which is equivalent as number of units in hidden layer
    mode : str
        Classification or regression (class, reg)

    Returns
    -------
    None.

    '''
    assert isinstance(train_data, np.ndarray) and train_data.ndim==2
    assert isinstance(train_target, np.ndarray) and train_target.ndim==1
    assert isinstance(num_signal_features, int) and num_signal_features<=train_data.shape[1]
    assert isinstance(num_components, int) and num_components>0
    assert mode=='class' or mode=='reg'
    
    # Split train_data into signal and sd parts
    train_signal, train_sd = train_data[:, :num_signal_features], train_data[:, num_signal_features:]
    
    # Apply PCA for train_data
    pca = PCA(n_components=num_components)
    pca.fit(train_signal)
    weight1 = pca.components_.T     # (num_signal_features) x (num_components)
    bias1 = -weight1.T.dot(np.mean(train_signal,0))   # (num_components)
    z = sigmoid(train_signal.dot(weight1))
    
    # Concatenate z with sd features
    z = np.concatenate((z, train_sd), axis=1)
    
    # Get Least square solution
    if mode == 'reg':
        weight2 = np.linalg.pinv(z).dot(train_target)[:,np.newaxis]
    elif mode == 'class':
        # C is the user-defined parameter and provides a trade-off between the distance of 
        # the separating margin and training error
        train_target[train_target==0] = -1      # Let target be (1,-1) for the binary ELM
        weight2 = z.T.dot(np.linalg.inv( np.eye(z.shape[0])/C+z.dot(z.T) )).dot(train_target)[:,np.newaxis]
    
    return PCAFC_SD(num_signal_features, train_data.shape[1]-num_signal_features, weight1, weight2, bias1, mode)
    
def simplefc(in_features):
    return SimpleFC(in_features)

def deepfc(in_features):
    return DeepFC(in_features)

def mynet(in_features):
    model = MyNet(in_features)
    return model

def relu(data):
    data[data<0] = 0
    return data

def sigmoid(data):
    return 1/(1+np.exp(-data))
    
