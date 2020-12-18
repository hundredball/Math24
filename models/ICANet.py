import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA, FastICA

__all__ = ['icarnn', 'icarnnown', 'icalstmown']

class ICARNN(nn.Module):
    
    def __init__(self, V_weight, W_weight, V_bias, num_channel, hidden_size, output_size):
        super(ICARNN, self).__init__()
        
        # ICA layer
        self.V = nn.Linear(num_channel, num_channel)
        self.W = nn.Linear(num_channel, num_channel, bias=False)
        
        # RNN layer
        self.hidden_size = hidden_size
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.nonlinearity = 'tanh'
        
        self.rnn = nn.RNN(num_channel, hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, nonlinearity=self.nonlinearity)
        self.fc = nn.Linear(hidden_size*self.num_directions, output_size)
        
        # Initialize V by whitenting matrix and W by unmixing matrix
        with torch.no_grad():
            # Weight in Linear: (out_features, in_features)
            self.V.weight = nn.Parameter(torch.FloatTensor(V_weight))
            self.W.weight = nn.Parameter(torch.FloatTensor(W_weight))
            self.V.bias = nn.Parameter(torch.FloatTensor(V_bias))
        self.V.weight.requires_grad = False
        self.W.weight.requires_grad = False
        self.V.bias.requires_grad = False
        
    def forward(self, x, h0):
        
        # (example, channel, time) -> (time, example, channel)
        x = x.permute(2,0,1)
        
        # ICA
        x = self.V(x)
        x = self.W(x)
        
        # RNN
        x, hn = self.rnn(x,h0)
        x = x[-1,:,:]    # Select last output
        x = self.fc(x)
        
        return x
    
    def initHidden(self, num_example, device):
        return torch.zeros(self.num_directions*self.num_layers, num_example, self.hidden_size, device=device)

def icarnn(train_data, train_target, hidden_size=32, output_size=1):
    '''
    Build ICARNN network, all subjects share the same model

    Parameters
    ----------
    train_data : np.ndarray (example, channel, time)
        Two dimensional training data
    train_target : np.ndarray
        Training targets
    hidden_size : int
        Hidden size of rnn layer
    output_size : int
        Final output size

    Returns
    -------
    None.

    '''
    assert isinstance(train_data, np.ndarray) and train_data.ndim==3
    assert isinstance(train_target, np.ndarray) and train_target.ndim==1
    
    # (example, channel, time) -> (example*time, channel)
    train_data = np.swapaxes(train_data, 1, 2)
    train_data = train_data.reshape((-1, train_data.shape[2]))
    
    # Apply ICA for train_data
    num_channel = train_data.shape[1]
    ica = FastICA(n_components=num_channel, random_state=23)
    ica.fit(train_data)
    
    unmixing_matrix = ica.components_.dot(np.linalg.inv(ica.whitening_))
    bias = -ica.whitening_.dot(ica.mean_)
    
    return ICARNN(ica.whitening_, unmixing_matrix, bias, num_channel, hidden_size, output_size)

def icarnnown(train_data, train_sub, train_target, hidden_size=32, output_size=1):
    '''
    Build ICARNN networks, each subjects has their own model

    Parameters
    ----------
    train_data : np.ndarray (example, channel, time)
        Two dimensional training data
    train_target : np.ndarray
        Training targets
    hidden_size : int
        Hidden size of rnn layer
    output_size : int
        Final output size

    Returns
    -------
    None.

    '''
    assert isinstance(train_data, np.ndarray) and train_data.ndim==3
    assert isinstance(train_sub, np.ndarray) and train_sub.ndim==1
    assert isinstance(train_target, np.ndarray) and train_target.ndim==1
    
    # Separate each subject
    num_channel = train_data.shape[1]
    subIDs = np.unique(train_sub)
    print('Create %d ICARNN'%(len(subIDs)))
    models = []
    
    for subID in subIDs:
        sub_data = train_data[train_sub==subID,:]
        print('Sub %d: %d'%(subID, len(sub_data)))
        
        # (example, channel, time) -> (example*time, channel)
        sub_data = np.swapaxes(sub_data, 1, 2)
        sub_data = sub_data.reshape((-1, sub_data.shape[2]))
        
        # Apply ICA for train_data of certain subjects
        ica = FastICA(n_components=num_channel, random_state=23)
        ica.fit(sub_data)

        unmixing_matrix = ica.components_.dot(np.linalg.inv(ica.whitening_))
        bias = -ica.whitening_.dot(ica.mean_)
        
        models.append(ICARNN(ica.whitening_, unmixing_matrix, bias, num_channel, hidden_size, output_size))
    
    return models

class ICALSTM(nn.Module):
    
    def __init__(self, V_weight, W_weight, V_bias, num_channel, hidden_size, output_size):
        super(ICALSTM, self).__init__()
        
        # ICA layer
        self.V = nn.Linear(num_channel, num_channel)
        self.W = nn.Linear(num_channel, num_channel, bias=False)
        
        # RNN layer
        self.hidden_size = hidden_size
        self.bidirectional = False
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        
        self.lstm = nn.LSTM(num_channel, hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size*self.num_directions, output_size)
        
        # Initialize V by whitenting matrix and W by unmixing matrix
        with torch.no_grad():
            # Weight in Linear: (out_features, in_features)
            self.V.weight = nn.Parameter(torch.FloatTensor(V_weight))
            self.W.weight = nn.Parameter(torch.FloatTensor(W_weight))
            self.V.bias = nn.Parameter(torch.FloatTensor(V_bias))
        self.V.weight.requires_grad = False
        self.W.weight.requires_grad = False
        self.V.bias.requires_grad = False
        
    def forward(self, x, h0, c0):
        
        # (example, channel, time) -> (time, example, channel)
        x = x.permute(2,0,1)
        
        # ICA
        x = self.V(x)
        x = self.W(x)
        
        # RNN
        x, hn, cn = self.lstm(x,h0,c0)
        x = x[-1,:,:]    # Select last output
        x = self.fc(x)
        
        return x
    
    def initHidden(self, num_example, device):
        h0 = torch.zeros(self.num_directions*self.num_layers, num_example, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_directions*self.num_layers, num_example, self.hidden_size, device=device)
        return h0, c0
    
def icalstmown(train_data, train_sub, train_target, hidden_size=32, output_size=1):
    '''
    Build ICALSTM networks, each subjects has their own model

    Parameters
    ----------
    train_data : np.ndarray (example, channel, time)
        Two dimensional training data
    train_target : np.ndarray
        Training targets
    hidden_size : int
        Hidden size of rnn layer
    output_size : int
        Final output size

    Returns
    -------
    None.

    '''
    assert isinstance(train_data, np.ndarray) and train_data.ndim==3
    assert isinstance(train_sub, np.ndarray) and train_sub.ndim==1
    assert isinstance(train_target, np.ndarray) and train_target.ndim==1
    
    # Separate each subject
    num_channel = train_data.shape[1]
    subIDs = np.unique(train_sub)
    print('Create %d ICALSTM'%(len(subIDs)))
    models = []
    
    for subID in subIDs:
        sub_data = train_data[train_sub==subID,:]
        print('Sub %d: %d'%(subID, len(sub_data)))
        
        # (example, channel, time) -> (example*time, channel)
        sub_data = np.swapaxes(sub_data, 1, 2)
        sub_data = sub_data.reshape((-1, sub_data.shape[2]))
        
        # Apply ICA for train_data of certain subjects
        ica = FastICA(n_components=num_channel, random_state=23)
        ica.fit(sub_data)

        unmixing_matrix = ica.components_.dot(np.linalg.inv(ica.whitening_))
        bias = -ica.whitening_.dot(ica.mean_)
        
        models.append(ICALSTM(ica.whitening_, unmixing_matrix, bias, num_channel, hidden_size, output_size))
    
    return models