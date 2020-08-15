import torch
import torch.nn as nn

__all__ = ['convnet', 'convlstm', 'convfc']

def conv7():

    net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1), 
        nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
    
    return net

class ConvNet(nn.Module):

    def __init__(self, input_size):
        super(ConvNet, self).__init__()
        self.conv = conv7()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        in_features = int(input_size*input_size*128/(4**3))
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ConvLSTM(nn.Module):
    
    def __init__(self, input_size, fc_hidden_size, rnn_hidden_size, seq_len):
        super(ConvLSTM, self).__init__()
        
        in_features = in_features = int(input_size*input_size*128/(4**3))
        
        for i in range(seq_len):
            setattr(self, 'conv%d'%(i), conv7())
            setattr(self, 'fc%d'%(i), nn.Linear(in_features, fc_hidden_size))
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.rnn = nn.LSTM(fc_hidden_size, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, 1)
        
    def forward(self, x):
        outputs = []
        
        # Convolution for each time step
        for i in range(x.shape[1]):
            output = getattr(self, 'conv%d'%(i))(x[:,i,:,:,:])
            output = output.view(x.shape[0],-1)
            output = getattr(self, 'fc%d'%(i))(output)
            output = self.relu(output)
            outputs.append(output)
            
        outputs = torch.stack(outputs)
        
        # Dimension for the input of rnn (seq, batch, features) 
        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(outputs)
        
        # Take output from the last time
        outputs = outputs[-1,:]
        outputs = self.sigmoid(outputs)
        
        '''
        
        # Combine all the outputs
        outputs = outputs.transpose(0,1).reshape(x.shape[0],-1)
        '''
        
        outputs = self.fc(outputs)
        
        return outputs
    
class ConvFC(nn.Module):
    
    def __init__(self, input_size, fc_hidden_size, seq_len):
        super(ConvFC, self).__init__()
        
        in_features = in_features = int(input_size*input_size*128/(4**3))
        
        for i in range(seq_len):
            setattr(self, 'conv%d'%(i), conv7())
            setattr(self, 'fc%d'%(i), nn.Linear(in_features, fc_hidden_size))
        
        self.relu = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(fc_hidden_size*seq_len, 1)
        
    def forward(self, x):
        outputs = []
        
        # Convolution for each time step
        for i in range(x.shape[1]):
            output = getattr(self, 'conv%d'%(i))(x[:,i,:,:,:])
            output = output.view(x.shape[0],-1)
            output = getattr(self, 'fc%d'%(i))(output)
            output = self.relu(output)
            outputs.append(output)
            
        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0,1).reshape(x.shape[0], -1)
        outputs = self.fc(outputs)
        
        return outputs
    
    
def convnet(input_size):
    model = ConvNet(input_size)
    return model

def convlstm(input_size, fc_hidden_size, rnn_hidden_size, seq_len):
    model = ConvLSTM(input_size, fc_hidden_size, rnn_hidden_size, seq_len)
    return model

def convfc(input_size, fc_hidden_size, seq_len):
    model = ConvFC(input_size, fc_hidden_size, seq_len)
    return model
    
    
