import torch
import torch.nn as nn

__all__ = ['mynet', 'simplefc', 'deepfc']


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
    
def simplefc(in_features):
    return SimpleFC(in_features)

def deepfc(in_features):
    return DeepFC(in_features)

def mynet(in_features):
    model = MyNet(in_features)
    return model
