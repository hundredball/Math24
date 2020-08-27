import torch.nn as nn

__all__ = ['MyNet', 'mynet']


class MyNet(nn.Module):

    def __init__(self, in_features):
        super(MyNet, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=1, groups=in_features)
        self.batchnorm = nn.BatchNorm1d(in_features)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
            )

    def forward(self, x):
        
        # Depthwise separable convolution
        x = x.reshape((x.shape[0], -1, 1))
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.sigmoid(x)
        
        # Linear
        x = x.reshape((x.shape[0],-1))
        x = self.fc(x)

        return x


def mynet(in_features):
    model = MyNet(in_features)
    return model
