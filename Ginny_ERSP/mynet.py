import torch.nn as nn

__all__ = ['MyNet', 'mynet']


class MyNet(nn.Module):

    def __init__(self, in_features):
        super(MyNet, self).__init__()
        self.act = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


def mynet(in_features):
    model = MyNet(in_features)
    return model
