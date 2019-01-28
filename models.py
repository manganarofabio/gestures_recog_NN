from torch import nn
import torch
from torch.nn import functional as F


# output channels inspired by tutorial pytorch
class LeNet(nn.Module):
    def __init__(self, input_channels, input_size, n_classes):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16*int((input_size*7/32)*(input_size*7/32)), 120) # input varia a seconda della dimensione dell'immagine di input passare l'input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #da 64x64 -> 32x32
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), (2, 2)) # 32x32 -> 14x14 -> linear
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))

