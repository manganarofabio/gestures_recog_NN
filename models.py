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
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))


class AlexNet(nn.Module):

    def __init__(self, input_channels, input_size, n_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, n_classes)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), (3, 3), stride=2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout((F.relu(self.fc1(x))))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))

class AlexNet(nn.Module):

    def __init__(self, input_channels, input_size, n_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, n_classes)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), (3, 3), stride=2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout((F.relu(self.fc1(x))))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))


class AlexNetBN(nn.Module):

    def __init__(self, input_channels, input_size, n_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, n_classes)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (3, 3), stride=2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (3, 3), stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(F.relu(self.bn5(self.conv5(x))), 4)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout((F.relu(self.fc1(x))))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))


class Vgg16(nn.Module):

    def __init__(self, input_channels, input_size, n_classes):
        super(Vgg16, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=25088, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=n_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2), stride=2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(F.relu(self.conv7(x)), (2, 2), stride=2)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.max_pool2d(F.relu(self.conv10(x)), (2, 2), stride=2)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(F.relu(self.conv13(x)), (2, 2), stride=2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))


