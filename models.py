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

        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_classes, num_layers=2, batch_first=True):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        # self.fc = nn.Linear(hidden_size, num_classes)

        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), num_classes)

    def forward(self, x):
        # transpose 0 with 1 (batch, seq_len)

        if self.batch_first:
            x = x.permute(1, 0, 2)

        # out, hidden = self.lstm(x, hidden)
        out, (ht, ct) = self.lstm(x)
        # vogliamo solo l'ultimo output

        out = out[-1]

        # fc_output = F.relu(self.fc(out))
        fc_output = F.relu(self.fc1(out))
        fc_output = self.fc2(fc_output)
        return fc_output

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


class Gru(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, num_classes, num_layers=2, batch_first=True):
        super(Gru, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # transpose 0 with 1 (batch, seq_len)

        if self.batch_first:
            x = x.permute(1, 0, 2)

        # out, hidden = self.lstm(x, hidden)
        out, ht = self.gru(x)
        # vogliamo solo l'ultimo output

        out = out[-1]

        fc_output = self.fc(out)
        return fc_output

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_classes, num_layers=2, rnn_type='LSTM', final_layer='fc'):
        super(Rnn, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.final_layer = final_layer
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':

            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.rnn1 = nn.LSTM(input_size=hidden_size, hidden_size=num_classes, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
            self.rnn1 = nn.GRU(input_size=hidden_size, hidden_size=num_classes, num_layers=num_layers,
                               batch_first=True)
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, hidden_size//2 if self.final_layer == 'fc2' else num_classes)
        self.fc1 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        # the order must be batch, seq_len, input size

        out, (ht, ct) = self.rnn(x)

        if self.final_layer == 'fc':
            # vogliamo solo l'ultimo output

            out = out[:, -1]

            out = self.dropout(out)
            out = self.fc(out)

        elif self.final_layer == 'fc2':
            # vogliamo solo l'ultimo output

            out = out[:, -1]

            out = self.dropout(out)
            out = self.fc(out)
            out = F.relu(out)
            out = self.dropout(out)
            out = self.fc1(out)

        elif self.final_layer == 'lstm':
            out, (ht, ct) = self.rnn1(out)
            if self.classification:
                out = out[:, -1]

        return out

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(28672, 4096) # modificato l'input
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        # added
        # self.fc9 = nn.Linear(2048, 1024)
        # self.fc10 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, self.num_flat_features(h))
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        logits = self.fc8(h)

        # added
        # h = self.relu(self.fc8(h))
        # h = self.dropout(h)
        # h = self.relu(self.fc9(h))
        # h = self.dropout(h)
        # logits = self.fc10(h)
        # probs = self.softmax(logits)

        return logits

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

