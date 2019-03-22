from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.final_layer = final_layer
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':

            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                               dropout=0.2)
            self.rnn1 = nn.LSTM(input_size=hidden_size, hidden_size=num_classes, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
            self.rnn1 = nn.GRU(input_size=hidden_size, hidden_size=num_classes, num_layers=num_layers,
                               batch_first=True)
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, hidden_size//2 if self.final_layer == 'fc1' else num_classes)
        self.fc1 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        # the order must be batch, seq_len, input size

        out, (ht, ct) = self.rnn(x)

        if self.final_layer == 'fc':
            # vogliamo solo l'ultimo output

            out = out[:, -1]

            out = self.dropout(out)
            out = self.fc(out)

        elif self.final_layer == 'fc1':
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

    def __init__(self, num_classes, rgb):
        super(C3D, self).__init__()

        self.num_classe = num_classes
        self.rgb = rgb

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

        # 8192 dipende dal numero di frame che gli si passano 30 -> 8192, 40 -> 16384
        self.fc6 = nn.Linear(8192, 4096) # modificato l'input da modificare in base all'input 112*112 (112*200 = 28672; 112*112 = 16384) (prima era 8192
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487) # num classes
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


# Conv-lstm

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda()
                )


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""


class DeepConvLstm(nn.Module):

    def __init__(self, input_channels_conv, input_size_conv, n_classes, batch_size, n_frames=40):
        super(DeepConvLstm, self).__init__()

        self.input_channels_conv = input_channels_conv
        self.input_size_conv = input_size_conv
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_frames = n_frames

        self.conv = nn.Conv2d(in_channels=self.input_channels_conv, out_channels=32, kernel_size=3)
        #self.conv0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))

        self.convLstm = ConvLSTM(input_size=(14, 14),
                                 input_dim=128,
                                 hidden_dim=[128, 128],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=True
                                 )

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=12, kernel_size=3)

    def forward(self, x):

        # trasformo il tensore in input da b, s, c, h, w in b*s, :
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = F.max_pool2d(F.relu(self.conv(x)), kernel_size=(2, 2))
        # x = F.max_pool2d(F.relu(self.conv0(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        # ritorno alla dimensione b, s, c, h, w per la conv lstm

        x = x.view(self.batch_size, self.n_frames, x.shape[1], x.shape[2], x.shape[3])

        # x = x.unsqueeze(dim=2)
        x = self.convLstm(x)
        # prendo l'ultimo hs
        x = x[0][0][:, -1]
        out = F.adaptive_avg_pool2d(self.conv3(x), (1, 1))
        out = torch.squeeze(out, dim=-1)
        out = torch.squeeze(out, dim=-1)

        return out


class CrossConvNet(nn.Module):
    def __init__(self, n_classes, depth_model, ir_model, rgb_model, mode='depth_ir_rgb', cross_mode=None):
        super(CrossConvNet, self).__init__()

        self.n_classes = n_classes


        self.depth_model = depth_model
        self.ir_model = ir_model
        self.rgb_model = rgb_model
        self.mode = mode
        self.cross_mode = cross_mode



        #modifiche ultimop layer reti
        # self.model_depth_ir.classifier = nn.Linear(in_features=2208, out_features=128)
        # self.model_rgb.classifier = nn.Linear(in_features=2208, out_features=128)
        # self.model_lstm.fc = nn.Linear(in_features=512, out_features=128)
        self.conv = nn.Conv2d(in_channels=2208 * 3 if self.mode == 'depth_ir_rgb' else 2208 * 2
                              , out_channels=512, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3)
        self.fc = nn.Linear(in_features=1152, out_features=256)
        self.fc1 = nn.Linear(in_features=256, out_features=self.n_classes)

    def forward(self, *input):


        out_depth, out_ir, out_rgb, x_depth, x_ir, x_rgb = None, None, None, None, None, None
        xs = [value for value in input]

        if self.mode == 'depth_ir_rgb':
            x_depth = xs[0]
            x_ir = xs[1]
            x_rgb = xs[2]
        elif self.mode == 'depth_ir':
            x_depth = xs[0]
            x_ir = xs[1]
        elif self.mode == 'depth_rgb':
            x_depth = xs[0]
            x_rgb = xs[1]
        elif self.mode == 'ir_rgb':
            x_ir = xs[0]
            x_rgb = xs[1]

        with torch.no_grad():
            if x_depth is not None:
                out_depth = self.depth_model(x_depth)

            if x_ir is not None:
                out_ir = self.ir_model(x_ir)

            if x_rgb is not None:
                out_rgb = self.rgb_model(x_rgb)

        # concateno le  3 uscite
        if out_depth is not None and out_ir is not None and out_rgb is not None:
            if self.cross_mode == 'avg':
                x = (out_depth + out_ir + out_rgb)/3
            else:
                x = torch.cat((out_depth, out_ir, out_rgb), 1)
        # concateno gli out
        elif out_depth is not None and out_ir is not None:
            if self.cross_mode == 'avg':
                x = (out_depth + out_ir)/2
            else:
                x = torch.cat((out_depth, out_ir), 1)
        elif out_depth is not None and out_rgb is not None:
            if self.cross_mode == 'avg':
                x = (out_depth + out_rgb)/2
            else:
                x = torch.cat((out_depth, out_rgb), 1)
        elif out_ir is not None and out_rgb is not None:
            if self.cross_mode == 'avg':
                x = (out_ir + out_rgb)/2
            else:
                x = torch.cat((out_ir, out_rgb), 1)

        if self.cross_mode == 'avg':
            out = x
        else:
            x = F.relu(self.conv(x))
            x = F.dropout2d(x, p=0.2)
            x = F.relu(self.conv1(x))
            x = F.dropout2d(x, p=0.2)

            x = x.view(-1, self.num_flat_features(x))

            x = F.dropout(self.fc(x), p=0.2)
            out = self.fc1(x)

        return out

        # # concateno gli out
        # if out_depth and out_ir:
        #     out = torch.cat((out_depth, out_ir), 1)
        # elif out_depth and out_rgb:
        #     out = torch.cat((out_depth, out_rgb), 1)
        # elif out_ir and out_rgb:
        #     out = torch.cat((out_ir, out_rgb), 1)

        # out = F.dropout(F.relu(self.fc(out)), p=0.2)
        # out = F.dropout(F.relu(self.fc1(out)), p=0.2)
        # out = self.fc2(out)
        #
        # return out

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))
