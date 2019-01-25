import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import GesturesDataset

# parser = argparse.ArgumentParser(description='PyTorch conv2d')
# parser.add_argument('--batch-size', type=int, default=4, metavar='N',
#                     help='input batch size for training (default: 4)')
# parser.add_argument('--epochs', type=int, default=2, metavar='N',
#                     help='number of epochs to train (default: 2)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# # parser.add_argument('--seed', type=int, default=1, metavar='S',
# #                     help='random seed (default: 1)')
# parser.add_argument('--save-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before checkpointing')
# parser.add_argument('--resume', action='store_true', default=False,
#                     help='resume training from checkpoint')
# parser.add_argument('--rgb', type=bool, default=False,
#                     help='input rgb images')
# parser.add_argument('--nframes', type=int, default=15,
#                     help='number of frames per input')
#
#
# args = parser.parse_args()
#
#
N_CLASSES = 12
# RESIZE_DIM = 64

# output channels inspired by tutorial pytorch
class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(args.nframes if not args.rgb else args.nframes * 3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*14*14, 120) # input varia a seconda della dimensione dell'immagine di input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, N_CLASSES)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #da 64x64 -> 32x32
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # 32x32 -> 14x14 -> linear
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



# def main():

    # use_cuda = torch.cuda.is_available() and not args.no_cuda
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # # torch.manual_seed(args.seed)
    # # if use_cuda:
    # #   torch.cuda.manual_seed(args.seed)
    #
    # # DATALOADER
    #
    # # definizione di traformate da passare al costruttore del dataset
    # #gray scale
    #
    # transform = None
    # if not args.rgb:
    #     transform = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (1.0,))
    #
    #     ])
    #
    # dataset = GesturesDataset(csv_path='csv_dataset', mode='ir', normalization_type=1)
    # data = dataset[10]
    #
    #
    #
    # net = LeNet().to(device)


# if __name__ == '__main__':
#     main()