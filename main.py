import torch
from torch import nn, optim
import argparse
from dataloader import GesturesDataset
from models import LeNet
from trainer import Trainer
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch conv2d')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
parser.add_argument('--rgb', type=bool, default=False,
                    help='input rgb images')
parser.add_argument('--nframes', type=int, default=30,
                    help='number of frames per input')


args = parser.parse_args()


N_CLASSES = 12
RESIZE_DIM = 64


def main():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    # torch.manual_seed(args.seed)
    # if use_cuda:
    #   torch.cuda.manual_seed(args.seed)

    # DATALOADER


    # transform = None
    # if not args.rgb:
    #     transform = transforms.Compose([
    #         # transforms.ToPILImage(),
    #         transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (1.0,))
    #
    #     ])

    train_dataset = GesturesDataset(csv_path='csv_dataset', mode='ir', normalization_type=1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = LeNet(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    if args.resume:
        model.load_state_dict(torch.load('net.pth'))
        optimizer.load_state_dict(torch.load('optimizer.pth'))

    trainer = Trainer(model, loss_function, optimizer, train_loader, device)

    for ep in range(args.epochs):
        trainer.train(ep)


if __name__ == '__main__':
    main()