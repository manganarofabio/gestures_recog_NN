import torch
from torch import nn, optim
import argparse
from dataloader import GesturesDataset
from models import LeNet
from trainer import Trainer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch conv2d')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--mode', type=str, default='ir',
                    help='mode of dataset')
parser.add_argument('--rgb', type=bool, default=False,
                    help='input rgb images')
parser.add_argument('--n_frames', type=int, default=30,
                    help='number of frames per input')
parser.add_argument('--input_size', type=int, default=64,
                    help='number of frames per input')
parser.add_argument('--n_classes', type=int, default=12,
                    help='number of frames per input')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')

args = parser.parse_args()


def main():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # DATALOADER

    train_dataset = GesturesDataset(csv_path='csv_dataset', mode=args.mode, normalization_type=1,
                                    n_frames=args.n_frames)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # paramteri per la rete
    in_channels = args.n_frames if not args.rgb else args.n_frames * 3
    n_classes = args.n_classes

    model = LeNet(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_function = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load("./checkpointLeNet.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        print("Resuming state:\n-epoch: {}\n{}".format(start_epoch, model))

    writer = SummaryWriter()
    # add info experiment
    writer.add_text('Info experiment',
                    "batch_size:{}\n\nepochs:{}\n\nlr:{}\n\nmomentum:{}"
                    "\n\nn_frames:{}"
                    "\n\ninput_size:{}"
                    "\n\nn_classes:{}"
                    "\n\nmode:{}"
                    "\n\nrgb:{}"
                    "".format(args.batch_size, args.epochs, args.lr, args.momentum, args.n_frames, args.input_size,
                              args.n_classes, args.mode, args.rgb))

    trainer = Trainer(model, loss_function, optimizer, train_loader, device, writer)

    for ep in range(start_epoch, args.epochs):
        trainer.train(ep)


if __name__ == '__main__':
    main()
