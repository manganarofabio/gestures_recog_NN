import torch
from torch import nn, optim
import argparse
from dataloader import GesturesDataset
from models import LeNet, AlexNet, AlexNetBN, Vgg16
from trainer import Trainer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import models
import time, os
import random
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch conv2d')

parser.add_argument('-model', type=str, default='DenseNet121P',
                    help='model of CNN')
parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=99, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--opt', type=str, default='SGD',
                    help="Optimizer (default: SGD)")
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dn_lr', default=True,
                    help="adjust dinamically lr")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='M',
                    help='Adam weight_decay (default: 0.0001')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1994, metavar='S',
                    help='random seed (default: 1994)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
parser.add_argument('--n_workers', type=int, default=2,
                    help="number of workers")
parser.add_argument('--mode', type=str, default='depth_ir',
                    help='mode of dataset')
parser.add_argument('--n_frames', type=int, default=40,
                    help='number of frames per input')
parser.add_argument('--input_size', type=int, default=224, #227 alexnet, 64 lenet, 224 vgg16 and Resnet, denseNet
                    help='number of frames per input')
parser.add_argument('--n_classes', type=int, default=12,
                    help='number of frames per input')

args = parser.parse_args()


def main():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    rgb = False
    if args.mode == 'rgb':
        rgb = True

    # DATALOADER

    train_dataset = GesturesDataset(csv_path='csv_dataset', train=True, mode=args.mode, rgb=rgb, normalization_type=1,
                                    n_frames=args.n_frames, resize_dim=args.input_size, transform_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    test_dataset = GesturesDataset(csv_path='csv_dataset', train=False, mode=args.mode, rgb=rgb, normalization_type=1,
                                   n_frames=args.n_frames, resize_dim=args.input_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    # paramteri per la rete

    in_channels = args.n_frames if not rgb else args.n_frames * 3
    n_classes = args.n_classes

    if args.model == 'LeNet':
        model = LeNet(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)

    elif args.model == 'AlexNet':
        model = AlexNet(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)

    elif args.model == 'AlexNetBN':
        model = AlexNetBN(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)

    elif args.model == "Vgg16":
        model = Vgg16(input_channels=in_channels, input_size=args.input_size, n_classes=n_classes).to(device)

    elif args.model == "Vgg16P":
        model = models.vgg16(pretrained=True)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['0'] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        model.classifier._modules['6'] = nn.Linear(4096, n_classes)
        # model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

    elif args.model == "ResNet18P":
        model = models.resnet18(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        model._modules['conv1'] = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

    elif args.model == "ResNet34P":
        model = models.resnet34(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        model._modules['conv1'] = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

    elif args.model == "DenseNet121P":
        model = models.densenet121(pretrained=True)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        model = model.to(device)



    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load("./checkpoint{}.pth.tar".format(args.model))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        print("Resuming state:\n-epoch: {}\n{}".format(start_epoch, model))

    #name experiment
    personal_name = "DenseNet121_depth_ir_dataset_updated"
    log_dir = "logs"
    if personal_name:
        exp_name = (("exp_{}_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    else:
        exp_name = (("exp_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    writer = SummaryWriter("{}".format(os.path.join(log_dir, exp_name)))

    # add info experiment
    writer.add_text('Info experiment',
                    "model:{}"
                    "\n\nbatch_size:{}"
                    "\n\nepochs:{}"
                    "\n\noptimizer:{}"
                    "\n\nlr:{}"
                    "\n\ndn_lr:{}"
                    "\n\nmomentum:{}"
                    "\n\nweight_decay:{}"
                    "\n\nn_frames:{}"
                    "\n\ninput_size:{}"
                    "\n\nn_classes:{}"
                    "\n\nmode:{}"
                    "\n\nn_workers:{}"
                    "\n\nseed:{}"
                    "".format(args.model, args.batch_size, args.epochs, args.opt, args.lr, args.dn_lr, args.momentum,
                              args.weight_decay, args.n_frames, args.input_size,
                              args.n_classes, args.mode, args.n_workers, args.seed))

    trainer = Trainer(model, loss_function, optimizer, train_loader, test_loader, device, writer, personal_name,
                      args.dn_lr)

    print("model: {}".format(args.model + personal_name))
    for ep in range(start_epoch, args.epochs):
        trainer.train(ep)
        trainer.test(ep)


if __name__ == '__main__':
    main()
