import torch
from torch import nn, optim
import argparse
from dataloader import GesturesDataset
from models import LeNet, AlexNet, AlexNetBN, Vgg16, Lstm, Gru, C3D
from trainer import Trainer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import models
import time, os
import random
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch conv2d')

parser.add_argument('--model', type=str, default='Lstm',
                    help='model of NN')
parser.add_argument('--pretrained', default=True,
                    help='pretrained net')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--opt', type=str, default='SGD',
                    help="Optimizer (default: SGD)")
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dn_lr', default=True,
                    help="adjust dinamically lr")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M',
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
parser.add_argument('--mode', type=str, default='rgb',
                    help='mode of dataset')
parser.add_argument('--n_frames', type=int, default=40,
                    help='number of frames per input')
parser.add_argument('--input_size', type=int, default=224, # 227 alexnet, 64 lenet, 224 vgg16 and Resnet, denseNet
                    help='number of frames per input')
parser.add_argument('--train_transforms', default=True,
                    help="training transforms")
parser.add_argument('--n_classes', type=int, default=12,
                    help='number of frames per input')
# LSTM
parser.add_argument('--input_size_rnn', type=int, default=675, # num features leap_motion
                    help='input size of rnn')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='hidden size of rnn')
parser.add_argument('--n_layers', type=int, default=2,
                    help='n layers of rnn')

args = parser.parse_args()


def main():

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

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

    train_dataset = GesturesDataset(model=args.model, csv_path='csv_dataset', train=True, mode=args.mode, rgb=rgb, normalization_type=1,
                                    n_frames=args.n_frames, resize_dim=args.input_size,
                                    transform_train=args.train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    test_dataset = GesturesDataset(model=args.model, csv_path='csv_dataset', train=False, mode=args.mode, rgb=rgb, normalization_type=1,
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
        model = models.vgg16(pretrained=args.pretrained)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['0'] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        model.classifier._modules['6'] = nn.Linear(4096, n_classes)
        # model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

    elif args.model == "ResNet18P":
        model = models.resnet18(pretrained=args.pretrained)
        for params in model.parameters():
            params.requires_grad = False
        model._modules['conv1'] = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

    elif args.model == "ResNet34P":
        model = models.resnet34(pretrained=args.pretrained)
        for params in model.parameters():
            params.requires_grad = False
        model._modules['conv1'] = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)

    elif args.model == "DenseNet121P":
        model = models.densenet121(pretrained=args.pretrained)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        model = model.to(device)

    elif args.model == "DenseNet161P":
        model = models.densenet161(pretrained=args.pretrained)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=2208, out_features=n_classes, bias=True)
        model = model.to(device)

    elif args.model == "DenseNet169P":
        model = models.densenet169(pretrained=args.pretrained)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=1664, out_features=n_classes, bias=True)
        model = model.to(device)

    elif args.model == "DenseNet201P":
        model = models.densenet201(pretrained=args.pretrained)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=1920, out_features=n_classes, bias=True)
        model = model.to(device)
    # RNN
    elif args.model == 'Lstm':
        model = Lstm(input_size=args.input_size_rnn, hidden_size=args.hidden_size, batch_size=args.batch_size,
                     num_classes=args.n_classes, num_layers=args.n_layers).to(device)
    elif args.model == 'Gru':
        model = Gru(input_size=args.input_size_rnn, hidden_size=args.hidden_size, batch_size=args.batch_size,
                    num_classes=args.n_classes, num_layers=args.n_layers).to(device)
    # C3D

    elif args.model == 'C3D':
        model = C3D(args.n_classes).to(device)
    else:
        print('no model selected')
        exit(-1)

    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load("/projects/fabio/weights/gesture_recog_weights/checkpoint{}.pth.tar".format(args.model))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        print("Resuming state:\n-epoch: {}\n{}".format(start_epoch, model))

    #name experiment
    personal_name = "{}_{}_036".format(args.model, args.mode)
    info_experiment = "{}: ".format(personal_name)
    log_dir = "/projects/fabio/logs/gesture_recog_logs"
    if personal_name:
        exp_name = (("exp_{}_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    else:
        exp_name = (("exp_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    writer = SummaryWriter("{}".format(os.path.join(log_dir, exp_name)))

    # add info experiment
    writer.add_text('Info experiment',
                    "model:{}"
                    "\n\npretrained:{}"
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
                    "\n\ninfo:{}"
                    "".format(args.model, args.pretrained, args.batch_size, args.epochs, args.opt, args.lr, args.dn_lr, args.momentum,
                              args.weight_decay, args.n_frames, args.input_size,
                              args.n_classes, args.mode, args.n_workers, args.seed, info_experiment))

    rnn = True if args.model == 'Lstm' else False

    trainer = Trainer(model, loss_function, optimizer, train_loader, test_loader, args.batch_size, device, writer, personal_name,
                      args.dn_lr, rnn=rnn)

    print("experiment: {}".format(personal_name))
    start = time.time()
    for ep in range(start_epoch, args.epochs):
        trainer.train(ep)
        trainer.test(ep)

    end = time.time()
    h, rem = divmod(end - start, 3600)
    m, s, = divmod(rem, 60)
    print("\nelapsed time (ep.{}):{:0>2}:{:0>2}:{:05.2f}".format(args.epochs, int(h), int(m), s))


if __name__ == '__main__':
    main()
