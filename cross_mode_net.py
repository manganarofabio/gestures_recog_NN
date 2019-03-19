import torch
from torch import nn, optim
import argparse
from dataloader_cross_mode import GesturesDatasetCrossMode
from models import CrossConvNet
from trainer_cross_mode import TrainerCrossMode
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import models
import time, os
import random
import numpy as np
from torch.nn import functional as F

import argparse

parser = argparse.ArgumentParser(description='PyTorch conv2d')

parser.add_argument('--rgb_model_path', type=str, default=None)
parser.add_argument('--depth_model_path', type=str, default=None)
parser.add_argument('--ir_model_path', type=str, default=None)



parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--opt', type=str, default='SGD',
                    help="Optimizer (default: SGD)")
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dn_lr', action='store_true', default=False,
                    help="adjust dinamically lr")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M',
                    help='Adam weight_decay (default: 0.0001')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1994, metavar='S',
                    help='random seed (default: 1994)')
parser.add_argument('--n_workers', type=int, default=2,
                    help="number of workers")
parser.add_argument('--n_frames', type=int, default=40,
                    help='number of frames per input')
parser.add_argument('--input_size', type=int, default=224, # 227 alexnet, 64 lenet, 224 vgg16 and Resnet, denseNet
                    help='number of frames per input')
parser.add_argument('--train_transforms', action='store_true', default=False,
                    help="training transforms")
parser.add_argument('--n_classes', type=int, default=12,
                    help='number of cla per input')
parser.add_argument('--mode', type=str, default='depth_ir_rgb')
parser.add_argument('--cross_mode', type=str, default=None)

parser.add_argument('--weight_dir', type=str)
parser.add_argument('--exp_name', type=str, default="prova")
args = parser.parse_args()


# class CrossConvNet(nn.Module):
#     def __init__(self, n_classes, depth_model, ir_model, rgb_model, mode='depth_ir_rgb', cross_mode=None):
#         super(CrossConvNet, self).__init__()
#
#         self.n_classes = n_classes
#
#         self.depth_model = depth_model
#         self.ir_model = ir_model
#         self.rgb_model = rgb_model
#
#         self.mode = mode
#         self.cross_mode = cross_mode
#
#
#         #modifiche ultimop layer reti
#         # self.model_depth_ir.classifier = nn.Linear(in_features=2208, out_features=128)
#         # self.model_rgb.classifier = nn.Linear(in_features=2208, out_features=128)
#         # self.model_lstm.fc = nn.Linear(in_features=512, out_features=128)
#         self.conv = nn.Conv2d(in_channels=2208 * 3 if self.mode == 'depth_ir_rgb' else 2208 * 2
#                               , out_channels=512, kernel_size=3)
#         self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3)
#         self.fc = nn.Linear(in_features=1152, out_features=256)
#         self.fc1 = nn.Linear(in_features=256, out_features=self.n_classes)
#
#     def forward(self, *input):
#
#
#         out_depth, out_ir, out_rgb, x_depth, x_ir, x_rgb = None, None, None, None, None, None
#         xs = [value for value in input]
#
#         if self.mode == 'depth_ir_rgb':
#             x_depth = xs[0]
#             x_ir = xs[1]
#             x_rgb = xs[2]
#         elif self.mode == 'depth_ir':
#             x_depth = xs[0]
#             x_ir = xs[1]
#         elif self.mode == 'depth_rgb':
#             x_depth = xs[0]
#             x_rgb = xs[1]
#         elif self.mode == 'ir_rgb':
#             x_ir = xs[0]
#             x_rgb = xs[1]
#
#         with torch.no_grad():
#             if x_depth is not None:
#                 out_depth = self.depth_model(x_depth)
#
#             if x_ir is not None:
#                 out_ir = self.ir_model(x_ir)
#
#             if x_rgb is not None:
#                 out_rgb = self.rgb_model(x_rgb)
#
#         # concateno le  3 uscite
#         if out_depth is not None and out_ir is not None and out_rgb is not None:
#             if self.cross_mode == 'avg':
#                 x = np.vstack((out_depth.numpy(), out_ir.numpy(), out_rgb.numpy()))
#                 x = np.mean(x, axis=0)
#                 x = torch.from_numpy(x)
#             else:
#                 x = torch.cat((out_depth, out_ir, out_rgb), 1)
#         # concateno gli out
#         elif out_depth is not None and out_ir is not None:
#             if self.cross_mode == 'avg':
#                 x = np.vstack((out_depth.numpy(), out_ir.numpy()))
#                 x = np.mean(x, axis=0)
#                 x = torch.from_numpy(x)
#             else:
#                 x = torch.cat((out_depth, out_ir), 1)
#         elif out_depth is not None and out_rgb is not None:
#             if self.cross_mode == 'avg':
#                 x = np.vstack((out_depth.numpy(), out_rgb.numpy()))
#                 x = np.mean(x, axis=0)
#                 x = torch.from_numpy(x)
#             else:
#                 x = torch.cat((out_depth, out_rgb), 1)
#         elif out_ir is not None and out_rgb is not None:
#             if self.cross_mode == 'avg':
#                 x = np.vstack((out_ir.numpy(), out_rgb.numpy()))
#                 x = np.mean(x, axis=0)
#                 x = torch.from_numpy(x)
#             else:
#                 x = torch.cat((out_ir, out_rgb), 1)
#
#         if self.cross_mode == 'avg':
#             out = x
#         else:
#             x = F.relu(self.conv(x))
#             x = F.dropout2d(x, p=0.2)
#             x = F.relu(self.conv1(x))
#             x = F.dropout2d(x, p=0.2)
#
#             x = x.view(-1, self.num_flat_features(x))
#
#             x = F.dropout(self.fc(x), p=0.2)
#             out = self.fc1(x)
#
#         return out
#
#         # # concateno gli out
#         # if out_depth and out_ir:
#         #     out = torch.cat((out_depth, out_ir), 1)
#         # elif out_depth and out_rgb:
#         #     out = torch.cat((out_depth, out_rgb), 1)
#         # elif out_ir and out_rgb:
#         #     out = torch.cat((out_ir, out_rgb), 1)
#
#         # out = F.dropout(F.relu(self.fc(out)), p=0.2)
#         # out = F.dropout(F.relu(self.fc1(out)), p=0.2)
#         # out = self.fc2(out)
#         #
#         # return out
#
#     def num_flat_features(self, x):
#         return torch.prod(torch.tensor(x.size()[1:]))


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    random.seed(1994)
    np.random.seed(1994)
    torch.manual_seed(1994)
    if use_cuda:
        torch.cuda.manual_seed(1994)
        torch.backends.cudnn.deterministic = True

    depth, ir, rgb = False, False, False
    if args.mode == 'depth_ir_rgb':
        depth = True
        ir = True
        rgb = True
    elif args.mode == 'depth_ir':
        depth = True
        ir = True
    elif args.mode == 'depth_rgb':
        depth = True
        rgb = True
    elif args.mode == 'ir_rgb':
        ir = True
        rgb = True

    ir_kind = 'depth_ir'

    train_dataset = GesturesDatasetCrossMode(csv_path='csv_dataset', train=True, depth=depth, ir=ir, rgb=rgb,
                                             ir_kind=ir_kind, mode=args.mode)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    validation_dataset = GesturesDatasetCrossMode(csv_path='csv_dataset', train=False, depth=depth, ir=ir, rgb=rgb,
                                             ir_kind=ir_kind, mode=args.mode)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_depth, model_ir, model_rgb = None, None, None
    if rgb:
        model_rgb = models.densenet161().to(device)
        # for params in model_rgb.parameters():
        #     params.required_grad = False
        model_rgb.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames*3, out_channels=96, kernel_size=(7, 7),
                                                         stride=(2, 2), padding=(3, 3))
        model_rgb = torch.nn.Sequential(*(list(model_rgb.children())[0][:-1]))
        # model_rgb.classifier = nn.Linear(in_features=2208, out_features=12, bias=True)

        checkpoint_rgb = torch.load(args.rgb_model_path, map_location=device)['state_dict']
        model_rgb.load_state_dict(checkpoint_rgb, strict=False)

    if ir:
        model_ir = models.densenet161().to(device)
        # for params in model_ir.parameters():
        #     params.required_grad = False
        model_ir.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                        stride=(2, 2), padding=(3, 3))
        model_ir = torch.nn.Sequential(*(list(model_ir.children())[0][:-1]))

        checkpoint_ir = torch.load(args.ir_model_path, map_location=device)['state_dict']
        model_ir.load_state_dict(checkpoint_ir, strict=False)
        # model_rgb.classifier = nn.Linear(in_features=2208, out_features=12, bias=True)

    if depth:
        model_depth = models.densenet161().to(device)
        # for params in model_depth.parameters():
        #     params.required_grad = False
        model_depth.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                           stride=(2, 2), padding=(3, 3))
        model_depth = torch.nn.Sequential(*(list(model_depth.children())[0][:-1]))
        # model_rgb.classifier = nn.Linear(in_features=2208, out_features=12, bias=True)
        checkpoint_depth = torch.load(args.depth_model_path, map_location=device)['state_dict']
        model_depth.load_state_dict(checkpoint_depth, strict=False)

    # if rgb:
    #     model_rgb.load_state_dict(((torch.load(args.rgb_model_path))['state_dict']))
    # # state_dict = model_dense_rgb['state_dict']
    # # model_rgb.load_state_dict(state_dict)
    # if ir:
    #     model_ir.load_state_dict(((torch.load(args.ir_model_path))['state_dict']))
    # if depth:
    #     model_depth.load_state_dict(((torch.load(args.depthdepth_model_path))['state_dict']))

    model = CrossConvNet(args.n_classes, depth_model=model_depth, ir_model=model_ir, rgb_model=model_rgb, mode=args.mode
                         ).to(device)

    optimizer = None
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss().to(device)

    personal_name = "CrossNet_{}".format(args.mode)
    # name experiment
    # if depth and rgb and ir:
    #     personal_name = "CrossNet_depth_rgb_ir_{}".format(args.exp_name)
    # elif depth and rgb:
    #     personal_name = "CrossNet_depth_rgb_{}".format(args.exp_name)
    # elif depth and ir:
    #     personal_name = "CrossNet_depth_{}_{}".format(ir_kind, args.exp_name)
    # elif ir and rgb:
    #     personal_name = "CrossNet_{}_rgb_{}".format(ir_kind, args.exp_name)

    info_experiment = "{}: ".format(personal_name)
    log_dir = "/projects/fabio/logs/gesture_recog_logs/exps"
    log_file = open("{}/{}.txt".format("/projects/fabio/logs/gesture_recog_logs/txt_logs", personal_name), 'w')
    log_file.write(personal_name + "\n\n")
    weight_dir = personal_name
    if personal_name:
        exp_name = (("exp_{}_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    else:
        exp_name = (("exp_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    writer = SummaryWriter("{}".format(os.path.join(log_dir, exp_name)))

    # add info experiment
    writer.add_text('Info experiment',
                    "personal_name: {}"
                    "\n\ndepth_model_path: {}"
                    "\n\nir_model_path: {}"
                    "\n\nrgb_model_path: {}"
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
                    "\n\ncross_mode:{}"
                    "\n\nn_workers:{}"
                    "\n\nseed:{}"
                    "\n\ninfo:{}"
                    "".format(personal_name, args.depth_model_path, args.ir_model_path, args.rgb_model_path,
                              args.batch_size, args.epochs, args.opt, args.lr, args.dn_lr,
                              args.momentum,
                              args.weight_decay, args.n_frames, args.input_size,
                              args.n_classes, args.mode, args.cross_mode, args.n_workers, args.seed, info_experiment))

    # # name experiment
    # personal_name = "cross_mode_depth_ir_rgb_lstm_3fc"
    # info_experiment = "{}: ".format(personal_name)
    # log_dir = "/projects/fabio/logs/gesture_recog_logs"
    # if personal_name:
    #     exp_name = (("exp_{}_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    # else:
    #     exp_name = (("exp_{}".format(time.strftime("%c"), personal_name)).replace(" ", "_")).replace(":", "-")
    # writer = SummaryWriter("{}".format(os.path.join(log_dir, exp_name)))
    #
    # # add info experiment

    trainer = TrainerCrossMode(model=model, loss_function=loss_function, optimizer=optimizer, train_loader=train_loader,
                               validation_loader=validation_loader, batch_size=args.batch_size, device=device, writer=writer,
                               initial_lr=args.lr, dynamic_lr=args.dn_lr, log_file=log_file, personal_name=personal_name,
                               weight_dir=weight_dir, mode=args.mode)
    print("experiment: {}".format(personal_name))
    start = time.time()
    for ep in range(args.epochs):
        trainer.train(ep)
        trainer.val(ep)



        # display classes results
    classes = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11']
    # for i in range(12):
    #     print('Accuracy of {} : {:.3f}%%'.format(
    #         classes[i], 100 * trainer.class_correct[i] / trainer.class_total[i]))

    log_file.write("\n\n")
    for i in range(args.n_classes):
        log_file.write('Accuracy of {} : {:.3f}%\n'.format(
            classes[i], 100 * trainer.class_correct[i] / trainer.class_total[i]))
    log_file.close()

    end = time.time()
    h, rem = divmod(end - start, 3600)
    m, s, = divmod(rem, 60)
    print("\nelapsed time (ep.{}):{:0>2}:{:0>2}:{:05.2f}".format(args.epochs, int(h), int(m), s))


if __name__ == '__main__':
    main()
