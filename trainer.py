from __future__ import division, print_function
import torch
import time
import numpy as np
from torch.backends import cudnn
from os.path import join
from tensorboardX import SummaryWriter
from datetime import datetime


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, data_train, device, verbose=True):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        # data loaders
        self.data_train = data_train

        self.device = device
        self.verbose = verbose
        self.writer = SummaryWriter(log_dir="./logs/1/train")

    def train(self, epoch):

        self.model.train()

        running_loss = 0

        nof_steps = len(self.data_train)
        print('')
        train_losses = []
        for step, data in enumerate(self.data_train):
            img, target = data
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.loss_function(output, torch.max(target, 1)[1])
            loss.backward()
            running_loss += loss.item()
            train_losses.append(loss.item())

            if self.verbose:
                if step % 2 == 1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch, step, running_loss / step))
                    # self.writer.add_scalar('data/running_loss', running_loss, epoch, step)
                    running_loss = 0.0
                    self.writer.add_scalar('data/running_loss', running_loss/step, step)

        self.writer.add_scalar('data/loss_epoch', loss.item(), epoch)

        torch.save(self.model.state_dict(), 'model.pth')
        torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        torch.save(train_losses, 'train_losses.pth')
        # self.writer.export_scalars_to_json("./all_scalars.json")
        # self.writer.close()
