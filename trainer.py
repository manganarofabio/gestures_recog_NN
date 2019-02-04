from __future__ import division, print_function
import torch
from tensorboardX import SummaryWriter
import utilities


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, train_loader, test_loader, device, writer, personal_name,
                 dynamic_lr=False, verbose=True):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        # data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device
        self.verbose = verbose
        self.writer = writer
        self.personal_name = personal_name
        self.dynamic_lr = dynamic_lr

    def train(self, epoch):

        self.model.train()
        print("###TRAIN###")
        running_loss, train_accuracy, train_running_accuracy = 0., 0., 0.

        nof_steps = len(self.train_loader)
        if epoch == 0:
            print('nof:', nof_steps)
        train_losses = []
        for step, data in enumerate(self.train_loader):
            img, target = data
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(img)
            # loss = self.loss_function(output, torch.max(target, 1)[1])
            loss = self.loss_function(output, target.squeeze(dim=1))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            train_losses.append(loss.item())
            # Train accuracy
            predicted = torch.argmax(output, dim=1)
            correct = target.squeeze()
            train_accuracy = float((predicted == correct).sum().item()) / output.size()[1]
            train_running_accuracy += train_accuracy

            if self.verbose:
                if step % 2 == 1:
                    print('[epoch: {:d}, iter:  {:5d}] loss(avg): {:.3f}\t train_acc(avg): {:.3f}%'.format(
                        epoch, step, running_loss / (step+1), train_running_accuracy/(step+1) * 100))
                    self.writer.add_scalar('data/train_loss', loss.item(), epoch * nof_steps + step)
                    self.writer.add_scalar('data/train_accuracy', train_accuracy, epoch * nof_steps + step)

        if self.dynamic_lr:
            utilities.adjust_learning_rate(epoch, self.optimizer)





        # SALVATAGGIO STATO a fine di ogni epoca

        state = {

            'epoch': epoch,
            'running_loss': loss.item(),
            'avg_loss':  running_loss / (step+1),
            'train_losses': train_losses,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'info': self.personal_name

        }

        torch.save(state, 'checkpoint{}_{}.pth.tar'.format(str(self.model).split('(')[0], self.personal_name))

    def test(self, epoch):

        self.model.eval()
        print("###TEST###")
        test_loss, running_test_loss, test_accuracy, test_running_accuracy = 0., 0., 0., 0.
        nof_steps = len(self.test_loader)

        with torch.no_grad():
            for step, data in enumerate(self.test_loader):
                img, target = data
                img, target = img.to(self.device), target.to(self.device)
                output = self.model(img)
                test_loss = self.loss_function(output, target.squeeze(dim=1))
                running_test_loss += test_loss
                predicted = torch.argmax(output, dim=1)
                correct = target.squeeze()
                test_accuracy = float((predicted == correct).sum().item()) / output.size()[1]
                test_running_accuracy += test_accuracy

                if self.verbose:
                    if step % 2 == 1:
                        print('[iter:  {:5d}] test_loss(avg): {:.3f}\t test_acc(avg): {:.3f}%'.format(
                              step, running_test_loss / (step + 1), test_running_accuracy / (step + 1) * 100))
                        self.writer.add_scalar('data/test_loss', test_loss.item(), epoch * nof_steps + step)
                        self.writer.add_scalar('data/test_accuracy', test_accuracy, epoch * nof_steps + step)

    def __del__(self):
        self.writer.close()
