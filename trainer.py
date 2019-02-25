from __future__ import division, print_function
import torch
from tensorboardX import SummaryWriter
import utilities


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, train_loader, test_loader, batch_size, device, writer, personal_name,
                 dynamic_lr=False, rnn=False, verbose=True, num_classes=12):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.rnn = rnn

        # data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size

        self.device = device
        self.verbose = verbose
        self.writer = writer
        self.personal_name = personal_name
        self.dynamic_lr = dynamic_lr
        self.num_classes = num_classes

        # class accuracy
        self.class_correct = [0. for i in range(self.num_classes)]
        self.class_total = [0. for i in range(self.num_classes)]

    def train(self, epoch):

        self.model.train()
        print("###TRAIN###")
        running_loss, train_accuracy, train_running_accuracy = 0., 0., 0.

        nof_steps = len(self.train_loader)
        if epoch == 0:
            print('nof:', nof_steps)
        train_losses = []

        for step, data in enumerate(self.train_loader):
            x, label = data
            x, label = x.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            # introdotto 20/02
            #label = label.permute(1, 0)  # switchare
            # output = output.view(-1, output.size(2))  #squeeze su 0
            #output = output.squeeze(dim=0)
            loss = self.loss_function(output, label.squeeze(dim=1))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            train_losses.append(loss.item())
            # Train accuracy
            predicted = torch.argmax(output, dim=1)
            correct = label
            #if self.batch_size != 1:
            correct = correct.squeeze(dim=1)
            train_accuracy = float((predicted == correct).sum().item()) / len(correct)
            train_running_accuracy += train_accuracy

            if self.verbose:
                if step % 2 == 1:
                    print('[epoch: {:d}, iter:  {:5d}] loss(avg): {:.3f}\t train_acc(avg): {:.3f}%'.format(
                        epoch, step, running_loss / (step+1), train_running_accuracy/(step+1) * 100))
                    self.writer.add_scalar('data/train_loss', loss.item(), epoch * nof_steps + step)
                    self.writer.add_scalar('data/train_accuracy', train_accuracy, epoch * nof_steps + step)

        if self.dynamic_lr:
            utilities.adjust_learning_rate(epoch, self.optimizer)

        state = {

            'epoch': epoch,
            'running_loss': loss.item(),
            'avg_loss':  running_loss / (step+1),
            'train_losses': train_losses,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'info': self.personal_name

        }

        torch.save(state, '/projects/fabio/weights/gesture_recog_weights/checkpoint{}_{}.pth.tar'.format(str(self.model).split('(')[0], self.personal_name))

    def test(self, epoch):

        self.model.eval()
        print("###TEST###")
        test_loss, running_test_loss, test_accuracy, test_running_accuracy = 0., 0., 0., 0.
        nof_steps = len(self.test_loader)

        with torch.no_grad():
            for step, data in enumerate(self.test_loader):
                x, label = data
                x, label = x.to(self.device), label.to(self.device)
                output = self.model(x)
                test_loss = self.loss_function(output, label.squeeze(dim=1))
                running_test_loss += test_loss
                predicted = torch.argmax(output, dim=1)

                correct = label
                # if self.batch_size != 1:
                correct = correct.squeeze(dim=1)
                test_accuracy = float((predicted == correct).sum().item()) / len(correct)
                test_running_accuracy += test_accuracy

                c = (predicted == correct).squeeze()
                for i in range(self.batch_size):
                    i_label = label[i]
                    if self.batch_size > 1:
                        self.class_correct[i_label] += c[i].item()
                    else:
                        self.class_correct[i_label] += c.item()
                    self.class_total[i_label] += 1

                if self.verbose:
                    if step % 2 == 1:
                        print('[epoch: {:d}, iter:  {:5d}] test_loss(avg): {:.3f}\t test_acc(avg): {:.3f}%'.format(
                              epoch, step, running_test_loss / (step + 1), test_running_accuracy / (step + 1) * 100))
                        self.writer.add_scalar('data/test_loss', test_loss.item(), epoch * nof_steps + step)
                        self.writer.add_scalar('data/test_accuracy', test_accuracy, epoch * nof_steps + step)

    def __del__(self):
        self.writer.close()
