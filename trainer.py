from __future__ import division, print_function
import torch
from tensorboardX import SummaryWriter
import tqdm



class Trainer(object):
    def __init__(self, model, loss_function, optimizer, data_train, device, writer, verbose=True):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        # data loaders
        self.data_train = data_train

        self.device = device
        self.verbose = verbose
        self.writer = writer

    def train(self, epoch):

        self.model.train()

        running_loss = 0.
        train_accuracy = 0.
        train_running_accuracy = 0.

        nof_steps = len(self.data_train)
        print('')
        train_losses = []
        for step, data in enumerate(self.data_train):
            img, target = data
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(img)
            # loss = self.loss_function(output, torch.max(target, 1)[1])
            loss = self.loss_function(output, target.squeeze())
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
                    self.writer.add_scalar('data/running_loss', loss.item(), epoch * nof_steps + step)
                    self.writer.add_scalar('data/train_accuracy', train_accuracy, epoch * nof_steps + step)



        # SALVATAGGIO STATO a fine di ogni epoca

        state = {

            'epoch': epoch,
            'running_loss': loss.item(),
            'avg_loss':  running_loss / (step+1),
            'train_losses': train_losses,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),

        }

        torch.save(state, 'checkpointLeNet.pth.tar')
        # torch.save(self.model.state_dict(), 'model.pth')
        # torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        # torch.save(train_losses, 'train_losses.pth')
        # self.writer.export_scalars_to_json("./all_scalars.json")
        # self.writer.close()

    def __del__(self):
        self.writer.close()
