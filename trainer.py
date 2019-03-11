from __future__ import division, print_function
import torch
from tensorboardX import SummaryWriter
import utilities
import os


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, train_loader, validation_loader, batch_size, initial_lr, device, writer, personal_name,
                 log_file, weight_dir,
                 dynamic_lr=False, rnn=False, verbose=True, num_classes=12):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.rnn = rnn

        # data loaders
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.batch_size = batch_size
        self.initial_lr = initial_lr

        self.device = device
        self.verbose = verbose
        self.writer = writer
        self.personal_name = personal_name
        self.log_file = log_file
        self.dynamic_lr = dynamic_lr
        self.num_classes = num_classes
        self.weight_dir = weight_dir

        # class accuracy
        self.class_correct = [0. for i in range(self.num_classes)]
        self.class_total = [0. for i in range(self.num_classes)]

    def train(self, epoch):

        self.model.train()
        print("###TRAIN###")
        self.log_file.write("TRAIN\n")
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
                    # print('[epoch: {:d}, iter:  {:5d}] loss(avg): {:.3f}\t train_acc(avg): {:.3f}%'.format(
                    #     epoch, step, running_loss / (step+1), train_running_accuracy/(step+1) * 100))
                    # self.writer.add_scalar('data/train_loss', loss.item(), epoch * nof_steps + step)
                    # self.writer.add_scalar('data/train_accuracy', train_accuracy, epoch * nof_steps + step)

                    utilities.print_training_info(log_file=self.log_file, writer=self.writer, epoch=epoch, step=step,
                                                  running_loss=running_loss, train_running_accuracy=train_running_accuracy,
                                                  loss=loss, nof_steps=nof_steps, train_accuracy=train_accuracy)
        if self.dynamic_lr:
            print('dynamic')
            utilities.adjust_learning_rate(self.initial_lr, epoch, self.optimizer)
        else:
            print('static')

        state = {

            'epoch': epoch,
            'running_loss': loss.item(),
            'avg_loss':  running_loss / (step+1),
            'train_losses': train_losses,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'info': self.personal_name

        }

        # if epoch % 10 == 0:
        # salvo ogni epoca

        save_path = '/projects/fabio/weights/gesture_recog_weights/{}/ep_{:03d}_checkpoint{}_{}_.pth.tar'.format(
                        self.weight_dir, epoch, str(self.model).split('(')[0], self.personal_name)

        if not os.path.exists('/projects/fabio/weights/gesture_recog_weights/{}'.format(self.weight_dir)):
            os.makedirs('/projects/fabio/weights/gesture_recog_weights/{}'.format(self.weight_dir))

        torch.save(state, save_path)

        # torch.save(state, '/projects/fabio/weights/gesture_recog_weights/checkpoint{}_{}.pth.tar'.format(str(self.model).split('(')[0], self.personal_name))

    def val(self, epoch):

        self.model.eval()
        print("###VALIDATION###")
        self.log_file.write("###VALIDATION###\n")
        validation_loss, running_validation_loss, validation_accuracy, validation_running_accuracy = 0., 0., 0., 0.
        nof_steps = len(self.validation_loader)

        with torch.no_grad():
            for step, data in enumerate(self.validation_loader):
                x, label = data
                x, label = x.to(self.device), label.to(self.device)
                output = self.model(x)
                validation_loss = self.loss_function(output, label.squeeze(dim=1))
                running_validation_loss += validation_loss
                predicted = torch.argmax(output, dim=1)

                correct = label
                # if self.batch_size != 1:
                correct = correct.squeeze(dim=1)
                validation_accuracy = float((predicted == correct).sum().item()) / len(correct)
                validation_running_accuracy += validation_accuracy

                c = (predicted == correct).squeeze()
                for i in range(len(correct)):
                    i_label = correct[i]
                    if self.batch_size > 1:
                        self.class_correct[i_label] += c[i].item()
                    else:
                        self.class_correct[i_label] += c.item()
                    self.class_total[i_label] += 1

                if self.verbose:
                    if step % 2 == 1:
                        # print('[epoch: {:d}, iter:  {:5d}] validation_loss(avg): {:.3f}\t validation_acc(avg): {:.3f}%'.format(
                        #       epoch, step, running_validation_loss / (step + 1), validation_running_accuracy / (step + 1) * 100))
                        # self.writer.add_scalar('data/validation_loss', validation_loss.item(), epoch * nof_steps + step)
                        # self.writer.add_scalar('data/validation_accuracy', validation_accuracy, epoch * nof_steps + step)
                        utilities.print_validation_info(log_file=self.log_file, writer=self.writer, epoch=epoch,
                                                        step=step, running_loss=running_validation_loss,
                                                        validation_running_accuracy=validation_running_accuracy,
                                                        loss=validation_loss, nof_steps=nof_steps,
                                                        validation_accuracy=validation_accuracy)

    def __del__(self):
        self.writer.close()
