from __future__ import division, print_function
import torch
from tensorboardX import SummaryWriter
import utilities
import os


class TrainerCrossMode(object):
    def __init__(self, model, loss_function, optimizer, train_loader, validation_loader, batch_size, device, writer, log_file,
                 personal_name, weight_dir, initial_lr,
                 dynamic_lr=False, verbose=True, num_classes=12, mode='z_ir_rgb'):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer


        # data loaders
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.batch_size = batch_size

        self.device = device
        self.verbose = verbose
        self.writer = writer
        self.log_file = log_file
        self.personal_name = personal_name
        self.weight_dir = weight_dir

        self.initial_lr = initial_lr
        self.dynamic_lr = dynamic_lr
        self.num_classes = num_classes
        self.mode = mode

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

            x_depth = None
            x_ir = None
            x_rgb = None

            label = label.to(self.device)
            self.optimizer.zero_grad()

            if self.mode == 'depth_ir_rgb':
                x_depth = x[0].to(self.device)
                x_ir = x[1].to(self.device)
                x_rgb = x[2].to(self.device)

                output = self.model(x_depth, x_ir, x_rgb)

            elif self.mode == 'depth_ir':
                x_depth = x[0].to(self.device)
                x_ir = x[1].to(self.device)

                output = self.model(x_depth, x_ir)

            elif self.mode == 'depth_rgb':
                x_depth = x[0].to(self.device)
                x_rgb = x[1].to(self.device)

                output = self.model(x_depth, x_rgb)

            elif self.mode == 'ir_rgb':
                x_ir = x[0].to(self.device)
                x_rgb = x[1].to(self.device)

                output = self.model(x_ir, x_rgb)

            else:
                raise NotImplementedError


            # label = label.to(self.device)
            # self.optimizer.zero_grad()
            # output = self.model(x_depth, x_ir, x_rgb)
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
                if step % 2 == 0:
                    tmp_loss = loss.item()
                    tmp_train_accuracy = train_accuracy
                else:

                    utilities.print_training_info(log_file=self.log_file, writer=self.writer, epoch=epoch, step=step,
                                                  running_loss=running_loss, train_running_accuracy=train_running_accuracy,
                                                  loss=(loss.item() + tmp_loss) / 2,
                                                  nof_steps=nof_steps,
                                                  train_accuracy=(train_accuracy + tmp_train_accuracy) / 2
                                                  )
        if self.dynamic_lr:
            utilities.adjust_learning_rate(self.initial_lr, epoch, self.optimizer)

    def val(self, epoch):

        self.model.eval()
        print("###VALIDATION###")
        validation_loss, running_validation_loss, validation_accuracy, validation_running_accuracy = 0., 0., 0., 0.
        nof_steps = len(self.validation_loader)

        with torch.no_grad():
            for step, data in enumerate(self.validation_loader):
                x, label = data

                label = label.to(self.device)
                self.optimizer.zero_grad()

                if self.mode == 'depth_ir_rgb':
                    x_depth = x[0].to(self.device)
                    x_ir = x[1].to(self.device)
                    x_rgb = x[2].to(self.device)

                    output = self.model(x_depth, x_ir, x_rgb)

                elif self.mode == 'depth_ir':
                    x_depth = x[0].to(self.device)
                    x_ir = x[1].to(self.device)

                    output = self.model(x_depth, x_ir)

                elif self.mode == 'depth_rgb':
                    x_depth = x[0].to(self.device)
                    x_rgb = x[1].to(self.device)

                    output = self.model(x_depth, x_rgb)

                elif self.mode == 'ir_rgb':
                    x_ir = x[0].to(self.device)
                    x_rgb = x[1].to(self.device)

                    output = self.model(x_ir, x_rgb)

                else:
                    raise NotImplementedError

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
                    i_label = label[i]
                    if self.batch_size > 1:
                        self.class_correct[i_label] += c[i].item()
                    else:
                        self.class_correct[i_label] += c.item()
                    self.class_total[i_label] += 1

                    if self.verbose:
                        if step % 2 == 0:
                            tmp_val_loss = validation_loss.item()
                            tmp_val_accuracy = validation_accuracy

                            utilities.print_validation_info(log_file=self.log_file, writer=self.writer, epoch=epoch,
                                                            step=step, running_loss=running_validation_loss,
                                                            validation_running_accuracy=validation_running_accuracy,
                                                            loss=(validation_loss.item() + tmp_val_loss) / 2,
                                                            nof_steps=nof_steps,
                                                            validation_accuracy=(validation_accuracy + tmp_val_accuracy) / 2
                                                            )

                state = {

                    'epoch': epoch,
                    'running_loss': validation_loss.item(),
                    'accuracy': validation_accuracy,
                    'avg_loss': running_validation_loss / (step + 1),
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'info': self.personal_name

                }

                if not os.path.exists('/projects/fabio/weights/gesture_recog_weights/{}'.format(self.weight_dir)):
                    os.makedirs('/projects/fabio/weights/gesture_recog_weights/{}'.format(self.weight_dir))

                if epoch == 0:
                    self.best_val_loss = validation_loss
                    self.best_val_acc = validation_accuracy
                    self.best_val_loss_state = state
                    self.best_val_acc_state = state

                    save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_acc_checkpoint_{}.pth.tar'.format(
                        self.weight_dir, self.personal_name)
                    torch.save(state, save_path)
                elif epoch % 10 == 0:
                    # salvo come checkpoint
                    save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_loss_checkpoint_{}.pth.tar'.format(
                        self.weight_dir, self.personal_name)
                    torch.save(self.best_val_loss_state, save_path)

                    save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_acc_checkpoint_{}'.format(
                        self.weight_dir, self.personal_name)
                    torch.save(self.best_val_acc_state, save_path)


                else:
                    if validation_loss <= self.best_val_loss:
                        self.best_val_loss = validation_loss
                        self.best_val_loss_state = state
                        save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_loss_checkpoint_{}.pth.tar'.format(
                            self.weight_dir, self.personal_name)

                        torch.save(state, save_path)

                    if validation_accuracy >= self.best_val_acc:
                        self.best_val_acc = validation_accuracy
                        self.best_val_acc_state = state
                        save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_acc_checkpoint_{}.pth.tar'.format(
                            self.weight_dir, self.personal_name)

                        torch.save(state, save_path)

    def __del__(self):
        self.writer.close()
        # prima di chiudere in trainig salvo i migliori stati
        save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_loss_checkpoint_{}.pth.tar'.format(
            self.weight_dir, self.personal_name)
        torch.save(self.best_val_loss_state, save_path)

        save_path = '/projects/fabio/weights/gesture_recog_weights/{}/best_val_acc_checkpoint_{}.pth.tar'.format(
            self.weight_dir, self.personal_name)
        torch.save(self.best_val_acc_state, save_path)

