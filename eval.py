import numpy as np
import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
import csv
import utilities
import torch
import random
import tqdm
from models import CrossConvNet, Rnn
import math

parser = argparse.ArgumentParser(description='Evaluator')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1994, metavar='S',
                    help='random seed (default: 1994)')

parser.add_argument('--n_frames', type=int, default=40,
                    help='number of frames per input')
parser.add_argument('--input_size', type=int, default=224, # 227 alexnet, 64 lenet, 224 vgg16 and Resnet, denseNet
                    help='input size')
parser.add_argument('--n_classes', type=int, default=12,
                    help='number of frames per input')
parser.add_argument('--model', type=str)
parser.add_argument('--model_path', type=str)
# per CrossConvNet
parser.add_argument('--model_path_depth', type=str, default=None)
parser.add_argument('--model_path_ir', type=str, default=None)
parser.add_argument('--model_path_rgb', type=str, default=None)
parser.add_argument('--cross_mode', type=str, default=None)


parser.add_argument('--mode', type=str)
parser.add_argument('--gray_scale', action='store_true', default=False)
parser.add_argument('--res_name', type=str)
parser.add_argument('--exp_result_dir', type=str, default="exp_results")
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()


class GestureTestSet(Dataset):
    def __init__(self, csv_test_path, mode='depth_ir', normalization_type=-1,
                 preprocessing_type=-1, resize_dim=224, n_frames=30, gray_scale=False):
        super().__init__()
        self.csv_path = csv_test_path
        self.normalization_type = normalization_type
        self.preprocessing_type = preprocessing_type
        self.resize_dim = resize_dim
        self.n_frames = n_frames
        self.mode = mode
        self.gray_scale = gray_scale

        self.list_data = []
        self.list_of_rows_with_same_mode = []
        with open(self.csv_path, 'r') as csv_in:
            reader = csv.reader(csv_in)

            # depth_z
            self.list_of_rows_with_first_frame_z = [row for row in reader if row[4] == 'depth_z' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'depth_z'])
            csv_in.seek(0)

            # depth_ir
            self.list_of_rows_with_first_frame_ir = [row for row in reader if row[4] == 'depth_ir' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'depth_ir'])
            csv_in.seek(0)

            # rgb
            self.list_of_rows_with_first_frame_rgb = [row for row in reader if row[4] == 'rgb' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'rgb'])
            csv_in.seek(0)

            # L_raw
            self.list_of_rows_with_first_frame_l_raw = [row for row in reader if row[4] == 'L_raw' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'L_raw'])
            csv_in.seek(0)

            # L_undistorted
            self.list_of_rows_with_first_frame_l_undistorted = [row for row in reader if row[4] == 'L_undistorted' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'L_undistorted'])
            csv_in.seek(0)

            # R_raw
            self.list_of_rows_with_first_frame_r_raw = [row for row in reader if row[4] == 'R_raw' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'R_raw'])
            csv_in.seek(0)

            # R_undistorted
            self.list_of_rows_with_first_frame_r_undistorted = [row for row in reader if
                                                                row[4] == 'R_undistorted' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'R_undistorted'])
            csv_in.seek(0)

            # leap_tracking_data
            self.list_of_rows_with_first_frame_tracking_data = [row for row in reader if row[4] == 'leap_motion_tracking_data' and row[6] == 'True']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode.append([row for row in reader if row[4] == 'leap_motion_tracking_data'])
            csv_in.seek(0)

            # create list_data

            self.list_data += (([x for x in self.list_of_rows_with_first_frame_z],
                                [x for x in self.list_of_rows_with_first_frame_ir],
                                [x for x in self.list_of_rows_with_first_frame_rgb],
                                [x for x in self.list_of_rows_with_first_frame_l_raw],
                                [x for x in self.list_of_rows_with_first_frame_l_undistorted],
                                [x for x in self.list_of_rows_with_first_frame_r_raw],
                                [x for x in self.list_of_rows_with_first_frame_r_undistorted],
                                [x for x in self.list_of_rows_with_first_frame_tracking_data]))

            # loading norm e std per ogni mode

            # depth_z
            file = np.load("mean_std_depth_z.npz")
            self.mean_z = file['arr_0']
            self.std_z = file['arr_1']
            print('mean, std depth_z loaded.'.format(self.mode))

            # depth_ir
            file = np.load("mean_std_depth_ir.npz")
            self.mean_ir = file['arr_0']
            self.std_ir = file['arr_1']
            print('mean, std depth_ir loaded.'.format(self.mode))

            # rgb
            file = np.load("mean_std_rgb.npz")
            self.mean_rgb = file['arr_0']
            self.std_rgb = file['arr_1']
            print('mean, std rgb loaded.'.format(self.mode))

            # ToDO
            # # L_raw
            # file = np.load("mean_std_L_raw.npz")
            # self.mean_L_raw = file['arr_0']
            # self.std_L_raw = file['arr_1']
            # print('mean, std L_raw loaded.'.format(self.mode))
            #
            # # L_undistorted
            # file = np.load("mean_std_L_undistorted.npz")
            # self.mean_L_undistorted = file['arr_0']
            # self.std_L_undistorted = file['arr_1']
            # print('mean, std L_undistorted loaded.'.format(self.mode))
            #
            # # R_raw
            # file = np.load("mean_std_R_raw.npz")
            # self.mean_R_raw = file['arr_0']
            # self.std_R_raw = file['arr_1']
            # print('mean, std R_raw loaded.'.format(self.mode))
            #
            # # R_undistorted
            # file = np.load("mean_std_R_undistorted.npz")
            # self.mean_R_undistorted = file['arr_0']
            # self.std_R_undistorted = file['arr_1']
            # print('mean, std R_undistorted loaded.'.format(self.mode))
            #
            # leap_motion_tracking_data
            file = np.load("mean_std_leap_motion_tracking_data.npz")
            self.mean_leap_motion_tracking_data = file['arr_0']
            self.std_leap_motion_tracking_data = file['arr_1']
            print('mean, std leap_motion_tracking_data loaded.'.format(self.mode))

            print('dataset_initialized')

    def __getitem__(self, item):

        # return all kind of data
        img_data_depth, img_data_ir, img_data_rgb = None, None, None

        if self.mode == 'ir':
            img_data = self.list_data[1][item]
            list_same_mode = self.list_of_rows_with_same_mode[1]

        elif self.mode == 'rgb':
            img_data = self.list_data[2][item]
            list_same_mode = self.list_of_rows_with_same_mode[2]

        elif self.mode == 'depth':
            img_data = self.list_data[0][item]
            list_same_mode = self.list_of_rows_with_same_mode[0]

        elif self.mode == 'depth_ir_rgb' or self.mode == 'depth_ir' or self.mode == 'depth_rgb' or self.mode == 'ir_rgb':
            img_data_depth = self.list_data[0][item]
            list_same_mode_depth = self.list_of_rows_with_same_mode[0]
            img_data_ir = self.list_data[1][item]
            list_same_mode_ir = self.list_of_rows_with_same_mode[1]
            img_data_rgb = self.list_data[2][item]
            list_same_mode_rgb = self.list_of_rows_with_same_mode[2]

        elif self.mode == 'leap_motion_tracking_data':
            img_data_track = self.list_data[7][item]
            target = torch.LongTensor(np.asarray([int(img_data_track[5])]))

            list_same_mode_tracking_data = self.list_of_rows_with_same_mode[7]

            list_of_img_of_same_record = [img[0] for img in list_same_mode_tracking_data
                                          if img[1] == img_data_track[1]  # sessione
                                          and img[2] == img_data_track[2]  # gesture
                                          and img[3] == img_data_track[3]
                                          ]  # record

            center_of_list = math.floor(len(list_of_img_of_same_record) / 2)
            crop_limit = math.floor(self.n_frames / 2)
            start = center_of_list - crop_limit
            end = center_of_list + crop_limit
            list_of_img_of_same_record_cropped = list_of_img_of_same_record[
                                                 start: end + 1 if self.n_frames % 2 == 1 else end]

            img_data_track = []
            for js in list_of_img_of_same_record_cropped:
                f_js, j = utilities.from_json_to_list(js)
                # if self.tracking_data_mod:
                #     f_js = utilities.extract_features_tracking_data(f_js)
                img_data_track.append(f_js)

            # normalizzo
            # img_data_track = utilities.normalization(np.asarray(img_data_track), self.mean_leap_motion_tracking_data,
            #                         self.std_leap_motion_tracking_data)

            img_data_track = (img_data_track - self.mean_leap_motion_tracking_data) / self.std_leap_motion_tracking_data
            img_data_track = np.float32(img_data_track)

            return torch.tensor(img_data_track), target

        else:
            raise NotImplementedError

        if self.mode != 'depth' and self.mode != 'ir' and self.mode != 'rgb':

            target = None
            clip_depth = None
            clip_ir = None
            clip_rgb = None

            if img_data_depth is not None:
                list_of_img_of_same_record = [img[0] for img in list_same_mode_depth
                                              if img[1] == img_data_depth[1]  # sessione
                                              and img[2] == img_data_depth[2]  # gesture
                                              and img[3] == img_data_depth[3]]  # record

                clip_depth = utilities.create_clip(list_of_img_of_same_record=list_of_img_of_same_record,
                                             n_frames=self.n_frames,
                                             mode='depth_z', resize_dim=self.resize_dim)

                clip_depth = utilities.normalization(clip_depth, self.mean_z, self.std_z)

                target = torch.LongTensor(np.asarray([int(img_data_depth[5])]))

            if img_data_ir is not None:
                list_of_img_of_same_record = [img[0] for img in list_same_mode_ir
                                              if img[1] == img_data_depth[1]  # sessione
                                              and img[2] == img_data_depth[2]  # gesture
                                              and img[3] == img_data_depth[3]]  # record

                clip_ir = utilities.create_clip(list_of_img_of_same_record=list_of_img_of_same_record,
                                                   n_frames=self.n_frames,
                                                   mode='depth_ir', resize_dim=self.resize_dim)

                clip_ir = utilities.normalization(clip_ir, self.mean_ir, self.std_ir)

                if target is None:
                    target = torch.LongTensor(np.asarray([int(img_data_ir[5])]))

            if img_data_rgb is not None:
                list_of_img_of_same_record = [img[0] for img in list_same_mode_rgb
                                              if img[1] == img_data_depth[1]  # sessione
                                              and img[2] == img_data_depth[2]  # gesture
                                              and img[3] == img_data_depth[3]]  # record

                clip_rgb = utilities.create_clip(list_of_img_of_same_record=list_of_img_of_same_record,
                                                n_frames=self.n_frames,
                                                mode='rgb', resize_dim=self.resize_dim, rgb=not self.gray_scale)

                clip_rgb = utilities.normalization(clip_rgb, self.mean_rgb, self.std_rgb)

                if target is None:
                    target = torch.LongTensor(np.asarray([int(img_data_ir[5])]))


            # ritorno la tupla di clip
            item = None
            if self.mode == 'depth_ir_rgb':
                item = (np.float32(clip_depth), np.float32(clip_ir), np.float32(clip_rgb))

            elif self.mode == 'depth_ir':
                item = (np.float32(clip_depth), np.float32(clip_ir))

            elif self.mode == 'depth_rgb':
                item = (np.float32(clip_depth), np.float32(clip_rgb))

            elif self.mode == 'ir_rgb':
                item = (np.float32(clip_ir), np.float32(clip_rgb))

            else:
                raise NotImplementedError

            return item, target

        else:
            target = torch.LongTensor(np.asarray([int(img_data[5])]))

            list_of_img_of_same_record = [img[0] for img in list_same_mode
                                          if img[1] == img_data[1]  # sessione
                                          and img[2] == img_data[2]  # gesture
                                          and img[3] == img_data[3]]  # record

            clip = utilities.create_clip(list_of_img_of_same_record=list_of_img_of_same_record,
                                         n_frames=self.n_frames,
                                         mode=self.mode, resize_dim=self.resize_dim, rgb= not self.gray_scale)

            if self.normalization_type is not None:
                mean, std = None, None
                if self.mode == 'ir':
                    mean = self.mean_ir
                    std = self.std_ir
                elif self.mode == 'rgb':
                    mean = self.mean_rgb
                    std = self.std_rgb

                elif self.mode == 'depth':
                    mean = self.mean_z
                    std = self.std_z
                else:
                    raise NotImplementedError

                clip = utilities.normalization(clip, mean, std, 1).astype(np.float32)

            return torch.Tensor(clip), target

    def __len__(self):
        return len(self.list_data[0])


class Eval:

    def __init__(self, test_loader, model, mode, device, exp_result_dir, result_file='result.txt', n_classes=12, batch_size=1):

        self.test_loader = test_loader
        self.model = model
        self.mode = mode
        self.device = device
        self.result_file = result_file
        self.exp_result_dir = exp_result_dir
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.conf_matrix = [[0 for j in range(self.n_classes)] for i in range(self.n_classes)]
        # class accuracy
        self.class_correct = [0. for i in range(self.n_classes)]
        self.class_total = [0. for i in range(self.n_classes)]
        self.list_of_pred = []
        self.list_of_gt = []

        self.name_model = str(self.model)[:str(self.model).find('(')]

    def eval(self):

        accuracy, running_accuracy = 0., 0.
        self.model.eval()

        for step, data in enumerate(tqdm.tqdm(self.test_loader)):

            x, label = data
            label = label.to(self.device)

            if self.name_model == 'CrossConvNet':
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

            else:
                x = x.to(self.device)
                output = self.model(x)

            predicted = torch.argmax(output, dim=1)
            correct = label.squeeze(dim=1)

            for i in range(len(correct)):
                self.list_of_pred.append(predicted[i].item())
                self.list_of_gt.append(correct[i].item())

            #metrics

            accuracy = float((predicted == correct).sum().item()) / len(correct)
            running_accuracy += accuracy
            for i in range(len(correct)):
                self.conf_matrix[correct[i].item()][predicted[i].item()] += 1

            c = (predicted == correct).squeeze()
            for i in range(len(correct)):
                i_label = correct[i]
                if self.batch_size > 1:
                    self.class_correct[i_label] += c[i].item()
                else:
                    self.class_correct[i_label] += c.item()
                self.class_total[i_label] += 1

        # save results on file
        classes = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11']

        print("Accuracy: {}".format(running_accuracy/len(self.test_loader)))
        for i in range(self.n_classes):
            print('Accuracy of {} : {:.3f}%'.format(classes[i], 100 * self.class_correct[i] / self.class_total[i]))

            # save results
            utilities.create_result_json(out_file=self.result_file,
                                         result_dir=self.exp_result_dir,
                                         accuracy=running_accuracy/len(self.test_loader),
                                         conf_matrix=self.conf_matrix,
                                         class_correct=self.class_correct,
                                         class_total=self.class_total,
                                         list_of_pred=self.list_of_pred,
                                         list_of_gt=self.list_of_gt
                                         )


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

    rgb = True
    if args.gray_scale is True:
        rgb = False

    in_channels = args.n_frames if not rgb else args.n_frames * 3
    n_classes = args.n_classes

    test_set = GestureTestSet(csv_test_path='csv_testset', n_frames=args.n_frames, resize_dim=args.input_size,
                              mode=args.mode, gray_scale=args.gray_scale)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size)

    if args.model == 'DenseNet':
        model = models.densenet161(pretrained=True)
        # for params in model.parameters():
        #     params.requires_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=2208, out_features=n_classes, bias=True)
        model = model.to(device)

        # carico pesi

        model.load_state_dict(torch.load(args.model_path, map_location=device)['state_dict'])

    elif args.model == 'CrossConvNet':

        model_depth, model_ir, model_rgb = None, None, None

        if args.model_path_depth is not None:
            model_depth = models.densenet161(pretrained=True)
            # for params in model_depth.parameters():
            #     params.requires_grad = False
            model_depth.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                         stride=(2, 2), padding=(3, 3))
            model_depth.classifier = nn.Linear(in_features=2208, out_features=n_classes, bias=True)
            model_depth = model_depth.to(device)
            model_depth.eval()

            # carico pesi

            model_depth.load_state_dict(torch.load(args.model_path_depth, map_location=device)['state_dict'])

        if args.model_path_ir is not None:
            model_ir = models.densenet161(pretrained=True)
            # for params in model_ir.parameters():
            #     params.requires_grad = False
            model_ir.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                         stride=(2, 2), padding=(3, 3))
            model_ir.classifier = nn.Linear(in_features=2208, out_features=n_classes, bias=True)
            model_ir = model_ir.to(device)
            model_ir.eval()

            # carico pesi

            model_ir.load_state_dict(torch.load(args.model_path_ir, map_location=device)['state_dict'])

        if args.model_path_rgb is not None:
            model_rgb = models.densenet161(pretrained=True)
            # for params in model_rgb.parameters():
            #     params.requibed_grad = False
            model_rgb.features._modules['conv0'] = nn.Conv2d(in_channels=args.n_frames*3 if not rgb else
                                                             args.n_frames, out_channels=96, kernel_size=(7, 7),
                                                         stride=(2, 2), padding=(3, 3))
            model_rgb.classifier = nn.Linear(in_features=2208, out_features=n_classes, bias=True)
            model_rgb = model_rgb.to(device)
            model_rgb.eval()

            # carico pesi

            model_rgb.load_state_dict(torch.load(args.model_path_rgb, map_location=device)['state_dict'])

        model = CrossConvNet(args.n_classes, depth_model=model_depth, ir_model=model_ir, rgb_model=model_rgb,
                             mode=args.mode, cross_mode=args.cross_mode).to(device)

    elif args.model == 'LSTM':
        model = Rnn(rnn_type=args.model, input_size=args.input_size, hidden_size=256,
                    batch_size=1,
                    num_classes=12, num_layers=2,
                    final_layer='fc').to(device)

        model.load_state_dict(torch.load(args.model_path, map_location=device)['state_dict'])
        model.eval()

    result_file = "result_{}_{}_{}".format(args.model, args.mode, args.res_name)

    evaluator = Eval(test_loader=test_loader, model=model, mode=args.mode, device=device, result_file=result_file,
                     exp_result_dir=args.exp_result_dir, batch_size=args.batch_size)

    evaluator.eval()


if __name__ == '__main__':
    main()
