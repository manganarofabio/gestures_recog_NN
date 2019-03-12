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
parser.add_argument('--mode', type=str)
parser.add_argument('--result_file', type=str)
parser.add_argument('--exp_result_dir', type=str, default="exp_results")
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()


class GestureTestSet(Dataset):
    def __init__(self, csv_test_path, mode='depth_ir', normalization_type=-1,
                 preprocessing_type=-1, resize_dim=64, n_frames=30):
        super().__init__()
        self.csv_path = csv_test_path
        self.normalization_type = normalization_type
        self.preprocessing_type = preprocessing_type
        self.resize_dim = resize_dim
        self.n_frames = n_frames
        self.mode = mode

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

            # ToDO
            # # rgb
            # file = np.load("mean_std_rgb.npz")
            # self.mean_rgb = file['arr_0']
            # self.std_rgb = file['arr_1']
            # print('mean, std rgb loaded.'.format(self.mode))
            #
            #
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
            # # leap_motion_tracking_data
            # file = np.load("mean_std_leap_motion_tracking_data.npz")
            # self.mean_leap_motion_tracking_data = file['arr_0']
            # self.std_leap_motion_tracking_data = file['arr_1']
            # print('mean, std leap_motion_tracking_data loaded.'.format(self.mode))

            print('dataset_initialized')

    def __getitem__(self, item):

        # return all kind of data

        if self.mode == 'depth_ir':
            img_data = self.list_data[1][item]

            target = torch.LongTensor(np.asarray([int(img_data[5])]))

            list_of_img_of_same_record = [img[0] for img in self.list_of_rows_with_same_mode[1]
                                          if img[1] == img_data[1]  # sessione
                                          and img[2] == img_data[2]  # gesture
                                          and img[3] == img_data[3]]  # record

            clip = utilities.create_clip(list_of_img_of_same_record=list_of_img_of_same_record,
                                         n_frames=self.n_frames,
                                         mode=self.mode, resize_dim=self.resize_dim)
        else:
            raise NotImplementedError

        if self.normalization_type is not None:
            mean, std = None, None
            if self.mode == 'depth_ir':
                mean = self.mean_ir
                std = self.std_ir
            else:
                raise NotImplementedError
            clip = utilities.normalization(clip, mean, std, 1).astype(np.float32)

        return torch.Tensor(clip), target

    def __len__(self):
        return len(self.list_data[0])


class Eval:

    def __init__(self, test_loader, model, device, exp_result_dir, result_file='result.txt', n_classes=12, batch_size=1):

        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.result_file = result_file
        self.exp_result_dir = exp_result_dir
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.conf_matrix = [[0 for j in range(self.n_classes)] for i in range(self.n_classes)]
        # class accuracy
        self.class_correct = [0. for i in range(self.n_classes)]
        self.class_total = [0. for i in range(self.n_classes)]

    def eval(self):

        self.model.eval()

        for step, data in enumerate(tqdm.tqdm(self.test_loader)):

            x, label = data
            x, label = x.to(self.device), label.to(self.device)
            output = self.model(x)

            predicted = torch.argmax(output, dim=1)
            correct = label.squeeze(dim=1)

            #metrics

            accuracy = float((predicted == correct).sum().item()) / len(correct)
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

        print("Accuracy: {}%".format(accuracy*100))
        for i in range(self.n_classes):
            print('Accuracy of {} : {:.3f}%\n'.format(classes[i], 100 * self.class_correct[i] / self.class_total[i]))

            # save results
            utilities.create_result_json(out_file=self.result_file,
                                         result_dir = self.exp_result_dir,
                                         accuracy=accuracy,
                                         conf_matrix=self.conf_matrix,
                                         class_correct=self.class_correct,
                                         class_total=self.class_total
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

    rgb = False
    if args.mode == 'rgb':
        rgb = True

    in_channels = args.n_frames if not rgb else args.n_frames * 3
    n_classes = args.n_classes

    test_set = GestureTestSet(csv_test_path='csv_testset', n_frames=args.n_frames, resize_dim=args.input_size,
                              mode=args.mode)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size)

    if args.model == 'DenseNet':
        model = models.densenet161(pretrained=True)
        for params in model.parameters():
            params.required_grad = False
        model.features._modules['conv0'] = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(7, 7),
                                                     stride=(2, 2), padding=(3, 3))
        model.classifier = nn.Linear(in_features=2208, out_features=n_classes, bias=True)
        model = model.to(device)

        # carico pesi

        model.load_state_dict(torch.load(args.model_path, map_location=device)['state_dict'])

        evaluator = Eval(test_loader=test_loader, model=model, device=device, result_file=args.result_file,
                         exp_result_dir=args.exp_result_dir, batch_size=args.batch_size)

        evaluator.eval()






if __name__ == '__main__':
    main()
