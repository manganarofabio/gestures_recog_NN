from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import numpy as np
import csv
import torch
import utilities
import PIL
import math
import gzip

# csv
# img_path,session_id,gesture_id,record,mode,label,first


class GesturesDatasetCrossMode(Dataset):
    def __init__(self, csv_path, train=False, mode='z_ir_rgb', n_frames=40, resize_dim=224, depth=False, ir=False,
                 rgb=False, ir_kind=False):
        super().__init__()

        self.csv_path = csv_path
        self.train = train
        self.mode = mode
        self.n_frames = n_frames
        self.resize_dim = resize_dim

        # item mode
        self.depth = depth
        self.ir = ir
        self.rgb = rgb
        self.ir_kind = ir_kind

        with open('csv_dataset', 'r') as csv_in:
            reader = csv.reader(csv_in)
            if self.depth:
                self.list_of_rows_with_first_frame_depth = [row for row in reader if
                                                            row[4] == 'depth_z' and row[6] == 'True']
                csv_in.seek(0)
            if self.rgb:
                self.list_of_rows_with_first_frame_rgb = [row for row in reader if
                                                          row[4] == 'rgb' and row[6] == 'True']
                csv_in.seek(0)
            if self.ir:
                self.list_of_rows_with_first_frame_ir = [row for row in reader if
                                                         row[4] == self.ir_kind and row[6] == 'True']
                csv_in.seek(0)

            # list_of_rows_with_same_mode = list_of_rows_with_same_mode[1:]  # list_of_row[1]['mode']
            if self.depth:
                self.list_of_rows_with_same_mode_depth = [row for row in reader if row[4] == 'depth_z']
                csv_in.seek(0)
            if self.rgb:
                self.list_of_rows_with_same_mode_rgb = [row for row in reader if row[4] == 'rgb']
                csv_in.seek(0)
            if self.ir:
                self.list_of_rows_with_same_mode_ir = [row for row in reader if row[4] == self.ir_kind]
                csv_in.seek(0)

        if self.rgb:
            len_dataset_per_session = len(set([int(x[1]) for x in self.list_of_rows_with_first_frame_rgb]))
        elif self.ir:
            len_dataset_per_session = len(set([int(x[1]) for x in self.list_of_rows_with_first_frame_ir]))
        else:
            len_dataset_per_session = len(set([int(x[1]) for x in self.list_of_rows_with_first_frame_depth]))

        self.list_data_depth, self.list_data_rgb, self.list_data_ir = [], [], []

        train_len = round(80 * len_dataset_per_session / 100)
        # divido il dataset in train e val per sessione
        for i in range(len_dataset_per_session):
            if self.train and i < train_len:
                if self.ir:
                    self.list_data_ir += [x for x in self.list_of_rows_with_first_frame_ir if (int(x[1])) == i]
                if self.rgb:
                    self.list_data_rgb += [x for x in self.list_of_rows_with_first_frame_rgb if (int(x[1])) == i]
                if self.depth:
                    self.list_data_depth += [x for x in self.list_of_rows_with_first_frame_depth if (int(x[1])) == i]

            elif not self.train and i >= train_len:
                if self.ir:
                    self.list_data_ir += [x for x in self.list_of_rows_with_first_frame_ir if (int(x[1])) == i]
                if self.rgb:
                    self.list_data_rgb += [x for x in self.list_of_rows_with_first_frame_rgb if (int(x[1])) == i]
                if self.depth:
                    self.list_data_depth += [x for x in self.list_of_rows_with_first_frame_depth if (int(x[1])) == i]



        # load mean and std
        # ir
        npzfile = np.load("mean_std_depth_ir.npz")
        self.mean_ir = npzfile['arr_0']
        self.std_ir = npzfile['arr_1']
        print('mean, std depth_ir loaded.')

        # depth_z
        npzfile = np.load("mean_std_depth_z.npz")
        self.mean_z = npzfile['arr_0']
        self.std_z = npzfile['arr_1']
        print('mean, std depth_z loaded.')

        # rgb
        npzfile = np.load("mean_std_rgb.npz")
        self.mean_rgb = npzfile['arr_0']
        self.std_rgb = npzfile['arr_1']
        print('mean, std depth_rgb loaded.')

        print('dataset_initialized')

    def __getitem__(self, index):
        """
               Args:
                   index (int): Index
               Returns:
                   tuple: (image, target) where target is class_index of the target class.
               """

        if self.ir:
            img_data_ir = self.list_data_ir[index]
        if self.rgb:
            img_data_rgb = self.list_data_rgb[index]
        if self.depth:
            img_data_depth = self.list_data_depth[index]

        target = None
        if self.rgb:
            target = torch.LongTensor(np.asarray([int(img_data_rgb[5])]))
        elif self.ir:
            target = torch.LongTensor(np.asarray([int(img_data_ir[5])]))
        else:
            target = torch.LongTensor(np.asarray([int(img_data_depth[5])]))

        if self.ir:
            list_of_img_of_same_record_ir = [img[0] for img in self.list_of_rows_with_same_mode_ir
                                           if img[1] == img_data_ir[1] # sessione
                                           and img[2] == img_data_ir[2] # gesture
                                           and img[3] == img_data_ir[3]]  # record

        if self.rgb:
            list_of_img_of_same_record_rgb = [img[0] for img in self.list_of_rows_with_same_mode_rgb
                                                   if img[1] == img_data_rgb[1]  # sessione
                                                   and img[2] == img_data_rgb[2]  # gesture
                                                   and img[3] == img_data_rgb[3]]  # record
        if self.depth:
            list_of_img_of_same_record_depth = [img[0] for img in self.list_of_rows_with_same_mode_depth
                                              if img[1] == img_data_depth[1]  # sessione
                                              and img[2] == img_data_depth[2]  # gesture
                                              and img[3] == img_data_depth[3]]  # record

        # slice image se non facciamo lstm variabile

        img_depth_concat, img_ir_concat, img_rgb_concat = None, None, None

        # depth_ir
        clip_depth = None
        clip_ir = None
        clip_rgb = None
        if self.ir:
            # center_of_list = math.floor(len(list_of_img_of_same_record_ir) / 2)
            # crop_limit = math.floor(self.n_frames / 2)
            # start = center_of_list - crop_limit
            # end = center_of_list + crop_limit
            # list_of_img_of_same_record_cropped_ir = list_of_img_of_same_record_ir[
            #                                      start: end + 1 if self.number_of_frames % 2 == 1 else end]

            clip_ir = utilities.create_clip(list_of_img_of_same_record_ir, self.n_frames, 'depth_ir', self.resize_dim)

            # list_img_ir = []
            # for img_path in list_of_img_of_same_record_cropped_ir:
            #
            #     img = cv2.imread(img_path, 0)
            #     img = cv2.resize(img, (self.resize_dim, self.resize_dim))
            #     img = np.expand_dims(img, axis=2)
            #     list_img_ir.append(img)
            #
            # # concateno numpy array
            # img_ir_concat = np.concatenate(list_img_ir, axis=2)

            clip_ir = utilities.normalization(clip_ir, self.mean_ir, self.std_ir)

        # prima di convertire in tensore transpose
        # img_concat = img_concat.transpose([2, 0, 1])

        # rgb
        if self.rgb:
            # list_of_img_of_same_record_cropped_rgb = []
            # center_of_list = math.floor(len(list_of_img_of_same_record_rgb) / 2)
            # crop_limit = math.floor(self.number_of_frames / 2)
            # start = center_of_list - crop_limit
            # end = center_of_list + crop_limit
            # list_of_img_of_same_record_cropped_rgb = list_of_img_of_same_record_rgb[
            #                                      start: end + 1 if self.number_of_frames % 2 == 1 else end]
            #
            # list_img_rgb = []
            # for img_path in list_of_img_of_same_record_cropped_rgb:
            #     img = cv2.imread(img_path, 1)
            #     img = cv2.resize(img, (self.resize_dim, self.resize_dim))
            #     list_img_rgb.append(img)
            #
            # # concateno numpy array
            # img_rgb_concat = np.concatenate(list_img_rgb, axis=2)
            #
            # utilities.normalization(img_rgb_concat, 1)

            clip_rgb = utilities.create_clip(list_of_img_of_same_record_rgb, n_frames=self.n_frames, mode='rgb',
                                             resize_dim=self.resize_dim)
            clip_rgb = utilities.normalization(clip_rgb, self.mean_rgb, self.std_rgb)


        if self.depth:

            # list_of_img_of_same_record_cropped_depth = []
            # center_of_list = math.floor(len(list_of_img_of_same_record_depth) / 2)
            # crop_limit = math.floor(self.number_of_frames / 2)
            # start = center_of_list - crop_limit
            # end = center_of_list + crop_limit
            # list_of_img_of_same_record_cropped_depth = list_of_img_of_same_record_depth[
            #                                          start: end + 1 if self.number_of_frames % 2 == 1 else end]
            #
            # list_img_depth = []
            # for img_path in list_of_img_of_same_record_cropped_depth:
            #         f = gzip.GzipFile(img_path, "r")
            #         img = np.loadtxt(f)
            #         img = cv2.resize(img, (self.resize_dim, self.resize_dim))
            #         img = np.expand_dims(img, axis=2)
            #         list_img_depth.append(img)
            #
            # img_depth_concat = np.concatenate(list_img_depth, axis=2)
            # img_concat = img_depth_concat.astype(np.float32)
            # utilities.normalization(img_depth_concat, 1)

            clip_depth = utilities.create_clip(list_of_img_of_same_record_depth, n_frames=self.n_frames, mode='depth_z',
                                               resize_dim=self.resize_dim)

            clip_depth = utilities.normalization(clip_depth, self.mean_z, self.std_z)

        # if self.rgb:
        #     img_rgb_concat = torch.Tensor(img_rgb_concat.transpose(2, 1, 0))
        # if self.ir:
        #     img_ir_concat = torch.Tensor(img_ir_concat.transpose(2, 1, 0))
        # if self.depth:
        #     img_depth_concat = torch.Tensor(img_depth_concat.transpose([2, 0, 1]))

        # return (img_depth_concat, img_ir_concat, img_rgb_concat), target
        item = None
        if self.depth and self.ir and self.rgb:
            item = (np.float32(clip_depth), np.float32(clip_ir), np.float32(clip_rgb))

        elif self.depth and self.ir:
            item = (np.float32(clip_depth), np.float32(clip_ir))

        elif self.depth and self.rgb:
            item = (np.float32(clip_depth), np.float32(clip_rgb))

        elif self.ir and self.rgb:
            item = (np.float32(clip_ir), np.float32(clip_rgb))

        else:
            raise NotImplementedError

        return item, target

    def __len__(self):
        if self.depth:
            return len(self.list_data_depth)
        elif self.ir:
            return len(self.list_data_ir)
        else:
            return len(self.list_data_rgb)

