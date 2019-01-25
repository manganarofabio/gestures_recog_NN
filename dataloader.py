from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import numpy as np
import csv
import torch
import utilities
import PIL
import math

# csv
# img_path,session_id,gesture_id,record,mode,label,first

CROP_LIMIT = 15
RESIZE_DIM = 64


class GesturesDataset(Dataset):
    def __init__(self, csv_path, mode='ir', data_agumentation=False, normalization_type=-1, preprocessing_type=-1, transform=True):
        super().__init__()
        self.csv_path = csv_path
        self.mode = mode
        self.data_augmentation = data_agumentation
        self.normalization_type = normalization_type
        self.preprocessing_type = preprocessing_type
        if transform:
            self.transforms = transforms.Compose([transforms.ToTensor()])

        #inizializzaione dataset
        # apertura file csv
        with open('csv_dataset', 'r') as csv_in:
            reader = csv.reader(csv_in)
            self.list_of_rows_with_first_frame = [row for row in reader if row[4] == self.mode and row[6] == 'True']
            # list_of_rows_with_same_mode = list_of_rows_with_same_mode[1:]  # list_of_row[1]['mode']
            csv_in.seek(0)
            self.list_of_rows_with_same_mode = [row for row in reader if row[4] == self.mode]

        self.list_data = self.list_of_rows_with_first_frame


    def __getitem__(self, index):
        """
               Args:
                   index (int): Index
               Returns:
                   tuple: (image, target) where target is class_index of the target class.
               """
        #apertura file csv
        # with open('csv_dataset', 'r') as csv_in:
        #     reader = csv.reader(csv_in)
        #     list_of_rows_with_first_frame = [row for row in reader if row[4] == self.mode and row[6] is True]
        #     # list_of_rows_with_same_mode = list_of_rows_with_same_mode[1:]  # list_of_row[1]['mode']
        #     list_of_rows_with_same_mode = [row for row in reader if row[4] == self.mode]

        # self.list_data = list_of_rows_with_first_frame

        img_data = self.list_of_rows_with_first_frame[index]

        list_of_img_of_same_record = [img[0] for img in self.list_of_rows_with_same_mode
                                       if img[1] == img_data[1] # sessione
                                       and img[2] == img_data[2] # gesture
                                       and img[3] == img_data[3]]  # record

        # slice image
        #qui da [10:35]
        center_of_list = math.floor(len(list_of_img_of_same_record) / 2)
        start = center_of_list - CROP_LIMIT
        end = center_of_list + CROP_LIMIT
        list_of_img_of_same_record_cropped = list_of_img_of_same_record[start:end]

        list_img = []
        for img_path in list_of_img_of_same_record_cropped:
            img = cv2.imread(img_path, 0 if self.mode == 'ir' else 1)
            img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
            # img = (torch.from_numpy(img)).unsqueeze(0)
            img = np.expand_dims(img, axis=2)
            list_img.append(img)

        #concateno numpy array
        img_concat = np.concatenate(list_img, axis=2)

        if self.normalization_type is not None:
            utilities.normalization(img_concat, 1)

        # prima di convertire in tensore transpose
        # img_concat = img_concat.transpose([2, 0, 1])
        img_concat = self.transforms(img_concat)

        # return torch.from_numpy(img_concat)
        # target = torch.F(int(img_data[5]))
        target = torch.LongTensor(np.asarray([int(img_data[5])]))
        # target = np.empty((1),dtype=np.int32)
        # target[0] = int(img_data[5])
        return img_concat, target






            #list_img.append(self.transforms((torch.from_numpy(cv2.imread(img_path, flags=0))).unsqueeze(0)))

            # 1) apertura immagine -> np.array
            # 2) conversione da np in torch tensor
            # 3) unsqueeze(0) per aggiungere 3 dim
            # 4) applicato le transforms (prima to PIL, poi trans, poi Tensor)




        # tensor = torch.cat(list_img, dim=0)

        # print(tensor, img_data[5])

    def __len__(self):
        return len(self.list_data)

