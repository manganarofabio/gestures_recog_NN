import cv2
import numpy as np
import csv
import torch

with open('csv_dataset', 'r') as csv_in:
    reader = csv.reader(csv_in)
    list_of_rows_with_same_mode = [row for row in reader if row[4] == 'ir']
    list_of_rows_with_same_mode = list_of_rows_with_same_mode[1:]  # list_of_row[1]['mode']

img_data = list_of_rows_with_same_mode[100]

list_of_img_of_same_gesture = [img[0] for img in list_of_rows_with_same_mode
                               if img[1] == img_data[1]
                               and img[2] == img_data[2]
                               and img[3] == img_data[3]]

#slice image
list_img = []
for img_path in list_of_img_of_same_gesture:
    list_img.append(tran(torch.from_numpy(cv2.imread(img_path, flags=0))).unsqueeze(0))

img_concatenated = np.concatenate(list_img, axis=2)
tensor = torch.from_numpy(img_concatenated.transpose([2, 0, 1]))

print(tensor, img_data[5])






print('ok')

# get all images of gesture randomly chosen from index



