import cv2
import numpy as np


def image_processing(img, type=1):
    pass


def normalization(img, type=1):
    if type == 1:
        img = (img - np.mean(img)) / np.std(img)

    return img
