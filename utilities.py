import cv2
import numpy as np
from torchvision import transforms
import random
import torch as th
import json


def image_processing(img, type=1):
    pass


def normalization(img, type=1):
    if type == 1:
        img = (img - np.mean(img)) / np.std(img)

    return img

def adjust_learning_rate(epoch, optimizer):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def from_json_to_list(json_file):
    with open(json_file) as f:
        j = json.load(f)
        if j['frame'] != 'invalid':
            j_vector = [
                # palm
                j['frame']['right_hand']['palm_position'][0],
                j['frame']['right_hand']['palm_position'][1],
                j['frame']['right_hand']['palm_position'][2],
                j['frame']['right_hand']['palm_position'][3],
                j['frame']['right_hand']['palm_position'][4],
                j['frame']['right_hand']['palm_position'][5],
                j['frame']['right_hand']['palm_normal'][0],
                j['frame']['right_hand']['palm_normal'][1],
                j['frame']['right_hand']['palm_normal'][2],
                j['frame']['right_hand']['palm_normal'][3],
                j['frame']['right_hand']['palm_normal'][4],
                j['frame']['right_hand']['palm_normal'][5],
                j['frame']['right_hand']['palm_velocity'][0],
                j['frame']['right_hand']['palm_velocity'][1],
                j['frame']['right_hand']['palm_velocity'][2],
                j['frame']['right_hand']['palm_velocity'][3],
                j['frame']['right_hand']['palm_velocity'][4],
                j['frame']['right_hand']['palm_velocity'][5],
                j['frame']['right_hand']['palm_width'],
                j['frame']['right_hand']['pinch_strength'],
                j['frame']['right_hand']['grab_strength'],
                j['frame']['right_hand']['direction'][0],
                j['frame']['right_hand']['direction'][1],
                j['frame']['right_hand']['direction'][2],
                j['frame']['right_hand']['direction'][3],
                j['frame']['right_hand']['direction'][4],
                j['frame']['right_hand']['direction'][5],
                j['frame']['right_hand']['sphere_center'][0],
                j['frame']['right_hand']['sphere_center'][1],
                j['frame']['right_hand']['sphere_center'][2],
                j['frame']['right_hand']['sphere_center'][3],
                j['frame']['right_hand']['sphere_center'][4],
                j['frame']['right_hand']['sphere_center'][5],
                j['frame']['right_hand']['sphere_radius'],
                # wrist
                j['frame']['right_hand']['wrist_position'][0],
                j['frame']['right_hand']['wrist_position'][1],
                j['frame']['right_hand']['wrist_position'][2],
                j['frame']['right_hand']['wrist_position'][3],
                j['frame']['right_hand']['wrist_position'][4],
                j['frame']['right_hand']['wrist_position'][5],
                # pointables
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_0']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_0']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_0']['direction'][0],
                j['frame']['right_hand']['pointables']['p_0']['direction'][1],
                j['frame']['right_hand']['pointables']['p_0']['direction'][2],
                j['frame']['right_hand']['pointables']['p_0']['direction'][3],
                j['frame']['right_hand']['pointables']['p_0']['direction'][4],
                j['frame']['right_hand']['pointables']['p_0']['direction'][5],
                j['frame']['right_hand']['pointables']['p_0']['width'],
                j['frame']['right_hand']['pointables']['p_0']['length'],
                float(j['frame']['right_hand']['pointables']['p_0']['is_extended']),
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_1']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_1']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_1']['direction'][0],
                j['frame']['right_hand']['pointables']['p_1']['direction'][1],
                j['frame']['right_hand']['pointables']['p_1']['direction'][2],
                j['frame']['right_hand']['pointables']['p_1']['direction'][3],
                j['frame']['right_hand']['pointables']['p_1']['direction'][4],
                j['frame']['right_hand']['pointables']['p_1']['direction'][5],
                j['frame']['right_hand']['pointables']['p_1']['width'],
                j['frame']['right_hand']['pointables']['p_1']['length'],
                float(j['frame']['right_hand']['pointables']['p_1']['is_extended']),
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_2']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_2']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_2']['direction'][0],
                j['frame']['right_hand']['pointables']['p_2']['direction'][1],
                j['frame']['right_hand']['pointables']['p_2']['direction'][2],
                j['frame']['right_hand']['pointables']['p_2']['direction'][3],
                j['frame']['right_hand']['pointables']['p_2']['direction'][4],
                j['frame']['right_hand']['pointables']['p_2']['direction'][5],
                j['frame']['right_hand']['pointables']['p_2']['width'],
                j['frame']['right_hand']['pointables']['p_2']['length'],
                float(j['frame']['right_hand']['pointables']['p_2']['is_extended']),
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_3']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_3']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_3']['direction'][0],
                j['frame']['right_hand']['pointables']['p_3']['direction'][1],
                j['frame']['right_hand']['pointables']['p_3']['direction'][2],
                j['frame']['right_hand']['pointables']['p_3']['direction'][3],
                j['frame']['right_hand']['pointables']['p_3']['direction'][4],
                j['frame']['right_hand']['pointables']['p_3']['direction'][5],
                j['frame']['right_hand']['pointables']['p_3']['width'],
                j['frame']['right_hand']['pointables']['p_3']['length'],
                float(j['frame']['right_hand']['pointables']['p_3']['is_extended']),
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][0],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][1],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][2],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][3],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][4],
                j['frame']['right_hand']['pointables']['p_4']['tip_position'][5],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][0],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][1],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][2],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][3],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][4],
                j['frame']['right_hand']['pointables']['p_4']['tip_velocity'][5],
                j['frame']['right_hand']['pointables']['p_4']['direction'][0],
                j['frame']['right_hand']['pointables']['p_4']['direction'][1],
                j['frame']['right_hand']['pointables']['p_4']['direction'][2],
                j['frame']['right_hand']['pointables']['p_4']['direction'][3],
                j['frame']['right_hand']['pointables']['p_4']['direction'][4],
                j['frame']['right_hand']['pointables']['p_4']['direction'][5],
                j['frame']['right_hand']['pointables']['p_4']['width'],
                j['frame']['right_hand']['pointables']['p_4']['length'],
                float(j['frame']['right_hand']['pointables']['p_4']['is_extended']),
                # fingers
                # j['frame']['right_hand']['fingers']['thumb']['length'],
                # j['frame']['right_hand']['fingers']['thumb']['width'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['center'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['center'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['center'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['center'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['center'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['center'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['direction'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['direction'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['direction'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['direction'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['direction'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['direction'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['length'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['width'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['metacarpal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['center'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['center'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['center'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['center'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['center'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['center'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['direction'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['direction'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['direction'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['direction'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['direction'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['direction'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['length'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['width'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['proximal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['center'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['center'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['center'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['center'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['center'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['center'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['direction'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['direction'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['direction'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['direction'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['direction'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['direction'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['length'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['width'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['intermediate']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['center'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['center'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['center'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['center'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['center'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['center'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['direction'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['direction'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['direction'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['direction'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['direction'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['direction'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['length'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['width'],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['thumb']['bones']['distal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['length'],
                # j['frame']['right_hand']['fingers']['index']['width'],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['center'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['center'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['center'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['center'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['center'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['center'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['direction'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['direction'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['direction'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['direction'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['direction'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['direction'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['length'],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['width'],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['metacarpal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['center'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['center'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['center'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['center'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['center'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['center'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['direction'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['direction'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['direction'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['direction'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['direction'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['direction'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['length'],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['width'],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['proximal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['center'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['center'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['center'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['center'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['center'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['center'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['direction'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['direction'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['direction'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['direction'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['direction'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['direction'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['length'],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['width'],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['intermediate']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['center'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['center'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['center'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['center'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['center'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['center'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['direction'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['direction'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['direction'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['direction'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['direction'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['direction'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['length'],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['width'],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['index']['bones']['distal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['length'],
                # j['frame']['right_hand']['fingers']['middle']['width'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['center'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['center'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['center'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['center'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['center'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['center'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['direction'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['direction'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['direction'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['direction'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['direction'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['direction'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['length'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['width'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['metacarpal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['center'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['center'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['center'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['center'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['center'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['center'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['direction'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['direction'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['direction'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['direction'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['direction'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['direction'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['length'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['width'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['proximal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['center'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['center'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['center'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['center'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['center'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['center'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['direction'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['direction'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['direction'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['direction'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['direction'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['direction'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['length'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['width'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['intermediate']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['center'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['center'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['center'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['center'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['center'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['center'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['direction'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['direction'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['direction'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['direction'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['direction'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['direction'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['length'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['width'],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['middle']['bones']['distal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['length'],
                # j['frame']['right_hand']['fingers']['ring']['width'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['center'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['center'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['center'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['center'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['center'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['center'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['direction'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['direction'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['direction'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['direction'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['direction'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['direction'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['length'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['width'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['metacarpal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['center'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['center'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['center'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['center'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['center'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['center'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['direction'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['direction'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['direction'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['direction'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['direction'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['direction'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['length'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['width'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['proximal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['center'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['center'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['center'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['center'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['center'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['center'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['direction'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['direction'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['direction'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['direction'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['direction'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['direction'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['length'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['width'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['intermediate']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['center'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['center'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['center'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['center'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['center'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['center'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['direction'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['direction'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['direction'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['direction'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['direction'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['direction'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['length'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['width'],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['ring']['bones']['distal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['length'],
                # j['frame']['right_hand']['fingers']['pinky']['width'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['center'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['center'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['center'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['center'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['center'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['center'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['direction'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['direction'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['direction'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['direction'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['direction'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['direction'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['length'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['width'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['metacarpal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['center'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['center'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['center'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['center'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['center'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['center'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['direction'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['direction'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['direction'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['direction'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['direction'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['direction'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['length'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['width'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['proximal']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['center'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['center'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['center'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['center'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['center'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['center'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['direction'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['direction'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['direction'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['direction'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['direction'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['direction'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['length'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['width'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['intermediate']['next_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['center'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['center'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['center'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['center'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['center'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['center'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['direction'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['direction'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['direction'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['direction'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['direction'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['direction'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['length'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['width'],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['prev_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['prev_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['prev_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['prev_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['prev_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['prev_joint'][5],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['next_joint'][0],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['next_joint'][1],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['next_joint'][2],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['next_joint'][3],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['next_joint'][4],
                # j['frame']['right_hand']['fingers']['pinky']['bones']['distal']['next_joint'][5]
            ]
        else:
            j_vector = False

        return j_vector, j


##############
# TRANSFORMS #
##############


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        img = image

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(img, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        img = image

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]

        return img


class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.
        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p
        v : boolean
            whether to vertically flip w/ probability p
        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, x, y=None):
        # x = x.numpy()
        # if y is not None:
        #     y = y.numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    y = y.swapaxes(2, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    y = y.swapaxes(1, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return x.copy()
        else:
            return x.copy(), y.copy()
