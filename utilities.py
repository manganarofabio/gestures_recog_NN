import cv2
import numpy as np
from torchvision import transforms
import random
import torch as th
import json
import math
import gzip
import os

# file has to be already opened
def print_training_info(log_file, writer, epoch, step, running_loss, train_running_accuracy, loss, nof_steps,
                        train_accuracy):
    # print su terminal
    print('[epoch: {:d}, iter:  {:5d}] loss(avg): {:.3f}\t train_acc(avg): {:.3f}%'.format(
        epoch, step, running_loss / (step + 1), train_running_accuracy / (step + 1) * 100))
    # print su file
    log_file.write('[epoch: {:d}, iter:  {:5d}] loss(avg): {:.3f}\t train_acc(avg): {:.3f}%\n'.format(
        epoch, step, running_loss / (step + 1), train_running_accuracy / (step + 1) * 100))
    # print su tensorboard

    writer.add_scalar('data/train_loss', loss, epoch * nof_steps + step)
    writer.add_scalar('data/train_accuracy', train_accuracy, epoch * nof_steps + step)


def print_validation_info(log_file, writer, epoch, step, running_loss, validation_running_accuracy, loss, nof_steps,
                          validation_accuracy):
    # print su terminal
    print('[epoch: {:d}, iter:  {:5d}] val_loss(avg): {:.3f}\t val_acc(avg): {:.3f}%'.format(
        epoch, step, running_loss / (step + 1), validation_running_accuracy / (step + 1) * 100))
    # print su file
    log_file.write('[epoch: {:d}, iter:  {:5d}] val_loss(avg): {:.3f}\t val_acc(avg): {:.3f}%\n'.format(
        epoch, step, running_loss / (step + 1), validation_running_accuracy / (step + 1) * 100))
    # print su tensorboard

    writer.add_scalar('data/validation_loss', loss, epoch * nof_steps + step)
    writer.add_scalar('data/validation_accuracy', validation_accuracy, epoch * nof_steps + step)


def create_clip(list_of_img_of_same_record, n_frames, mode, resize_dim, DeepConvLstm=False, rgb=False):

    # rgb = True if mode == 'rgb' else False
    center_of_list = math.floor(len(list_of_img_of_same_record) / 2)
    crop_limit = math.floor(n_frames / 2)
    start = center_of_list - crop_limit
    end = center_of_list + crop_limit
    list_of_img_of_same_record_cropped = list_of_img_of_same_record[
                                         start: end + 1 if n_frames % 2 == 1 else end]

    list_img = []
    for img_path in list_of_img_of_same_record_cropped:
        if mode != 'depth_z':
            img = cv2.imread(img_path, 0 if not rgb else 1)
            img = cv2.resize(img, (resize_dim, resize_dim))
            if not rgb:
                img = np.expand_dims(img, axis=2)
        elif mode == 'depth_z':
            f = gzip.GzipFile(img_path, "r")
            img = np.loadtxt(f)
            img = cv2.resize(img, (resize_dim, resize_dim))
            img = np.expand_dims(img, axis=2)
        list_img.append(img)

    # concateno numpy array
    if not DeepConvLstm:
        clip = np.concatenate(list_img, axis=2)
        clip = clip.transpose([2, 0, 1])

    else:
        clip = np.asarray(list_img)
        clip = clip.transpose([0, 3, 1, 2])

    return clip


def create_result_json(out_file, result_dir, accuracy, conf_matrix, class_correct, class_total, list_of_pred,
                       list_of_gt):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data = {'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'class_correct': class_correct,
            'class_total': class_total,
            'list_of_pred': list_of_pred,
            'list_of_gt': list_of_gt
            }
    with open("{}/{}.json".format(result_dir, out_file[:-4]), 'w') as out_file:
        json.dump(data, out_file)
    out_file.close()

# full 675, full_no_fingers 145, mod 192
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


def extract_features_tracking_data(js):
    # len 96
    js = [
          js[0], js[1], js[2], js[3], js[4], js[5],  # palm
          js[176], js[177], js[178], js[179], js[180], js[181],  # thumb 0
          js[202], js[203], js[204], js[205], js[206], js[207],  # thumb 1
          js[228], js[229], js[230], js[231], js[232], js[233],  # thumb 2
          js[282], js[283], js[284], js[285], js[286], js[287],  # index 0
          js[308], js[309], js[310], js[311], js[312], js[313],  # index 1
          js[334], js[335], js[336], js[337], js[338], js[339],  # index 2
          js[388], js[389], js[390], js[391], js[392], js[393],  # middle 0
          js[414], js[415], js[416], js[417], js[418], js[419],  # middle 1
          js[440], js[441], js[442], js[443], js[444], js[445],  # middle 2
          js[494], js[495], js[496], js[497], js[498], js[499],  # ring 0
          js[520], js[521], js[522], js[523], js[524], js[525],  # ring 1
          js[546], js[547], js[548], js[549], js[550], js[551],  # ring 2
          js[600], js[601], js[602], js[603], js[604], js[605],  # pinky 0
          js[626], js[627], js[628], js[629], js[630], js[631],  # pinky 1
          js[652], js[653], js[654], js[655], js[656], js[657],  # pinky 2
          ]
    return js

def increase_input_size_per_record(record, coeff=1000):

    mask = [True, True, True, False, False, False] * 16


    list_inc_features_vel = []
    list_inc_features_acc = []

    for j in range(len(record)):
        if j == 0:
            list_inc_features_vel.append(list(np.ones(16 * 3) / coeff))
            list_inc_features_acc.append(list(np.ones(16 * 3) / coeff))
        if j == 1:
            list_inc_features_acc.append(list(np.ones(16 * 3) / coeff))
        if j >= 1:
            list_inc_features_vel.append(
                list(np.asarray(record[j])[mask] - np.asarray(record[j - 1])[mask])
            )
            if j >= 2:
                list_inc_features_acc.append(
                    list((np.asarray(list_inc_features_vel[j]) - list_inc_features_vel[j - 1]))
                )

    # accelerazione
    list_inc_features_vel = np.hstack((list_inc_features_vel, list_inc_features_acc))

    record = (np.hstack((record, list_inc_features_vel))).tolist()  #192 input size

    # print('input size increased')
    return record


def image_processing(img, type=1):
    pass


def normalization(img, mean, std, type=1):
    if type == 1:
        img = (img - mean) / std

    return img


def adjust_learning_rate(lr, epoch, optimizer):

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


##############
# TRANSFORMS #
##############

# def increase_input_size_tracking_data(list_data, acceleration=False, coeff=1000):
#
#
#     list_file_increased = []
#     mask = [True, True, True, False, False, False, False] * 16
#     for i, file in enumerate(files):
#         list_inc_features_vel = []
#         list_inc_features_acc = []
#         labels = []
#         if not classification:
#             labels = [frame[1] for frame in file]
#         for j in range(len(file)):
#             if j == 0:
#                 list_inc_features_vel.append(list(np.ones(16 * 3) / coeff))
#                 list_inc_features_acc.append(list(np.ones(16 * 3) / coeff))
#             if j == 1:
#                 list_inc_features_acc.append(list(np.ones(16 * 3) / coeff))
#             if j >= 1:
#                 list_inc_features_vel.append(list(np.asarray(file[j])[mask] - np.asarray(file[j - 1])[mask] if classification else
#                                              np.asarray(file[j][0])[mask] - np.asarray(file[j-1][0])[mask]))
#                 if j >= 2:
#                     list_inc_features_acc.append(list((np.asarray(list_inc_features_vel[j]) - list_inc_features_vel[j - 1])))
#
#         if acceleration:
#             list_inc_features_vel = np.hstack((list_inc_features_vel, list_inc_features_acc))
#
#         file = (np.hstack((file if classification else [frame[0] for frame in file], list_inc_features_vel))).tolist()
#         if not classification:
#             file = [(frame, labels[i]) for i, frame in enumerate(file)]
#         list_file_increased.append(file)
#
#     list_data_increased = []
#     for i in range(len(list_data)):
#         if len(list_data[0]) == 3: # if item mode == first_balanced
#             list_data_increased.append([list_file_increased[i], list_data[i][1], list_data[i][2]])
#         else:
#             list_data_increased.append((list_file_increased[i], list_data[i][1]))
#
#     list_data = list_data_increased
#     print('input size increased')
#     return list_data
#
# else:
#
#     list_inc_features_vel = []
#     list_inc_features_acc = []
#     for j in range(len(list_data)):
#         if j == 0:
#             list_inc_features_vel.append(list(np.ones(112) / coeff))
#             list_inc_features_acc.append(list(np.ones(112) / coeff))
#         if j == 1:
#             list_inc_features_acc.append(list(np.ones(112) / coeff))
#         if j >= 1:
#             list_inc_features_vel.append(list((np.asarray(list_data[j]) - list_data[j - 1])))
#             if j >= 2:
#                 list_inc_features_acc.append(
#                     list((np.asarray(list_inc_features_vel[j]) - list_inc_features_vel[j - 1])))
#
#     if acceleration:
#         list_inc_features_vel = np.hstack((list_inc_features_vel, list_inc_features_acc))
#
#     list_data = (np.hstack((list_data, list_inc_features_vel))).tolist()
#     # if not classification:
#     #     file = [(frame, labels[i]) for i, frame in enumerate(file)]
#     # list_file_increased.append(file)
#
#     return list_data


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
