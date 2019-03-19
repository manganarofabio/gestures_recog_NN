import os, inspect, sys
import utils
import cv2
import numpy as np
import queue
import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options
import argparse


asd = 4

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = './leap_lib'

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

from leap_lib import Leap

parser = argparse.ArgumentParser(description='demo')

parser.add_argument('--n_frames', type=int, default=40)

gestures = ['g00', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12_test']


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q

    def onNewData(self, data):
        z_values = []
        gray_values = []
        for i in range(data.getNumPoints()):
            z_values.append(data.getZ(i))
            gray_values.append(data.getGrayValue(i))

        z_array = np.array(z_values)
        gray_array = np.array(gray_values)

        z_p = z_array.reshape(-1, data.width)
        gray_p = gray_array.reshape(-1, data.width)

        self.queue.put((z_p, gray_p))


def set_up(controller, rgb_cam=0):

    print("waiting for maps initialization...")
    while True:

        frame = controller.frame()
        image_l = frame.images[0]
        image_r = frame.images[1]

        if image_l.is_valid and image_r.is_valid:

            left_coordinates, left_coefficients = utils.convert_distortion_maps(image_l)
            right_coordinates, right_coefficients = utils.convert_distortion_maps(image_r)
            maps_initialized = True
            print('maps initialized')

            break
        else:
            print('\rinvalid leap motion frame', end="")

    # initialize video capture
    while True:
        cap = cv2.VideoCapture(rgb_cam)
        print(cap)
        if cap:
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0)
            # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            break
        else:
            print("\rerror rgb cam", end="")

    print("ready to go")
    return  cap


def run(controller, cam):

    # inizializzazione picoflexx
    q = queue.Queue()
    listener = MyListener(q)
    cam.registerDataListener(listener)
    # cam.startCapture()

    # setting up cameras
    cap = set_up(controller, rgb_cam=0)

    while True:

        print("press E to start")
        utils.draw_ui(text="press E to start")

        k = cv2.waitKey()
        if k == ord('e'):

            # RUNNING
            if not cap:
                print("error rgb cam")
                exit(-1)
            error_queue = False
            ready = False

            while True:
                try:
                    cam.startCapture()
                    print("picoflex ready")
                    break
                except RuntimeError:
                    print("error connection picoflex")

            while True:
                # print(frame_counter)

                if error_queue:
                    print(error_queue)
                    break

                frame = controller.frame()

                if utils.hand_is_valid(frame):
                    print('\nhand is valid -> ready to start')
                    ready = True
                    # print(self.listener.recording)
                    print("start gesture")
                    # print(self.listener.recording)
                else:
                    break

                # if ready:


        elif k == ord('s'):
            print("end")
            break









        # # end collection
        # print("end collection")
        # utils.save_session_info(session_id=session_counter - 1)

        # cam.stopCapture()






args = parser.parse_args()


def main():
    # PICOFLEXX

    parser1 = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser1)

    # parser1.add_argument("--seconds", type=int, default=15, help="duration to capture data")
    options = parser1.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()

    cam.setUseCase("MODE_5_45FPS_500")
    # print_camera_info(cam)
    # print("isConnected", cam.isConnected())
    # print("getFrameRate", cam.getFrameRate())

    # LEAP MOTION
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    run(controller, cam)


if __name__ == '__main__':
    main()

