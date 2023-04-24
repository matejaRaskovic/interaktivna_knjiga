import cv2
import serial
import numpy as np
import os
import random

# from time import time
import numpy as np
import os
import random

from page import page_0, page_1, page_2, page_3, page_4

from dnn_training.train import Net
from torchvision import transforms
import torch
import torchvision

from threading import Thread, Lock

# Global variables
# serialPort = serial.Serial(port='COM3', baudrate=9600, timeout=0)
lock = Lock()
# net = Net()

recog_pos = [[],
             [[85, 135], [700, 760]],
             [[80, 130], [655, 715]],
             [[65, 115], [535, 595]],
             [[65, 115], [480, 540]]]

def get_touchboard_input():
    sArduino = serialPort.readline(1)
    formatedSArduino = format(sArduino)

    if formatedSArduino:
        if formatedSArduino == "b'0'":
            return "0"
        elif formatedSArduino == "b'1'":
            return "1"
        elif formatedSArduino == "b'2'":
            return "2"
        elif formatedSArduino == "b'3'":
            return "3"
        else:
            return ""
    
    return ""

transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])

def check_page(image, page_num):
    with lock:
        in_img = image[recog_pos[page_num][0][0]:recog_pos[page_num][0][1], recog_pos[page_num][1][0]:recog_pos[page_num][1][1]]
        print(in_img.shape)
        tnsr = transform(in_img)
        tnsr = tnsr.unsqueeze(0)
        with torch.no_grad():
            out = net(tnsr)

        page_val = out[0, 1].item()
        no_page_val = out[0, 0].item()

        if page_val < no_page_val * 3:
            return True
        else:
            return False

def find_page(image):
    for page_num in [1, 2, 3, 4]:
        net.load_state_dict(torch.load("dnn_training\\latest_standardized_model_" + str(page_num) + ".pt"))
        with lock:
            in_img = image[recog_pos[page_num][0][0]:recog_pos[page_num][0][1], recog_pos[page_num][1][0]:recog_pos[page_num][1][1]]
            print(in_img.shape)
            tnsr = transform(in_img)
            tnsr = tnsr.unsqueeze(0)
            with torch.no_grad():
                out = net(tnsr)

            page_val = out[0, 1].item()
            no_page_val = out[0, 0].item()

            if no_page_val < page_val * 3:
                return page_num

    return 0
        

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        focus = 255  # min: 0, max: 255, increment:5
        self.stream.set(cv2.CAP_PROP_FOCUS, focus) 

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

        self.lighted = False

    def set_lighted(self, lighted):
        with lock:
            self.lighted = lighted

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.lighted:
                self.stream.set(cv2.CAP_PROP_EXPOSURE, -5)
            else:
                self.stream.set(cv2.CAP_PROP_EXPOSURE, -10)
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class FrameGenerator:
    """
    Class that generates next frame to be projected.
    """

    def __init__(self):
        self.current_page = page_0()
        self.current_frame = np.zeros((1920, 1280, 3)).astype(np.uint8)
        self.stopped = False

        self.read = False

    def get_frame(self):
        with lock:
            self.read = True
        
        return self.current_frame

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            with lock:
                if self.read:
                    self.current_frame = self.current_page.get_frame()
                    self.read = False

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 1920, 1280)

    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame2", 60, 50)
    
    # Use this to enable full screen for projection
    # cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_position = [[65, 115], [535, 595]]

    i = 0

    # video_getter = VideoGet(0).start()
    frame_gen = FrameGenerator().start()

    temp_frame_to_test = cv2.imread("dnn_training\\data_whole_images_with_shifted_projection_1\\1\\9.jpg")

    shift_random = False

    while True:
        # t_bef_get = time()
        frame = frame_gen.get_frame()
        # video_getter.set_lighted(frame_gen.current_page.lighted)
        if shift_random:
            # Used to generate versified dataset
            if i % 50 == 0:
                shift_by = 50
                M = np.float32([
                    [1, 0, random.randint(-shift_by, shift_by)],
                    [0, 1, random.randint(-shift_by, shift_by)]
                ])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        cv2.imshow("frame", frame)
        if i % 3 == 0:
            # fldr_out = "dnn_training\\data_whole_images_with_shifted_projection_4\\1"
            # os.makedirs(fldr_out, exist_ok=True)
            # frame_cam = video_getter.frame
            # frame_cam_cut = frame_cam[current_position[0][0]:current_position[0][1], current_position[1][0]:current_position[1][1]]
            # cv2.imshow("frame2", frame_cam_cut)
            # cv2.imwrite(os.path.join(fldr_out, str(len(os.listdir(fldr_out))) + ".jpg"), frame_cam)
            # print(current_position)
            # check_page(frame_cam, 1)
        # This needs to be replaced by recognition code
        wK = cv2.waitKey(1)
        if wK & 0xFF == ord('q'):
            break
        elif wK & 0xFF == ord('1'):
            del frame_gen.current_page
            frame_gen.current_page = page_1()
        elif wK & 0xFF == ord('2'):
            del frame_gen.current_page
            frame_gen.current_page = page_2()
        elif wK & 0xFF == ord('3'):
            del frame_gen.current_page
            frame_gen.current_page = page_3()
        elif wK & 0xFF == ord('4'):
            del frame_gen.current_page
            frame_gen.current_page = page_4()
        elif wK & 0xFF == ord('0'):
            del frame_gen.current_page
            frame_gen.current_page = page_0()
        elif wK & 0xFF == ord('i'):
            frame_gen.current_page.pass_control("0")
        elif wK & 0xFF == ord('o'):
            frame_gen.current_page.pass_control("1")
        elif wK & 0xFF == ord('k'):
            frame_gen.current_page.pass_control("2")
        elif wK & 0xFF == ord('l'):
            frame_gen.current_page.pass_control("3")
        elif wK & 0xFF == ord('w'):
            current_position[0] = [pos - 5 for pos in current_position[0]]
        elif wK & 0xFF == ord('s'):
            current_position[0] = [pos + 5 for pos in current_position[0]]
        elif wK & 0xFF == ord('a'):
            current_position[1] = [pos - 5 for pos in current_position[1]]
        elif wK & 0xFF == ord('d'):
            current_position[1] = [pos + 5 for pos in current_position[1]]

        # touchboard_input = get_touchboard_input()
        # if touchboard_input != "":
        #     frame_gen.current_page.pass_control(touchboard_input)

        i += 1

    # video_getter.stop()
    frame_gen.stop()