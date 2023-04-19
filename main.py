import cv2
import serial

from time import time
import numpy as np

from page import Page, page_0, page_1, page_2, page_3, page_4
from page_card import PageCard
from page_slideshow import PageSlideshow

from dnn_training.train import Net

default_book_size = [1920, 1280]
# serialPort = serial.Serial(port='COM3', baudrate=9600, timeout=0)

initial_frame = np.zeros((1920, 1280, 3)).astype(np.uint8)

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


from threading import Thread, Lock
import cv2

lock = Lock()

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)

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
                self.stream.set(cv2.CAP_PROP_EXPOSURE, -4)
            else:
                self.stream.set(cv2.CAP_PROP_EXPOSURE, -9)
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class FrameGenerator:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        self.current_page = page_0()
        self.current_frame = initial_frame
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
    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("frame", 1920, 1280)

    cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame2", 1920, 1280)
    
    # Use this to enable full screen for projection
    # cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # focus = 255  # min: 0, max: 255, increment:5
    # cap.set(cv2.CAP_PROP_FOCUS, focus) 

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    i = 0

    video_getter = VideoGet(0).start()
    frame_gen = FrameGenerator().start()

    current_page = page_0()
    while True:
        t_bef_get = time()
        frame = frame_gen.get_frame()
        # frame = cv2.resize(frame, (1440, 920))
        # cv2.imshow("frame", frame)
        if i % 5 == 0:
            frame_cam = video_getter.frame
            # print(frame_cam)
            cv2.imshow("frame2", frame_cam)
        # i += 1
        # t_after_get = time()
        # print(t_after_get)
        # print(t_bef_get)
        # wait_t = int(40 - (t_after_get - t_bef_get) * 1000)
        # print(wait_t)
        # wait_t = max(1, wait_t)
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
        # touchboard_input = get_touchboard_input()
        # if touchboard_input != "":
        #     # current_page = pages[int(touchbourd_input)]
        #     current_page.pass_control(touchboard_input)

        i += 1

    video_getter.stop()
    frame_gen.stop()