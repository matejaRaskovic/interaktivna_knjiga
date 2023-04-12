import cv2
import serial

from time import time

from page import Page, page_0, page_1, page_2, page_3, page_4
from page_card import PageCard
from page_slideshow import PageSlideshow

default_book_size = [1920, 1280]
serialPort = serial.Serial(port='COM3', baudrate=9600, timeout=0)

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


if __name__ == '__main__':
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 1920, 1280)
    
    # Use this to enable full screen for projection
    # cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_page = page_0()
    while True:
        t_bef_get = time()
        frame = current_page.get_frame()
        # frame = cv2.resize(frame, (1440, 920))
        cv2.imshow("frame", current_page.get_frame())
        t_after_get = time()
        print(t_after_get)
        print(t_bef_get)
        wait_t = int(40 - (t_after_get - t_bef_get) * 1000)
        print(wait_t)
        wait_t = max(1, wait_t)
        # This needs to be replaced by recognition code
        wK = cv2.waitKey(wait_t)
        if wK & 0xFF == ord('q'):
            break
        elif wK & 0xFF == ord('1'):
            del current_page
            current_page = page_1()
        elif wK & 0xFF == ord('2'):
            del current_page
            current_page = page_2()
        elif wK & 0xFF == ord('3'):
            del current_page
            current_page = page_3()
        elif wK & 0xFF == ord('4'):
            del current_page
            current_page = page_4()
        elif wK & 0xFF == ord('0'):
            del current_page
            current_page = page_0()
        touchboard_input = get_touchboard_input()
        if touchboard_input != "":
            # current_page = pages[int(touchbourd_input)]
            current_page.pass_control(touchboard_input)