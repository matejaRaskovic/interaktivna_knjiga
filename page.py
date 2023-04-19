import cv2
import numpy as np
from time import time

from page_card import PageCard
from page_slideshow import PageSlideshow

class Page:
    def __init__(self, video_path, overlays):
        self.video_path = video_path
        self.overlays = overlays
        self.finished_initial_video = False
        self.vid = cv2.VideoCapture(self.video_path)
        self.last_video_frame = None

    def get_frame(self):
        frame_to_display = self.last_video_frame
        if not self.finished_initial_video:
            ret, frame = self.vid.read()
            if not ret:
                self.finished_initial_video = True
                frame = self.last_video_frame
            else:
                self.last_video_frame = frame
            frame_to_display = frame

        if self.finished_initial_video:
            if len(self.overlays) > 0:
                left = self.overlays[0].get_frame()
                right = self.overlays[1].get_frame()
                frame_to_display = np.concatenate((left, right), axis=1)
                print(frame_to_display.shape)
        
        return frame_to_display
        

    def pass_control(self, control):
        if self.finished_initial_video:
            for overlay in self.overlays:
                overlay.pass_control(control)


def page_1():
    return Page("data\\page_videos\\strana1_salegendom.mp4", 
                [PageCard("data\\page_card_sequences\\levo_izvlacenje_kartice.mp4", "0"), 
                PageCard("data\\page_card_sequences\\desno_izvlacenje_kartice.mp4", "1")])

def page_2():
    return Page("data\\page_videos\\strana2_salegendom.mp4", 
                [PageCard("data\\page_card_sequences\\levo_izvlacenje_kartice.mp4", "0"), 
                PageSlideshow("data\\page_slideshows\\2_right", "2", "3")])

def page_3():
    return Page("data\\page_videos\\strana3_salegendom.mp4", 
                [PageCard("data\\page_card_sequences\\levo_izvlacenje_kartice.mp4", "0"), 
                PageCard("data\\page_card_sequences\\desno_izvlacenje_kartice.mp4", "1")])

def page_4():
    return Page("data\\page_videos\\strana4_salegendom.mp4", 
                [PageCard("data\\page_card_sequences\\levo_izvlacenje_kartice.mp4", "0"), 
                PageCard("data\\page_card_sequences\\desno_izvlacenje_kartice.mp4", "1")])

def page_0():
    return Page("data\\page_videos\\white_video.avi", [])