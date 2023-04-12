import cv2
import numpy as np
from time import time

from page_card import PageCard

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
            t1 = time()
            ret, frame = self.vid.read()
            t2 = time()
            print("Video load frame: ")
            print(t2 - t1)
            if not ret:
                self.finished_initial_video = True
                frame = self.last_video_frame
            else:
                self.last_video_frame = frame
            frame_to_display = frame

        if self.finished_initial_video:
            i = 1
            for overlay in self.overlays:
                # if i == 1:
                #     overlay.trigger()
                overlay_frame = overlay.get_frame()
                print(overlay_frame.shape)
                t1 = time()
                # if i == 0:
                #     frame_to_display[400:, :960, :] = np.where(overlay_frame[400:, :960, :] == np.array([0, 0, 0]), frame_to_display[400:, :960, :], overlay_frame[400:, :960, :])
                # else:
                #     frame_to_display[400:, 960:, :] = np.where(overlay_frame[400:, 960:, :] == np.array([0, 0, 0]), frame_to_display[400:, 960:, :], overlay_frame[400:, 960:, :])
                if i == 0:
                    frame_to_display[400:, :960, :] = np.where(overlay_frame == np.array([0, 0, 0]), frame_to_display[400:, :960, :], overlay_frame)
                else:
                    frame_to_display[400:, 960:, :] = np.where(overlay_frame == np.array([0, 0, 0]), frame_to_display[400:, 960:, :], overlay_frame)
                i -= 1
                t2 = time()
                print(t2 - t1)
        
        return frame_to_display
        

    def pass_control(self, control):
        if self.finished_initial_video:
            for overlay in self.overlays:
                overlay.pass_control(control)


def page_1():
    return Page("data\\page_videos\\strana1_salegendom.mp4", [PageCard("data\\page_card_sequences\\legenda_desno_izvlacenje_v001_no_transp", "0"), PageCard("data\\page_card_sequences\\legenda_levo_izvlacenje_v001_no_transp", "1")])

def page_2():
    return Page("data\\page_videos\\strana2_salegendom.mp4", [PageCard("data\\page_card_sequences\\legenda_desno_izvlacenje_v001_no_transp", "0"), PageCard("data\\page_card_sequences\\legenda_levo_izvlacenje_v001_no_transp", "1")])

def page_3():
    return Page("data\\page_videos\\strana3_salegendom.mp4", [PageCard("data\\page_card_sequences\\legenda_desno_izvlacenje_v001_no_transp", "0"), PageCard("data\\page_card_sequences\\legenda_levo_izvlacenje_v001_no_transp", "1")])

def page_4():
    return Page("data\\page_videos\\strana4_salegendom.mp4", [PageCard("data\\page_card_sequences\\legenda_desno_izvlacenje_v001_no_transp", "0"), PageCard("data\\page_card_sequences\\legenda_levo_izvlacenje_v001_no_transp", "1")])

def page_0():
    return Page("data\\page_videos\\white_video.avi", [])