import os
import cv2
import numpy as np
from time import time

import cv2

# black_img = np.zeros((1280, 1920, 3)).astype(np.uint8)
# black_img = np.zeros((880, 960, 3)).astype(np.uint8)

class PageCard:
  def __init__(self, video_path, trigger_control):
    self.trigger_control = trigger_control
    self.triggered = False
    self.vid = cv2.VideoCapture(video_path)
    ret, self.current_frame = self.vid.read()
    # print(self.current_frame.shape)
    self.last_img = None
    self.video_open = True

  def get_frame(self):
    # implement preparing next frame
    if not self.triggered:
      return self.current_frame
    if self.triggered and self.video_open:
      ret, self.current_frame = self.vid.read()
      if not ret:
        self.video_open = False
      else:
        self.last_img = self.current_frame

    return self.last_img

  def trigger(self):
    self.triggered = True

  def pass_control(self, control):
    if control == self.trigger_control:
      self.triggered = True