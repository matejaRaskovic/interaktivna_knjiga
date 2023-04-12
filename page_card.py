import os
import cv2
import numpy as np
from time import time

from PIL import Image

# black_img = np.zeros((1280, 1920, 3)).astype(np.uint8)
black_img = np.zeros((880, 960, 3)).astype(np.uint8)

class PageCard:
  def __init__(self, images_path, trigger_control):
    self.images_path = images_path
    self.trigger_control = trigger_control
    self.triggered = False
    self.motion_images = []

    self.img_paths = [os.path.join(images_path, pth) for pth in os.listdir(images_path)]
    self.last_img = None
    # self.img_paths = [cv2.imread(os.path.join(images_path, pth)) for pth in os.listdir(images_path)]

  def get_frame(self):
    # implement preparing next frame
    if self.last_img is not None:
      return self.last_img
    t1 = time()
    if self.triggered and len(self.img_paths) > 0:
      # img_to_ret = np.asarray(Image.open(self.img_paths[0]))[:, :, :]
      img_to_ret = cv2.imread(self.img_paths[0])
      # img_to_ret = np.load(self.img_paths[0])
      print("img_to_ret.shape")
      print(img_to_ret.shape)
      # img_to_ret = self.img_paths[0]
      if len(self.img_paths) > 1:
        del self.img_paths[0]
      else:
        self.last_img = img_to_ret
    else:
      img_to_ret = black_img
    print("1: ")
    print(time() - t1)
    return img_to_ret

  def trigger(self):
    self.triggered = True

  def pass_control(self, control):
    if control == self.trigger_control:
      self.triggered = True