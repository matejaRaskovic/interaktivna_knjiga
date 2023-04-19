import cv2
import numpy as np
import os

class PageSlideshow:
  def __init__(self, images_path, left_control, right_control):
    self.images_path = images_path
    self.left_control = left_control
    self.right_control = right_control
    self.img_pths = [os.path.join(images_path, pth) for pth in os.listdir(images_path)]
    self.images = [cv2.resize(cv2.imread(pth), (960, 1280)) for pth in self.img_pths]
    self.images = [img.astype(float) / 255.0 for img in self.images]

    self.current_ind = 0
    self.prev_ind = 0
    self.alphas = []

  def get_frame(self):
    if len(self.alphas) > 0:
      alpha = self.alphas[0]
      del self.alphas[0]
      next_frame = self.images[self.current_ind] * alpha + self.images[self.prev_ind] * (1 - alpha)
    else:
      next_frame = self.images[self.current_ind]

    # implement preparing next frame
    return (next_frame*255).astype(np.uint8)

  def pass_control(self, control):
    if control != self.left_control and control != self.right_control:
      return
    if control == self.left_control and self.current_ind > 0:
      self.prev_ind = self.current_ind
      self.current_ind -= 1
      self.alphas = list(np.linspace(0, 1, 4))
    if control == self.right_control and self.current_ind < len(self.images) - 1:
      self.prev_ind = self.current_ind
      self.current_ind += 1
      self.alphas = list(np.linspace(0, 1, 4))

