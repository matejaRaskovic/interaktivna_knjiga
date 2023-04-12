class PageSlideshow:
  def __init__(self, images_path, left_control, right_control):
    self.images_path = images_path
    self.left_control = left_control
    self.right_control = right_control

  def get_frame(self):
    # implement preparing next frame
    return

  def pass_control(self, control):
    if control != self.left_control and control != self.right_control:
      return
