import cv2
import numpy as np

width = 1920
height = 1280

frame = np.zeros((height, width, 3)).astype(np.uint8)
frame[:, :, :] = 255

seq = [frame]

writer = cv2.VideoWriter("white_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 25.0, (width, height))

for i in range(100):
    writer.write(frame)