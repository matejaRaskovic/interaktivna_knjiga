import cv2
import os
import numpy as np

folder_path = "C:\\Users\\Muzej\\interaktivna_knjiga\\data\\page_slideshows\\2_right"

for file in os.listdir(folder_path):
    image = cv2.imread(os.path.join(folder_path, file))
    image_resized = cv2.resize(image, (960, 1280))
    total_frame = np.zeros((1280, 1920, 3)).astype(np.uint8)

    total_frame[:, -960:, :] = image_resized
    out_path = os.path.join(folder_path, file[:-4] + "_appended.jpg")

    cv2.imwrite(out_path, total_frame)
