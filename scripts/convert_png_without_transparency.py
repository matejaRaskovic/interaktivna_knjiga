import cv2
import os
import numpy as np

folder_path = "C:\\Users\\Muzej\\interaktivna_knjiga\\data\\page_card_sequences\\legenda_levo_izvlacenje_v001"
folder_path_out = "C:\\Users\\Muzej\\interaktivna_knjiga\\data\\page_card_sequences\\legenda_levo_izvlacenje_v001_no_transp"

# folder_path = "C:\\Users\\Muzej\\interaktivna_knjiga\\data\\page_card_sequences\\legenda_desno_izvlacenje_v001"
# folder_path_out = "C:\\Users\\Muzej\\interaktivna_knjiga\\data\\page_card_sequences\\legenda_desno_izvlacenje_v001_no_transp"

for file in os.listdir(folder_path):
    image = cv2.imread(os.path.join(folder_path, file))
    os.makedirs(folder_path_out, exist_ok=True)
    out_path = os.path.join(folder_path_out, file[:-4] + ".png")
    print(out_path)

    cv2.imwrite(out_path, image[400:, :960, :])
    # np.savez(out_path[:-4], image[400:, :960, :])
    print(image[400:, :960, :].shape)

    # cv2.imwrite(out_path, image[400:, 960:, :])
    # np.savez(out_path[:-4], image[400:, 960:, :])
    # print(image[400:, 960:, :].shape)
