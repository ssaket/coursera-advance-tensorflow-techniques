from matplotlib import pyplot as plt
import os
import re
import glob
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd


def create_training_dataframe(files):
    df_list = []
    for imgf, fixf in files:
        img = load_img(imgf, color_mode='rgb')
        fix_img = load_img(fixf, color_mode='grayscale')
        if img.size != fix_img.size:
            raise "Image sizes are not equal"
        img_arr = img_to_array(img)
        fix_img_arr = img_to_array(fix_img)[:,:,0]
        fix_img_arr /= 255.
        bright_regions = np.argwhere(fix_img_arr>0.9)
        dark_regions = np.argwhere(fix_img_arr<0.1)
        sample_bright_idx = bright_regions[np.random.choice(bright_regions.shape[0], 10, replace=False)]
        sample_dark_idx = dark_regions[np.random.choice(dark_regions.shape[0], 20, replace=False)]
        for idx in sample_bright_idx:
            # left, top = (idx[0] - patch_size[0]//2), (idx[1] - patch_size[1]//2)
            # right, bottom = (idx[0] + patch_size[0]//2), (idx[1] + patch_size[1]//2)
            # if left < 0 or top < 0:
            #     cv2.copyMakeBorder(img_arr, abs(left), abs(top), 0, 0, cv2.BORDER_REFLECT)
            # cropped_img = img.crop((left, top, right, bottom))
            # plt.imshow(array_to_img(cropped_img))
            df_list.append((imgf, fixf, img.size, idx.tolist(), 1))
        for idx in sample_dark_idx:
            df_list.append((imgf, fixf, img.size, idx.tolist(), 0))
            
    df = pd.DataFrame(df_list, columns =['image_file', 'fixation_file', 'image_size' ,'center_index', 'is_salient'])
    df.to_csv('torronto_df.csv')



if __name__ == '__main__':
    TORRONRO_ROOT = 'data/toronto/fixdens/'
    TORRONTO_IMAGES_DIR = 'images'
    TORRONTO_SAL_DIR = 'output'

    image_files = sorted(glob.glob(os.path.join(TORRONRO_ROOT,
                                                TORRONTO_IMAGES_DIR, '*.jpg'),
                                   recursive=True),
                         key=lambda x: float(re.findall("(\d+)", x)[0]))
    fixation_files = sorted(glob.glob(os.path.join(TORRONRO_ROOT,
                                                   TORRONTO_SAL_DIR, '*.jpg'),
                                      recursive=True),
                            key=lambda x: float(re.findall("(\d+)", x)[0]))

    files = [(x, y) for x, y in zip(image_files, fixation_files)]
    create_training_dataframe(files)