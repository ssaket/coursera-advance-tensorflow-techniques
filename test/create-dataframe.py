import os
import re
import glob
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def create_training_dataframe(files):
    regions_list = []
    for img_filename, fix_filename in files:
        img = load_img(img_filename, color_mode='rgb')
        fix_img = load_img(fix_filename, color_mode='grayscale')

        if img.size != fix_img.size:
            raise "Image file and fixation file sizes are not equal!"

        fix_img_arr = img_to_array(fix_img)[:, :, 0]
        fix_img_arr /= 255.

        salient_regions = np.argwhere(fix_img_arr > 0.9)
        non_salient_regions = np.argwhere(fix_img_arr < 0.1)
        salient_idx = salient_regions[np.random.choice(
            salient_regions.shape[0],
            10,
            replace=False,
        )]

        non_salient_idx = non_salient_regions[np.random.choice(
            non_salient_regions.shape[0],
            20,
            replace=False,
        )]

        for idx in salient_idx:
            regions_list.append((
                img_filename,
                fix_filename,
                img.size[0],
                img.size[1],
                idx[1], #row
                idx[0], #column
                1,
                'salient',
            ))
        for idx in non_salient_idx:
            regions_list.append((
                img_filename,
                fix_filename,
                img.size[0],
                img.size[1],
                idx[1], #row
                idx[0], #column
                0,
                'non_salient',
            ))

    df = pd.DataFrame(regions_list,
                      columns=[
                          'image_file_path',
                          'fixation_file_path',
                          'image_width',
                          'image_height',
                          'center_loc_x',
                          'center_loc_y',
                          'is_salient',
                          'label',
                      ])
    return df


if __name__ == '__main__':
    TORRONRO_ROOT = os.path.join('test', 'data', 'toronto', 'fixdens')
    TORRONTO_IMAGES_DIR = 'images'
    TORRONTO_SAL_DIR = 'output'
    OUTPUT_PATH = 'torronto_df.csv'
    SHUFFLE = False

    image_files = sorted(glob.glob(os.path.join(TORRONRO_ROOT,
                                                TORRONTO_IMAGES_DIR, '*.jpg'),
                                   recursive=True),
                         key=lambda x: float(re.findall("(\d+)", x)[0]))
    fixation_files = sorted(glob.glob(os.path.join(TORRONRO_ROOT,
                                                   TORRONTO_SAL_DIR, '*.jpg'),
                                      recursive=True),
                            key=lambda x: float(re.findall("(\d+)", x)[0]))

    if len(image_files) != len(fixation_files):
        raise "different number of images and fixation files"

    files = [(x, y) for x, y in zip(image_files, fixation_files)]
    df = create_training_dataframe(files)

    if SHUFFLE:
        df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH)