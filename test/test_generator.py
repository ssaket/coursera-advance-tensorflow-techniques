import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mr_cnn import MrCNN
from custom_data import CustomDataGen

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



# def generate_custom_data(target_size):
#     # gen_1 = datagen.flow_from_dataframe(df,
#     #                                     x_col='image_file',
#     #                                     y_col='is_salient',
#     #                                     class_mode='binary',
#     #                                     shuffle=False,
#     #                                     target_size=target_size,
#     #                                     seed=121)
#     gen_2 = datagen.flow_from_dataframe(df,
#                                         x_col='image_file',
#                                         y_col=['is_salient', 'center_index'],
#                                         class_mode='raw',
#                                         shuffle=False,
#                                         target_size=target_size,
#                                         batch_size=64,
#                                         seed=121)
#     for batch in gen_2:
#         print(batch)
#         input_batch, output_batch = [batch[0], batch[1][:,1]], batch[1][:,0]
#         yield input_batch, output_batch

if __name__ == '__main__':

    datagen = ImageDataGenerator(featurewise_center=False)

    df = pd.read_csv('torronto_df.csv')
    df.is_salient = df.is_salient.astype(str)
    df.center_index = df.center_index.apply(eval).apply(np.array)
    df.head()
    batch_size = 8
    target_size = (400,400)

    traingen = CustomDataGen(df,
                             X_col={
                                 'path': 'image_file',
                                 'center': 'center_index'
                             },
                             y_col={
                                 'output': 'is_salient'
                             },
                             batch_size=batch_size,
                             input_size=target_size)

    # valgen = CustomDataGen(val_df,
    #                        X_col={
    #                            'path': 'filename',
    #                            'bbox': 'region_shape_attributes'
    #                        },
    #                        y_col={
    #                            'name': 'name',
    #                            'type': 'type'
    #                        },
    #                        batch_size=batch_siz,
    #                        input_size=target_size)
    # x, y = traingen[0]
    mr_cnn = MrCNN()
    mr_cnn.build(input_shape=[(1,42,42,3),(1,42,42,3),(1,42,42,3)])
    mr_cnn.summary()
    mr_cnn.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    mr_cnn.fit(traingen, epochs=5, steps_per_epoch=2)

    print(next(gen))