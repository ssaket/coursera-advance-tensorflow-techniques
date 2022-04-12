import numpy as np
import tensorflow as tf
import os
import re
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

x_shape = (3, 42, 42, 3)
y_shape = ()  # A single item (not array)

AUTOTUNE = tf.data.AUTOTUNE


def get_files():
    TORRONRO_ROOT = os.path.join('test', 'data', 'toronto', 'fixdens')
    TORRONTO_IMAGES_DIR = 'images'
    TORRONTO_SAL_DIR = 'output'
    SHUFFLE = False

    image_files = sorted(glob(os.path.join(TORRONRO_ROOT, TORRONTO_IMAGES_DIR,
                                           '*.jpg'),
                              recursive=True),
                         key=lambda x: float(re.findall("(\d+)", x)[0]))
    fixation_files = sorted(glob(os.path.join(TORRONRO_ROOT, TORRONTO_SAL_DIR,
                                              '*.jpg'),
                                 recursive=True),
                            key=lambda x: float(re.findall("(\d+)", x)[0]))

    if len(image_files) != len(fixation_files):
        raise "different number of images and fixation files"

    for x, y in zip(image_files, fixation_files):
        yield (x, y)


def get_locations():
    for _, fnames in enumerate(get_files()):
        # Synthesize an image and a class label.
        img_filename, fix_filename = fnames
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
        salient_idx = np.hstack((salient_idx, np.ones(
            (salient_idx.shape[0])).reshape(-1, 1)))
        non_salient_idx = non_salient_regions[np.random.choice(
            non_salient_regions.shape[0],
            20,
            replace=False,
        )]
        non_salient_idx = np.hstack(
            (non_salient_idx, np.zeros(
                (non_salient_idx.shape[0])).reshape(-1, 1)))
        Y = np.vstack((salient_idx, non_salient_idx))
        np.random.shuffle(Y)
        yield img, Y


def generator_fn(target_sizes, patch_size):
    """Return a function that takes no arguments and returns a generator."""

    def generator():
        for _, img_loc in enumerate(get_locations()):
            img, locations = img_loc
            for loc in locations:
                # image_arr = img_to_array(img)
                is_salient = loc[2]
                w, h = patch_size[0], patch_size[1]
                images = []
                for target_size in target_sizes:
                    image_arr = img_to_array(img)
                    image_arr = tf.image.resize(
                        image_arr,
                        target_size,
                    )
                    # fig, ax = plt.subplots(1)
                    # ax.set_aspect('equal')
                    # ax.imshow(array_to_img(image_arr))

                    new_center = ((loc[1::-1] / img.size) * target_size)
                    # circ = Circle(new_center + patch_size, 5, color='red') # add because of padding
                    xmin, ymin = (new_center + patch_size / 2).astype(np.int32) # add pad-size /2 because of global padding

                    padding = tf.constant([[w, w], [h, h], [0, 0]])
                    image_arr = tf.pad(image_arr, padding, "CONSTANT")
                    # ax.imshow(array_to_img(image_arr))
                    # rect = Rectangle((xmin, ymin), w, h, alpha=0.4)
                    # ax.add_patch(circ)
                    # ax.add_patch(rect)
                    crop_arr = tf.image.crop_to_bounding_box(
                        image_arr, ymin, xmin, h, w).numpy()
                    # rect = Rectangle((xmin, ymin), w, h, alpha=0.4)
                    # ax.imshow(array_to_img(image_arr))
                    # plt.show()
                    images.append(crop_arr)
                yield np.asarray(images, dtype=np.float32), is_salient

    return generator


def augment(x, y):
    return x / 255., y


target_sizes = np.asarray([[400, 400], [250, 250], [150, 150]], dtype=np.int32)
patch_size = np.asarray([42, 42])
batch_size = 64
epochs = 5

# Create dataset.
gen = generator_fn(target_sizes=target_sizes, patch_size=patch_size)
dataset = tf.data.Dataset.from_generator(generator=gen,
                                         output_types=(np.float64, np.int32),
                                         output_shapes=(x_shape, y_shape))

def draw_patches(img_patches):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs[0].imshow(array_to_img(img_patches[0]))
    axs[0].set_title('Small')

    axs[1].imshow(array_to_img(img_patches[1]))
    axs[1].set_title('Medium')


    axs[2].imshow(array_to_img(img_patches[2]))
    axs[2].set_title('Large')
    plt.show()

for images, label in dataset.take(2):
    draw_patches(images)
    print('images.shape: ', images.shape)
    print('labels.shape: ', label.shape)

# Parallelize the augmentation.
dataset = dataset.map(
    augment,
    num_parallel_calls=AUTOTUNE,
    # Order does not matter.
    deterministic=False)

dataset = dataset.batch(batch_size, drop_remainder=True)
# Prefetch some batches.
dataset = dataset.prefetch(AUTOTUNE)

from mr_cnn import MrCNN

model = MrCNN()
# Prepare model.
# model = tf.keras.applications.VGG16(weights=None,
#                                     input_shape=x_shape,)

# from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Add
# from tensorflow.keras.models import Model

# def initialize_base_network(stream_name, patch_size=(42, 42, 3)):
#     input = Input(shape=(patch_size), name='patch_input')
#     x = Conv2D(96, 7, activation='relu', name='conv_1')(input)
#     x = MaxPool2D((2, 2))(x)
#     x = Conv2D(160, 3, activation='relu', name='conv_2')(x)
#     x = MaxPool2D((2, 2))(x)
#     x = Conv2D(288, 3, activation='relu', name='conv_3')(x)
#     x = MaxPool2D((2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(512, activation='relu', name='dense_stream')(x)
#     return Model(inputs=input, outputs=x, name=stream_name)

# input_image = Input(shape=(
#     3,
#     42,
#     42,
#     3,
# ), name='input_img')

# input_image_1 = input_image[:,0,:]
# stream_s1 = initialize_base_network(stream_name='stream_1')
# vec_output_a = stream_s1(input_image_1)

# input_image_2 = input_image[:,1,:]
# stream_s2 = initialize_base_network(stream_name='stream_2')
# vec_output_b = stream_s2(input_image_2)

# input_image_3 = input_image[:,2,:]
# stream_s3 = initialize_base_network(stream_name='stream_3')
# vec_output_c = stream_s3(input_image_3)

# added = Add(name='add_streams')([vec_output_a, vec_output_b, vec_output_c])
# x = Dense(512, activation='relu', name='dense_all')(added)
# output = Dense(1, activation='sigmoid', name='output')(x)

# model = Model(input_image, output, name='mr-cnn')

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
adam = tf.keras.optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-07,
                                amsgrad=False,
                                name='Adam')
model.compile(optimizer=adam, loss=bce, metrics=['accuracy'])

# Train. Do not specify batch size because the dataset takes care of that.
model.fit(dataset, epochs=epochs)