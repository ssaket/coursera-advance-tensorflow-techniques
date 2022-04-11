import tensorflow as tf
import numpy as np

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(224, 224, 3),
                 patch_size=(42,42),
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.patch_size = patch_size
        
        self.n = len(self.df)
        self.n_name = df[y_col['output']].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input_img(self, path, center, target_size):
    
        image = tf.keras.preprocessing.image.load_img(path)
        w, h = self.patch_size
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, target_size)
        new_center = ((center / image.size) * target_size).astype(np.int32)
        xmin, ymin= new_center + [self.patch_size[0], self.patch_size[1]]

        padding = tf.constant([[w, w],[h, h], [0, 0]])
        image_arr = tf.pad(image_arr, padding, "CONSTANT")

        f =ymin + h 
        g= xmin + w

        if(image_arr.shape[0] < f) or ((image_arr.shape[1] < g)):
            print(image_arr.shape, f, g, new_center)

        image_arr = tf.image.crop_to_bounding_box(image_arr, ymin, xmin, h, w).numpy()
        # image_arr = tf.image.resize(image_arr, self.patch_size).numpy()
        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return label
        # return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        center_loc_batch = batches[self.X_col['center']]
        
        name_batch = batches[self.y_col['output']]
        # type_batch = batches[self.y_col['type']]

        X1_batch = np.asarray([self.__get_input_img(x, y, (400,400)) for x, y in zip(path_batch, center_loc_batch)])
        X2_batch = np.asarray([self.__get_input_img(x, y, (250,250)) for x, y in zip(path_batch, center_loc_batch)])
        X3_batch = np.asarray([self.__get_input_img(x, y, (150,150)) for x, y in zip(path_batch, center_loc_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        # y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        # return X_batch, tuple([y0_batch, y1_batch])
        return [X1_batch, X2_batch, X3_batch], y0_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size