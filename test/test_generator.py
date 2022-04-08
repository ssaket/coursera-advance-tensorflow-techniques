import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mr_cnn import MrCNN

def generate_custom_data(target_size):
    # gen_1 = datagen.flow_from_dataframe(df,
    #                                     x_col='image_file',
    #                                     y_col='is_salient',
    #                                     class_mode='binary',
    #                                     shuffle=False,
    #                                     target_size=target_size,
    #                                     seed=121)
    gen_2 = datagen.flow_from_dataframe(df,
                                        x_col='image_file',
                                        y_col=['is_salient', 'center_index'],
                                        class_mode='raw',
                                        shuffle=False,
                                        target_size=target_size,
                                        batch_size=64,
                                        seed=121)
    for batch in gen_2:
        print(batch)
        input_batch, output_batch = [batch[0], batch[1][:,1]], batch[1][:,0]
        yield input_batch, output_batch
       


if __name__ == '__main__':

    datagen = ImageDataGenerator(featurewise_center=False)

    df = pd.read_csv('torronto_df.csv')
    df.is_salient = df.is_salient.astype(str)
    df.center_index = df.center_index.apply(eval).apply(np.array)
    df.head()
    
    
    
    gen = generate_custom_data((500,500))
    mr_cnn = MrCNN()
    mr_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mr_cnn.fit_generator(gen, epochs=10)
    
    print(next(gen))