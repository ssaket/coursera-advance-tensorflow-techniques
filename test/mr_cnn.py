from requests import patch
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Resizing, Flatten, Cropping2D, Add
from tensorflow.keras.models import Model


class StreamBlock(Model):

    def __init__(self):
        super().__init__(name='StreamBlock')
        self.conv1 = Conv2D(96, 7, activation='relu')
        self.pool1 = MaxPool2D((2, 2))
        self.conv2 = Conv2D(160, 3, activation='relu')
        self.pool2 = MaxPool2D((2, 2))
        self.conv3 = Conv2D(288, 3, activation='relu')
        self.pool3 = MaxPool2D((2, 2))
        self.flatten = Flatten()
        self.dense = Dense(512, activation='relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense(x)
        
class MrCNN(Model):

    def __init__(self):
        super().__init__()
        self.resize_1 = Resizing(400, 400)
        self.resize_2 = Resizing(250, 250)
        self.resize_3 = Resizing(150, 150)
        self.stream_1 = StreamBlock()
        self.stream_2 = StreamBlock()
        self.stream_3 = StreamBlock()
        self.dense = Dense(512, activation='relu')
        self.add = Add()
        self.classifier = Dense(2, activation='sigmoid')

    def call(self, input_tensor):
        image = input_tensor[0]
        patch_center = input_tensor[1]
        
        s1 = self.resize_1(input_tensor)
        # s1 = Cropping2D()(s1)
        s1 = self.stream_1(s1)

        s2 = self.resize_2(input_tensor)
        # s2 = Cropping2D()(s2)
        s2 = self.stream_1(s2)

        s3 = self.resize_3(input_tensor)
        # s3 = Cropping2D()(s3)
        s3 = self.stream_1(s3)

        output = self.add([s1, s2, s3])
        output = self.dense(output)
        output = self.classifier(output)