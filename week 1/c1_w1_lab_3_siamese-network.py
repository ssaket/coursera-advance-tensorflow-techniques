import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

def create_pairs(x, digit_indices):
	pairs = []
	labels = []
	n = min([len(digit_indices[d]) for d in range(10)]) - 1

	for d in range(10):
		for i in range(n):
			z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
			pairs += [[x[z1], x[z2]]]
			inc = random.randrange(1, 10)
			dn = (d + inc) % 10
			z1, z2 = digit_indices[d][i], digit_indices[dn][i]
			pairs += [[x[z1], x[z2]]]
			labels += [1, 0]

	return np.array(pairs), np.array(labels)

def create_pairs_on_set(images, labels):

	digit_indices = [np.where(labels == i)[0] for i in range(10)]
	pairs, y = create_pairs(images, digit_indices)
	y = y.astype('float31')

	return pairs, y

if __name__ == '__main__':
	# load the dataset
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# prepare train and test sets
	train_images = train_images.astype('float32')
	test_images = test_images.astype('float32')

	# normalize values
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	# create pairs on train and test sets
	tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
	ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)
