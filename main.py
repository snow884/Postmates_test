#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:12:17 2018

@author: ivanskya
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt


#other
import pickle

def my_save(data_in):
    
    for data_label in list(data_in):
        with open(data_label + '.dat','wb') as outfile:
            pickle.dump(data_in[data_label], outfile)
    
def my_load(labels_in):
    
    data_out = {}
    
    for data_label in labels_in:
        with open(data_label + '.dat','rb') as infile:
            data_out[data_label] = pickle.load(infile)
    
    return(data_out)

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#my_save({'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test})

data_out = my_load(['X_train', 'y_train', 'X_test', 'y_test'])

X_train = data_out['X_train']
y_train = data_out['y_train']
X_test = data_out['X_test']
y_test = data_out['y_test']

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print(y_train.shape)

print(y_train[:10])

#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)

Y_train = y_train
Y_test = y_test

print(Y_train.shape)

class_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(X_train[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[y_train[i]])
#    


# input layer
visible = Input(shape=(1,28,28))

# first feature extractor
conv1 = Convolution2D(32, 28, 28, activation='relu', border_mode='same')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv1)

# second feature extractor
conv2_1 = Convolution2D(64, 14, 14, activation='relu', border_mode='same')(pool1)
pool2_1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv2_1)

# second feature extractor
conv2_2 = Convolution2D(64, 14, 14, activation='relu', border_mode='same')(pool1)
pool2_2 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv2_2)

# second feature extractor
conv3_1 = Convolution2D(512, 7, 7, activation='relu', border_mode='same')(pool2_1)

# second feature extractor
conv3_2 = Convolution2D(512, 7, 7, activation='relu', border_mode='same')(pool2_2)

# merge feature extractors
merge = concatenate([pool2_1, pool2_2])

flat = Flatten()(merge)

# interpretation layer
fc1 = Dense(1000, activation='relu')(flat)

# interpretation layer
fc2 = Dense(500, activation='relu')(fc1)

# prediction output
output = Dense(10, activation='softmax')(fc2)

model = Model(inputs=visible, outputs=output)

#model = Sequential()
# 
#model.add(Convolution2D(32, 28, 28, activation='relu', border_mode='same', input_shape=(1,28,28)))
#model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
#
#model.add(Convolution2D(64, 14, 14, activation='relu', border_mode='same'))
#model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
#
#model.add(Convolution2D(512, 7, 7, activation='relu', border_mode='same'))
#
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#
#model.add(Dense(500, activation='relu'))
#
#model.add(Dense(10, activation='softmax'))

# summarize layers
print(model.summary())


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test accuracy:', test_acc)