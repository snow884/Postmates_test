#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:12:17 2018

@author: ivanskya
"""

# TensorFlow and tf.keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.layers import Convolution2D, MaxPooling2D

#other
import pickle

def my_save(data_in):
    '''
    This is just a helper function for saving the data so that they dont have to
    be repeatedly downloaded.
    '''
    
    # save the data to the disk using pickle serilizer
    for data_label in list(data_in):
        with open(data_label + '.dat','wb') as outfile:
            pickle.dump(data_in[data_label], outfile)
    
def my_load(labels_in):
    '''
    This is another helper function for leading of the MNIST data from local disk
    '''
    
    data_out = {}
    
    # load data from disk using pickle serilizer
    for data_label in labels_in:
        with open(data_label + '.dat','rb') as infile:
            data_out[data_label] = pickle.load(infile)
    
    return(data_out)

def train_model():
    '''
    The main training heppens here
    '''
    
    # download the data from the server
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    #save the data locally
    my_save({'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test})
    
    #load the data
    data_out = my_load(['X_train', 'y_train', 'X_test', 'y_test'])
    
    # save the data into numpy arrays
    X_train = data_out['X_train']
    y_train = data_out['y_train']
    X_test = data_out['X_test']
    y_test = data_out['y_test']
    
    # Make sure the image values are between 0.0 and 1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # reshape the data as images with only one color channel
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    Y_train = y_train
    Y_test = y_test
    
    print(Y_train.shape)
    
    # NEURAL NETWORK STRUCTURE
    
    # input layer
    visible = Input(shape=(1,28,28))
    
    # first convolutional layer
    conv1 = Convolution2D(32, 28, 28, activation='relu', border_mode='same')(visible)
    pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv1)
    
    # second convolutional layer number 1
    conv2_1 = Convolution2D(64, 14, 14, activation='relu', border_mode='same')(pool1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv2_1)
    
    # second convolutional layer number 2
    conv2_2 = Convolution2D(64, 14, 14, activation='relu', border_mode='same')(pool1)
    pool2_2 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv2_2)
    
    # third convolutional layer number 1
    conv3_1 = Convolution2D(512, 7, 7, activation='relu', border_mode='same')(pool2_1)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv3_1)
    
    # third convolutional layer number 2
    conv3_2 = Convolution2D(512, 7, 7, activation='relu', border_mode='same')(pool2_2)
    pool3_2 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv3_2)
    
    # merge layer
    merge = concatenate([pool3_1, pool3_2])
    flat = Flatten()(merge)
    
    # first fully connected layer
    fc1 = Dense(1000, activation='relu')(flat)
    
    # secons fully connected layer
    fc2 = Dense(500, activation='relu')(fc1)
    
    # prediction output
    output = Dense(10, activation='softmax')(fc2)
    
    model = Model(inputs=visible, outputs=output)
    
    # summarize layers
    print(model.summary())
    
    # compile the model
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # fitting of the model
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=5, verbose=1)
    
    # model evaluation
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)
    
    #save model
    model.save('my_model.h5')

train_model()
    