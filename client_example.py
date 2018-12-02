#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:01:54 2018

@author: ivanskya
"""
# pickle for loading python variables
import pickle

# pycurl for sending requests to the server
import pycurl

# io for retieving responses from pycurl
from io import BytesIO

# matplotlib for plotting
from matplotlib import pyplot as plt

# jsom for parsing the output
import json

def my_load(labels_in):
    
    data_out = {}
    
    for data_label in labels_in:
        with open(data_label + '.dat','rb') as infile:
            data_out[data_label] = pickle.load(infile)
    
    return(data_out)

def call_server(X_test_in):
    
    # write numpy array as "raw image"
    X_test_in.astype('int8').tofile('image.raw')
    
    # send the image to the server
    c = pycurl.Curl()
    
    storage = BytesIO()
    c.setopt(pycurl.WRITEFUNCTION, storage.write)
    
    c.setopt(c.URL, 'http://127.0.0.1:5000/upload')
    c.setopt(c.POST, 1)
    c.setopt(c.HTTPPOST, [("file", (c.FORM_FILE, "image.raw"))])
    c.setopt(pycurl.HTTPHEADER, ['enctype:multipart/form-data ; Content-Type:multipart/form-data'])
    c.perform()
    c.close()
    
    content = json.loads(storage.getvalue())
    
    return([content['0'],content['1'],content['2'],content['3'],content['4'],content['5'],content['6'],content['7'],content['8'],content['9']] )
    
# load the testing data
data_out = my_load(['X_train', 'y_train', 'X_test', 'y_test'])

X_train = data_out['X_train']
y_train = data_out['y_train']
X_test = data_out['X_test']
y_test = data_out['y_test']

# transform the data to [N,1,28,28]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# select one datapoint, plot it and send it to server
data_index = 100
plt.imshow(X_test[data_index,0,:,:])
print( call_server(X_test[data_index,:,:,:]) )
print(y_test[data_index])


