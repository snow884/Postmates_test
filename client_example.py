#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:01:54 2018

@author: ivanskya
"""

import pickle

import os

import pycurl

from io import BytesIO

import json

def my_load(labels_in):
    
    data_out = {}
    
    for data_label in labels_in:
        with open(data_label + '.dat','rb') as infile:
            data_out[data_label] = pickle.load(infile)
    
    return(data_out)

def call_server(X_test_in):
    
    X_test_in.astype('int8').tofile('image.raw')
    
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
    
data_out = my_load(['X_train', 'y_train', 'X_test', 'y_test'])

X_train = data_out['X_train']
y_train = data_out['y_train']
X_test = data_out['X_test']
y_test = data_out['y_test']

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print(y_test[1310])

print( call_server(X_test[1310,:,:,:]) )




