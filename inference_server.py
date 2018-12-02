#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:46:27 2018

@author: ivanskya
"""

#for working with files as PIDs
import os

# flask http server
from flask import Flask
from flask import jsonify
from flask import request
from flask import Response
from werkzeug import create_environ
from werkzeug.utils import secure_filename

# keras and tensorflow to be able to read the model
from keras.models import load_model
import tensorflow as tf

# numpy for rescaling
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/ivanskya/Documents/Python/Postmates_test/uploaded_files/'
ALLOWED_EXTENSIONS = set(['xml'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
environ = create_environ('/','127.0.0.1:5000')
response = Response()

SESSION_TYPE = 'filesystem'

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   '''
   '''
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename('./uploaded_files/uploaded_image'+str(os.getpid())+'.raw'))
      
      #return probabilities as json
      return jsonify(clasify_image())

def clasify_image():
    '''
    '''
   
    X_test = np.fromfile('./uploaded_files_uploaded_image'+str(os.getpid())+'.raw', dtype='uint8')
    os.path.exists('./uploaded_files_uploaded_image'+str(os.getpid())+'.raw')
    
    X_test = X_test.reshape(1, 1, 28, 28)
    
    X_test = X_test.astype('float32')
    X_test /= 255
    
    # this is a trick to get keras working in multiple threads
    global graph
    with graph.as_default():
        probs = model.predict(X_test)
    
    return (
            {'0':int(probs[0,0]*100), 
             '1':int(probs[0,1]*100), 
             '2':int(probs[0,2]*100), 
             '3':int(probs[0,3]*100), 
             '4':int(probs[0,4]*100), 
             '5':int(probs[0,5]*100), 
             '6':int(probs[0,6]*100), 
             '7':int(probs[0,7]*100), 
             '8':int(probs[0,8]*100), 
             '9':int(probs[0,9]*100)
             }
            )
    
model = load_model('my_model.h5')
graph = tf.get_default_graph()

if __name__ == '__main__':
    app.config['SESSION_TYPE'] = SESSION_TYPE

    app.run(debug=True)