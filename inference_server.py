#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:46:27 2018

@author: ivanskya
"""
import os
from flask import Flask
from flask import jsonify
from flask import request
from flask import Response
from flask import redirect
from flask import url_for
from flask import flash
from flask import session as Session

from werkzeug import create_environ
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/ivanskya/Documents/Python/Postmates_test/uploaded_files/'
ALLOWED_EXTENSIONS = set(['xml'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
environ = create_environ('/','127.0.0.1:5000')
response = Response()

SESSION_TYPE = 'filesystem'

model = load_model('my_model.h5')

def allowed_file(filename):
    '''
    '''

    return '.' in filename and \
       filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename('uploaded_files//uploaded_image.raw'))
      
      return jsonify(clasify_image())

def clasify_image():
    
    
    
    probs = model.predict(X_test)
    
    return (
            {'0':probs[0], 
             '1':probs[1], 
             '2':probs[2], 
             '3':probs[3], 
             '4':probs[4], 
             '5':probs[5], 
             '6':probs[6], 
             '7':probs[7], 
             '8':probs[8], 
             '9':probs[9]
             }
            )
    
    
if __name__ == '__main__':
    app.config['SESSION_TYPE'] = SESSION_TYPE

    app.run(debug=True)