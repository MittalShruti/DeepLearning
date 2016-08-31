# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# After resizing images, predict if they have watermark
import logging


import os
from os import listdir
from os.path import isfile, join
import urllib, cStringIO
from flask import Flask, request, render_template
import numpy   # /usr/local/lib/python2.7/dist-packages/numpy
from PIL import Image
import keras
from keras.models import load_model
app = Flask(__name__)

@app.route('/')
def watermarkform():
    return render_template("watermark_form.html")

@app.route('/', methods=['POST'])
def watermarkform_post():    
    image = request.form['image_url']
    model = keras.models.load_model('shallowlargedropout.h5')
    #image = 'http://answers.opencv.org/upfiles/logo_2.png'
    file = cStringIO.StringIO(urllib.urlopen(image).read())
    img = Image.open(file)
    img = img.resize((32,32))
    im = numpy.array(img)

    # # arr = numpy.asarray(bytearray(req.read()), dtype=numpy.uint8)
    # # image = cv2.imdecode(arr,-1) # 'load it as it is'
    # imag = cv2.resize(image,(32,32), interpolation = cv2.INTER_AREA)
    # im = (numpy.array(imag)) #Converting to numpy array  #similar to cv2.imread(input_dir+'/'+image) #BGR 
    r = im[:,:,0] #Slicing to get R data
    g = im[:,:,1] #Slicing to get G data
    b = im[:,:,2] #Slicing to get B data        
    out = numpy.array([[r] + [g] + [b]], numpy.uint8) #Creating array with shape (3, 100, 100)
    out1 = out.astype('float32')/255
    predictions = model.predict_classes(out1)
    if predictions[0]==0:
        print "OLX"
    else:
        print "NON-OLX"

if __name__ == '__main__':
    app.run()
