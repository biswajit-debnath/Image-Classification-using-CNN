# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:07:34 2020

@author: Biswajit Debnath
"""
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
"""
model= load_model("Saved_Weight.h5")


//img2=image.load_img('C:/Users/ranjay kumar/Desktop/Ml/test 1/test/cat.jpg', target_size=(150, 150))
img2=image.img_to_array(img2)
img2=np.expand_dims(img2, axis = 0)
rst=model.predict(img2)
train_generator.class_indices
print("Cat" if rst[0][0] > rst[0][1] else "Dog")
"""


from keras.models import Sequential
from keras.layers import Conv2D,Activation,Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K

img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 


model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(64, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 

model.compile(loss ='binary_crossentropy', 
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 


model.load_weights("Saved_Weight.h5")


img2=image.load_img('C:/Users/Biswajit/Desktop/Ml/test 1/test/cat.jpg', target_size=(150, 150))
img2=image.img_to_array(img2)
img2=np.expand_dims(img2, axis = 0)
rst=model.predict(img2)
print("Cat" if rst[0][0] < 0.5 else "Dog")





