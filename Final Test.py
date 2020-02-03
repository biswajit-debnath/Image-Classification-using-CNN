# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:07:34 2020

@author: Biswajit Debnath
"""
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model= load_model("Saved_Weight.h5")


img2=image.load_img('cat.jpg', target_size=(150, 150))
img2=image.img_to_array(img2)
img2=np.expand_dims(img2, axis = 0)
rst=model.predict(img2)
train_generator.class_indices
print("Cat" if rst[0][0] > rst[0][1] else "Dog")




