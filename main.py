#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:12:56 2021

@author: kanghyun
"""

import os
from tensorflow import keras
import matplotlib.pyplot as plt
from custom_data_gen import DataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#%%
traingen = DataGenerator(num_images=704)
testgen = DataGenerator(num_images=128, is_train=False)

#%%
def create_simple_model(input_size):
    inp = keras.layers.Input(input_size)
    
    # just apply a bunch of convs
    x = keras.layers.BatchNormalization()(inp)
    x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(1, 1, activation='relu', padding='same')(x)
    
    model = keras.Model(inp, x)
    return model

#%%
simple_model = create_simple_model((256, 256, 21))
simple_model.compile(loss='MSE') # optimizer=Adam(lr=0.001),
# simple_model.summary()

#%%
simple_model.fit(traingen, validation_data=testgen, epochs=3, shuffle=False, workers=8)

#%%
index = 0
preds = simple_model.predict(testgen[index])

i = 8
plt.figure()
plt.imshow(testgen[index][1][i],vmin=0,vmax=1)
plt.figure()
plt.imshow(preds[i],vmin=0,vmax=1)