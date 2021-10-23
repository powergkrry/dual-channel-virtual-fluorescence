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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#%%
train_path = [['2021_10_08','00'],['2021_10_08','01'],['2021_10_08','03'],['2021_10_08','04'],['2021_10_08','05'],['2021_10_08','06'],['2021_10_08','08'],['2021_10_08','09'],['2021_10_08','10'],['2021_10_08','11'],['2021_10_08','12']]
test_path = [['2021_10_08','13'],['2021_10_08','14']]

#%%
traingen = DataGenerator(train_path, batch_size=16)
testgen = DataGenerator(test_path, batch_size=16)

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
simple_model.fit(traingen, validation_data=testgen, epochs=10, shuffle=False, workers=8)

#%%
# traingen[0][]