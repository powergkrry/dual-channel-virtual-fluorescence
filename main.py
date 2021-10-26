#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:12:56 2021

@author: kanghyun
"""

import os
import model
from config import get_config
from custom_data_gen import DataGenerator
from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#%%
config, unparsed = get_config()
traingen = DataGenerator(num_images=704,
                         batch_size=config.batch_size,
                         n_out_channels=config.n_out_channels,
                         shuffle=config.shuffle,
                         random_seed=config.random_seed)
testgen = DataGenerator(num_images=128,
                        is_train=False,
                        batch_size=config.batch_size//2,
                        n_out_channels=config.n_out_channels,
                        shuffle=config.shuffle,
                        random_seed=config.random_seed)

#%%
model = model.get_model((256,256), config.n_out_channels)
learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=config.init_lr,
    decay_steps=10000,
    end_learning_rate=config.init_lr/10,
    power=0.5)
optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate_fn)
model.compile(loss='MSE', optimizer=optimizer)
# model.summary()

#%%
model.fit(traingen,
          validation_data=testgen,
          epochs=config.epochs,
          shuffle=False,
          workers=8)

#%%
index = 12
preds = model.predict(testgen[index])

fig, axs = plt.subplots(nrows=3, ncols=4) # , figsize=(16,len(indices)*4)
counter = 0
for i in range(4):
    ax = axs[0,counter]
    ax.axis('off')
    ax.imshow(testgen[index][0][i,:,:,0], vmax=0.8, cmap='gray')
    ax.set_title(f"Bright Field {counter}")
    ax = axs[1,counter]
    ax.axis('off')
    ax.imshow(testgen[index][1][i,:,:,0], vmin=0.01, vmax=1)
    ax.set_title(f"Ground Truth Green {counter}")
    ax = axs[2,counter]
    ax.axis('off')
    ax.set_title(f"Prediction Green {counter}")
    ax.imshow(preds[i], vmin=0.01, vmax=1)
    counter += 1