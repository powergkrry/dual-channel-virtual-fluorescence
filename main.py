#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:12:56 2021

@author: kanghyun
"""
#%%
import os
import model
import tensorflow as tf
# import tensorflow.keras.backend as K
import numpy as np
from config import get_config
from custom_data_gen import DataGenerator
from tensorflow import keras
from scipy import signal
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from plotting import plot_acc

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#%%
config, unparsed = get_config()
traingen = DataGenerator(num_images=704,
                         is_green=config.is_green,
                         batch_size=config.batch_size,
                         n_out_channels=config.n_out_channels,
                         shuffle=config.shuffle,
                         random_seed=config.random_seed)
testgen = DataGenerator(num_images=128,
                        is_train=False,
                        is_green=config.is_green,
                        batch_size=config.batch_size,
                        n_out_channels=config.n_out_channels,
                        shuffle=config.shuffle,
                        random_seed=config.random_seed)

#%%
model = model.get_model((256,256), config.n_sample, config.n_out_channels)
# model.layers[1].trainable = False
# learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=config.init_lr,
#     decay_steps=10000,
#     end_learning_rate=config.init_lr/10,
#     power=0.5)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

def gaussian_kernel(kernel_size, std):
    gkern1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d/np.power(kernel_size, 2)

def blur_mse_loss(y_true, y_pred):
    kernel_size = 7
    std=1
    
    kernel = tf.constant(gaussian_kernel(kernel_size=kernel_size, std=std),
                                         shape=[kernel_size,
                                                kernel_size,
                                                1, 1],
                                         dtype=tf.float32)
    
    blurred_y_true = tf.nn.conv2d(y_true, kernel, 
                                  strides=(1,1), padding="SAME")
    blurred_y_pred = tf.nn.conv2d(y_pred, kernel, 
                                  strides=(1,1), padding="SAME")
    
    l2loss = keras.losses.mean_squared_error(blurred_y_true, blurred_y_pred)
    
    # l1loss = tf.keras.regularizers.L1()(y_pred)
    return l2loss


#%%
model.compile(loss='MSE', optimizer=optimizer, metrics=['mse'])
# model.compile(loss=blur_mse_loss, optimizer=optimizer)
# model.summary()

model.fit(traingen,
          validation_data=testgen,
          epochs=config.epochs,
          shuffle=False,
          workers=8)

#%%
# model.layers[1].trainable = True
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1,
                                                 patience=5,
                                                 min_delta=5e-4,
                                                 min_lr=0.000001)
model.compile(loss=blur_mse_loss, optimizer=optimizer, metrics=[blur_mse_loss])
history = model.fit(traingen,
          validation_data=testgen,
          epochs=100,  #config.epochs,
          shuffle=True,
          workers=8,
          callbacks=[reduce_lr])
plot_acc(history, "blur_mse_loss")

#%%
index = 5
preds = model.predict(testgen[index][0])

fig, axs = plt.subplots(nrows=3, ncols=8)  # , figsize=(16,len(indices)*4)
counter = 0
for i in range(8):
    ax = axs[0,counter]
    ax.axis('off')
    ax.imshow(testgen[index][0][i,:,:,0], cmap='gray')  # , vmax=0.8
    ax.set_title(f"Center {counter}")
    ax = axs[1,counter]
    ax.axis('off')
    ax.imshow(testgen[index][1][i,:,:,0], vmin=0.01, vmax=0.4)
    ax.set_title(f"Ground Truth Green {counter}")
    ax = axs[2,counter]
    ax.axis('off')
    ax.set_title(f"Prediction Green {counter}")
    ax.imshow(preds[i], vmin=0.01, vmax=0.4)  # , vmin=0.01, vmax=1
    counter += 1
plt.show()
#%%
index = 12

inter_output_model = keras.Model(model.input, model.get_layer(index = 1).output)
inter_output = inter_output_model.predict(testgen[index])
