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

from plotting import plot_acc, plot_predictions


#%%
config, unparsed = get_config()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
model = model.get_model((256,256), config.n_sample, config.n_out_channels,
                        config.final_activation)

learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=config.init_lr,
    decay_steps=10000,
    end_learning_rate=config.init_lr/100,
    power=0.5)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


def gaussian_kernel(kernel_size, std):
    gkern1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d/np.power(kernel_size, 2)


def blur_mse_loss(y_true, y_pred):
    kernel_size = 7
    std = 1
    
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
    
    l1loss = keras.losses.mean_absolute_error(y_pred, tf.zeros_like(y_pred))
    return l2loss + config.lamda*l1loss


def mse_plus_reg(y_true, y_pred):
    l2loss = keras.losses.mean_squared_error(y_true, y_pred)
    
    l1loss = keras.losses.mean_absolute_error(y_pred, tf.zeros_like(y_pred))
    return l2loss + config.lamda*l1loss


def get_loss():
    if config.loss == "blur":
        return blur_mse_loss
    elif config.loss == "mse-r":
        return mse_plus_reg
    else:
        print("Assuming that you are using a predefined loss")
        return config.loss

#%%
# model.layers[1].trainable = True
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                                  factor=0.1,
#                                                  patience=5,
#                                                  min_delta=5e-4,
#                                                  min_lr=0.000001)

model.compile(loss=get_loss(), optimizer=optimizer, metrics=['mse'])

history = model.fit(traingen,
          validation_data=testgen,
          epochs=config.epochs,
          shuffle=False,
          workers=8)

plot_acc(history, "loss")

#%%
import json

current_directory = os.getcwd()
print("Making a folder in current directory: {}".format(current_directory))
if not os.path.exists(current_directory+"/"+config.name):
    os.mkdir(current_directory+"/"+config.name)
os.chdir(current_directory+"/"+config.name)

val_metrics = {}
val_metrics["val_loss_last"] = history.history["val_loss"][-1]
val_metrics["val_mse_last"] = history.history["val_mse"][-1]
val_metrics["val_loss_best"] = min(history.history["val_loss"])
val_metrics["val_loss_best_epoch"] = history.history["val_loss"].index(
                                                val_metrics["val_loss_best"])
val_metrics["val_mse_at best_loss"] = history.history["val_mse"]\
                                        [val_metrics["val_loss_best_epoch"]]
val_metrics["val_mse_best"] = min(history.history["val_mse"])
val_metrics["val_mse_best_epoch"] = history.history["val_mse"].index(
                                                val_metrics["val_mse_best"])
val_metrics["val_loss_at best_mse"] = history.history["val_loss"]\
                                        [val_metrics["val_mse_best_epoch"]]



with open('experiment_params.json', 'w') as f:
    json.dump(config.__dict__, f, indent=2)
with open('validation_mterics.json', 'w') as f:
    json.dump(val_metrics, f, indent=2)

plot_predictions(trained_model=model, testgen=testgen)
#%%
# index = 12

# inter_output_model = keras.Model(model.input, model.get_layer(index = 1).output)
# inter_output = inter_output_model.predict(testgen[index])
