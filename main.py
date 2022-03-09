#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:12:56 2021

@author: kanghyun
"""
#%%
import os
import model
import json
import math
import losses
import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras.backend as K
from config import get_config
from custom_data_gen import DataGenerator

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from plotting import plot_acc, plot_predictions


#%%
config, unparsed = get_config()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

traingen = DataGenerator(num_images=1078,
                         is_green=config.is_green,
                         batch_size=config.batch_size,
                         n_out_channels=config.n_out_channels,
                         shuffle=config.shuffle,
                         random_seed=config.random_seed)
testgen = DataGenerator(num_images=294,
                        is_train=False,
                        is_green=config.is_green,
                        batch_size=config.val_batch_size,
                        n_out_channels=config.n_out_channels,
                        shuffle=config.shuffle,
                        random_seed=config.random_seed)

#%%
model = model.get_model((256, 256), config.n_sample, config.n_out_channels,
                        config.final_activation, config.layer_activation, 
                        maxpooling=config.maxpool, attention=config.att, 
                        annealing=config.ann)

callback = []

def step_decay(epoch):
    initial_lrate = config.init_lr
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    return lrate

if config.polydecay:
    learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config.init_lr,
        decay_steps=config.lr_decay_steps,
        end_learning_rate=config.init_lr/config.lr_reduction_factor,
        power=0.3162278)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-07)

elif config.plateaudecay:
    min_lr = config.init_lr/config.lr_reduction_factor
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=10,
                                                     cooldown=1,
                                                     min_delta=5e-6,
                                                     min_lr=min_lr,
                                                     verbose=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.init_lr,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-07)
    callback.append(reduce_lr)

elif config.stepdecay:
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.init_lr,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-07)
    callback.append(lrate)

else:
    print("No schedule for optimizer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.init_lr,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-07)      


#%%
model.compile(loss=losses.get_loss(), optimizer=optimizer, 
              metrics=['mse', losses.ssim])

history = model.fit(traingen,
                    validation_data=testgen,
                    epochs=config.epochs,
                    shuffle=False,
                    workers=8,
                    callbacks=callback)

#%%
current_directory = os.getcwd()
print("Making a folder in current directory: {}".format(current_directory))
if not os.path.exists(current_directory+"/"+config.name):
    os.mkdir(current_directory+"/"+config.name)
os.chdir(current_directory+"/"+config.name)

plot_acc(history, "loss", save=True)
plot_acc(history, "ssim", save=True, fname="ssim")

val_metrics = {}
val_metrics["val_loss_last"] = history.history["val_loss"][-1]
val_metrics["val_mse_last"] = history.history["val_mse"][-1]
val_metrics["ssim_last"] = history.history["ssim"][-1]
val_metrics["val_loss_best"] = min(history.history["val_loss"])
val_metrics["val_loss_best_epoch"] = history.history["val_loss"].index(
                                                val_metrics["val_loss_best"])
val_metrics["ssim_best_loss"] = history.history["ssim"][
                                        val_metrics["val_loss_best_epoch"]]
val_metrics["val_mse_at best_loss"] = history.history["val_mse"][
                                        val_metrics["val_loss_best_epoch"]]
val_metrics["val_mse_best"] = min(history.history["val_mse"])
val_metrics["val_mse_best_epoch"] = history.history["val_mse"].index(
                                                val_metrics["val_mse_best"])
val_metrics["val_loss_at best_mse"] = history.history["val_loss"][
                                        val_metrics["val_mse_best_epoch"]]
val_metrics["ssim_at best_mse"] = history.history["ssim"][
                                        val_metrics["val_mse_best_epoch"]]


with open('experiment_params.json', 'w') as f:
    json.dump(config.__dict__, f, indent=2)
with open('validation_mterics.json', 'w') as f:
    json.dump(val_metrics, f, indent=2)

plot_predictions(trained_model=model, testgen=testgen)
