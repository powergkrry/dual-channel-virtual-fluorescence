#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:58:19 2021

@author: kanghyun
"""


import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 num_images,
                 base_path='/home/kanghyun_gpu/Desktop/dual-channel-virtual-fluorescence/Data/',
                 is_train=True,
                 is_green=True,
                 batch_size=16,
                 crop_size=256,
                 n_in_channels=21,
                 n_out_channels=1,
                 shuffle=True,
                 random_seed=0):
        self.crop_size = crop_size
        self.num_images = num_images
        self.base_path = base_path
        self.is_train = is_train
        self.augmentor = ImageDataGenerator(
            horizontal_flip=True, vertical_flip=True)\
            if is_train else ImageDataGenerator()

        self.is_green = is_green
        self.batch_size = batch_size
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def __len__(self):
        return int(np.floor(self.num_images / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_images)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.crop_size,
                      self.crop_size, self.n_in_channels))
        y = np.empty((self.batch_size, self.crop_size,
                      self.crop_size, self.n_out_channels))

        for i, ID in enumerate(indexes):
            train_test = 'train' if self.is_train else 'test'
            green_red = 'green' if self.is_green else 'red'
            X[i] = np.load(os.path.join(
                self.base_path, train_test,
                f'cropped_LED_array_{ID:04d}_{train_test}.npy'),
                mmap_mode='r')
            y[i] = np.load(os.path.join(
              self.base_path, train_test,
              f'cropped_{green_red}_flu_AIF_array_{ID:04d}_{train_test}.npy'),
              mmap_mode='r')

        Xy = np.concatenate((X, y), axis=3)
        Xy_gen = self.augmentor.flow(Xy, batch_size=self.batch_size,
                                     shuffle=False)
        Xy = next(Xy_gen)
        return Xy[..., :self.n_in_channels], Xy[..., self.n_in_channels:]
