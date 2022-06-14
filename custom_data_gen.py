#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:58:19 2021

@author: kanghyun
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 num_images,
                 base_path='/home/kanghyun_gpu/Desktop/dual-channel-virtual-fluorescence/Data/', # /data2/amey/TB/Data/
                 is_train=True,
                 is_green=True,
                 is_semantic=False,
                 class1_weight=1.0,
                 class2_weight=1.0,
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
        self.is_semantic = is_semantic
        self.class1_weight = class1_weight
        self.class2_weight = class2_weight
        self.batch_size = batch_size
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if self.is_semantic:
            X, y, w = self.__data_generation(indexes)
            return X, y, w
        else:
            X, y = self.__data_generation(indexes)
            return X, y

    def __len__(self):
        return int(np.floor(self.num_images / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_images)
        if self.shuffle is True and self.is_train is True:
            np.random.shuffle(self.indexes)

    def custom_augmentor_flow_reindex(self, Xy,
                                      rotation_state, horizontal_flip_state):
        rotation_reindex = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21],
                            [0,4,1,2,3,11,12,5,6,7,8,9,10,19,20,13,14,15,16,17,18, 21],
                            [0,3,4,1,2,9,10,11,12,5,6,7,8,17,18,19,20,13,14,15,16, 21],
                            [0,2,3,4,1,7,8,9,10,11,12,5,6,15,16,17,18,19,20,13,14, 21]])
        horizontal_flip_reindex = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21],
                                   [0,2,1,4,3,10,9,8,7,6,5,12,11,18,17,16,15,14,13,20,19, 21]])
        if self.is_semantic:
            rotation_reindex = np.insert(rotation_reindex,
                                         rotation_reindex.shape[1], 22, axis=1)
            horizontal_flip_reindex = np.insert(horizontal_flip_reindex,
                                         horizontal_flip_reindex.shape[1], 22, axis=1)
        
        Xy = np.array(list(map(lambda Xy, rotation_state:
                               Xy[...,rotation_reindex[rotation_state]],
                               Xy, rotation_state)))
        Xy = np.array(list(map(lambda Xy, horizontal_flip_state:
                               Xy[...,horizontal_flip_reindex[horizontal_flip_state]],
                               Xy, horizontal_flip_state)))
        
        return Xy
    
    def custom_augmentor_flow_augment(self, Xy,
                              rotation_state, horizontal_flip_state):
        Xy = np.array(list(map(lambda Xy, rotation_state:
                               np.rot90(Xy, rotation_state),
                               Xy, rotation_state)))
        Xy = np.array(list(map(lambda Xy, horizontal_flip_state:\
                               np.flip(Xy, 1) if horizontal_flip_state\
                                   else Xy,\
                                       Xy, horizontal_flip_state)))
        
        return self.custom_augmentor_flow_reindex(Xy,
                                    rotation_state=rotation_state,
                                    horizontal_flip_state=horizontal_flip_state)

    def class_weight_generation(self, y):
        class_weights = tf.constant([1.0, self.class1_weight, self.class2_weight])
        class_weights = class_weights/tf.reduce_sum(class_weights)
        
        sample_weights = tf.gather(class_weights, indices=tf.cast(y, tf.int32))
        return sample_weights

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
            if self.is_semantic:
                y[i] = np.load(os.path.join(
                  self.base_path, train_test,
                  f'cropped_flu_mask_array_{ID:04d}_{train_test}.npy'),
                  mmap_mode='r')
            else:
                y[i] = np.load(os.path.join(
                  self.base_path, train_test,
                  f'cropped_{green_red}_flu_AIF_array_{ID:04d}_{train_test}.npy'),
                  mmap_mode='r')

        if self.is_semantic:
            sample_weights = self.class_weight_generation(y)
            Xy = np.concatenate((X, y, sample_weights), axis=3)
        else:
            Xy = np.concatenate((X, y), axis=3)
        rotation_state = np.random.choice(4, self.batch_size)
        horizontal_flip_state = np.random.choice(2, self.batch_size)
        if self.is_train:
            Xy = self.custom_augmentor_flow_augment(Xy,
                                    rotation_state=rotation_state,
                                    horizontal_flip_state=horizontal_flip_state)
        
        if self.is_semantic:
            return Xy[..., :self.n_in_channels],\
                Xy[..., self.n_in_channels:self.n_in_channels+self.n_out_channels],\
                Xy[..., self.n_in_channels+self.n_out_channels:]
        else:
            return Xy[..., :self.n_in_channels], Xy[..., self.n_in_channels:]
