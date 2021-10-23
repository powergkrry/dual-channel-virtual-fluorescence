#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:58:19 2021

@author: kanghyun
"""


import os
import cv2 as cv
from tensorflow import keras
import numpy as np


# paths=[['2021_10_08','00'],['2021_10_08','01'],['2021_10_08','03'],['2021_10_08','04'],['2021_10_08','05'],['2021_10_08','06'],['2021_10_08','08'],['2021_10_08','09'],['2021_10_08','10'],['2021_10_08','11'],['2021_10_08','12'],['2021_10_08','13'],['2021_10_08','14']]

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, base_path='/home/kanghyun/Desktop/', batch_size=16, crop_size=256, n_in_channels=21, n_out_channels=1, shuffle=True, random_seed=0):
        self.crop_size = crop_size
        self.list_IDs = np.repeat(list_IDs,np.power(2048//self.crop_size,2),axis=0)
        self.base_path = base_path
        self.batch_size = batch_size
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.on_epoch_end()
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp, indexes)
        return X, y
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, indexes):
        X = np.empty((self.batch_size, self.crop_size, self.crop_size, self.n_in_channels))
        y = np.empty((self.batch_size, self.crop_size, self.crop_size, self.n_out_channels))

        for i, ID in enumerate(list_IDs_temp):
            row_pos = indexes[i]%np.power(2048//self.crop_size,2) // self.crop_size
            col_pos = indexes[i]%np.power(2048//self.crop_size,2) % self.crop_size
            
            for led in range(self.n_in_channels):
                image = cv.imread(os.path.join(self.base_path, f'{ID[0]}/TB/{ID[1]}/cropped_LED_{led:03d}.tif'), cv.IMREAD_ANYDEPTH)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                X[i,:,:,led] = image[row_pos:row_pos+self.crop_size, col_pos:col_pos+self.crop_size]
            image = cv.imread(os.path.join(self.base_path, f'{ID[0]}/TB/{ID[1]}/cropped_red_flu_AIF.tif'), cv.IMREAD_ANYDEPTH)
            low = np.percentile(image, 5)
            high = np.percentile(image, 99.99)
            image = (image - np.min(low)) / (np.max(high) - np.min(low))
            image = np.clip(np.array(image), 0, 1)
            y[i,:,:,0] = image[row_pos:row_pos+self.crop_size, col_pos:col_pos+self.crop_size]

        return X, y