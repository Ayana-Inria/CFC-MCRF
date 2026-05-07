# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:11:27 2022

@author: marti

"""

import torch
import random
import numpy as np
import os
from skimage import io
try:
    from osgeo import gdal
except ImportError:
    gdal = None
from utils.utils_dataset import *
from utils.utils import *


# Dataset class
import os

class PRISMA_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, folder=input_folder, subfolders=subfolders, cache=False, augmentation=False):
        super(PRISMA_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        
        # List of files
        self.pan_files = [os.path.join(folder, id + '_Cube.tif') for id in ids]
        self.hsi_files = [os.path.join(folder, id + '_VNIR_SWIR.tif') for id in ids]
        self.label_files = [os.path.join(folder, id + '_gt_CRS_registered.tif') for id in ids]


        # Sanity check : raise an error if some files do not exist
        for f in self.pan_files + self.hsi_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.pan_cache_ = {}
        self.label_cache_ = {}
        self.hsi_cache_ = {}
            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.pan_files) - 1)
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.pan_cache_.keys():
            pan = self.pan_cache_[random_idx]
            hsi = self.hsi_cache_[random_idx]

        else:
            # Data is normalized in [0, 1]
            if gdal is None:
                raise ImportError('GDAL is required to read PRISMA .tif files. Install it with conda install -c conda-forge gdal.')
            pan = gdal.Open(self.pan_files[random_idx], gdal.GA_ReadOnly)
            pan = pan.ReadAsArray()
            hsi = gdal.Open(self.hsi_files[random_idx], gdal.GA_ReadOnly)
            hsi = hsi.ReadAsArray()

            pan = np.asarray(np.expand_dims(pan, axis=0), dtype='float32')
            hsi1 = np.asarray(hsi, dtype='float32')[3:66,:,:]
            hsi2 = np.asarray(hsi, dtype='float32')[69:,:,:]
            hsi = np.concatenate((hsi1, hsi2), 0)
            pan[np.isnan(pan)] = 0
            hsi[np.isnan(hsi)] = 0
            pan = 1/255 * pan
            print(pan.shape)
            hsi = 1/255 * hsi
            print(hsi.shape)
            
            if self.cache:
                self.pan_cache_[random_idx] = pan
                self.hsi_cache_[random_idx] = hsi
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            label = io.imread(self.label_files[random_idx])
            label = np.asarray(label[:,:], dtype='int64')
            print(label.shape)
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, x3, x4, y1, y2, y3, y4 = get_patches(hsi, label, WINDOW_SIZE)
        pan_p = pan[:,x1:x2,y1:y2]
        hsi_p = hsi[:,x3:x4,y3:y4]
        label_p = label[x1:x2,y1:y2]
        

        # Return the torch.Tensor values
        return (torch.from_numpy(pan_p), torch.from_numpy(hsi_p),
                torch.from_numpy(label_p))