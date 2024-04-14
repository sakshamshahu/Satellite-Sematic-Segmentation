#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:55:10 2024

@author: saksham
"""

import tensorflow as tf
import os
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
from patchify import patchify

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()


patch_size = 256

dir_root = '/Users/saksham/Projects/Unet/data'

img_dataset = []

for path, subdirs, files in os.walk(dir_root):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    print(dirname)
    if dirname == 'Image':
        images = os.listdir(path)
        for i, image_name in enumerate(images):
            if image_name.endswith('JPG'):
                image = cv2.imread(path + '/' + image_name, 1)
                
                print('Creating smaller patches', path+'/'+ image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        
                        single_patch_img = patches_img[i,j,:,:]
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                    
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                        img_dataset.append(single_patch_img)
                