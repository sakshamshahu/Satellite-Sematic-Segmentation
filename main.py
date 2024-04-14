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
import random
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

mask_dataset = []  
for path, subdirs, files in os.walk(dir_root): 
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'Masks':
        masks = os.listdir(path)
        for i, mask_name in enumerate(masks):  
            if mask_name.endswith(".png"):
               
                mask = cv2.imread(path+"/"+mask_name, 1)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                mask = Image.fromarray(mask)
                mask = np.array(mask)             
       
                print("Now patchifying mask:", path+"/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  
        
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        
                        single_patch_mask = patches_mask[i,j,:,:]
                        single_patch_mask = single_patch_mask[0]                          
                        mask_dataset.append(single_patch_mask) 
 
image_dataset = np.array(img_dataset)
mask_dataset =  np.array(mask_dataset)


image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(119)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()


