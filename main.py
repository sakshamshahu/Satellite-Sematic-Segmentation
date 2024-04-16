#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:55:10 2024

@author: saksham
"""

import os
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
from patchify import patchify
import random

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm

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
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

'''
Banana:128,192,128::
Coconut:250,125,187::
House:184,61,245::
Rice:250,250,55::
Road:250,50,83::
Tree:144,96,0::
Water:51,221,255::
background:0,0,0::

'''

Banana = np.array((128,192,128))
Coconut = np.array((250,125,187))
House  = np.array((184,61,245))
Rice = np.array((250,250,55))
Road = np.array((250,50,83))
Tree = np.array((144,96,0))
Water = np.array((51,221,255))
Unlabelled = np.array((0,0,0))


label = single_patch_mask


def rgb_to_2D_label(label):
    """
    Replace pixels with specific RGB values
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    
    label_seg [np.all(label == Banana,axis=-1)] = 0
    label_seg [np.all(label==Coconut,axis=-1)] = 1
    label_seg [np.all(label==House,axis=-1)] = 2
    label_seg [np.all(label==Rice,axis=-1)] = 3
    label_seg [np.all(label== Road,axis=-1)] = 4
    label_seg [np.all(label== Tree,axis=-1)] = 5
    label_seg [np.all(label== Water,axis=-1)] = 6
    label_seg [np.all(label== Unlabelled,axis=-1)] = 7    
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)
    
labels = np.array(labels)
labels = np.expand_dims(labels, axis = 3)

print(np.unique(labels))


image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()


n_classes = len(np.unique(labels))
from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)

weights = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #similar to IOU


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


from Multi_unet import multi_unet_model, jacard_coef  

metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)







