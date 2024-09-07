# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 19:21:50 2021

@author: user
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict

import os
from os import listdir, walk

'''
data_folder = 'lung cancer 0615-0600'
Label0 = "h1975"
Label1 = "or4"
Labels = [Label0,Label1]

## Data directory
origion_path = "D:\\Project_Jane\\StemCellReg2021\\Data\\" + data_folder
## Analysis directory
directory = "D:\Project_Jane\StemCellReg2021\Lung cancer binary classification_K-fold CV"

#Define the splitted file names and their ratios (sum=1)
split_ratio = OrderedDict({'train': 0.85, 'test': 0.15})
'''
#%%                                        
#                                               O. Full Data Preprocessing

#%%

#%%                                            0.1 Resize of Input Data set 
def Resize_Prep(input_data, resize):
    """
    Purpose: to resize an array of image data, e.g., (148, 1200, 1600, 3) --> (148, 300, 300, 3)
    Input:
        input_data:　numpy.ndarray of RGB images with shape: (N, H, W, 3)
        resize: None or tupple - (H',W') <-- the requirement for reset Height (H) & Width (W)    
    Output:
        output_data: numpy.ndarray of RGB images with new shape: (N, H',W', 3)
    """
    if resize != None:
        output_data = []
        for img in input_data:  
            img = cv2.resize(img, resize)
            output_data.append(img)     
        output_data = np.array(output_data)
    elif resize == None:
        output_data = input_data
    
    return output_data

#%%                                            0.2 Equalize & Binaritize Input Data set 

from skimage import exposure

def img_equalize_binary(image, threshold=None):
    '''
    equalized gray-scale/ RGB image using "Contrast Limited Adaptive Histogram Equalization (CLAHE)" (which turn data range into [0,1]),
    and then binarize image according to threshold(defaul 0.5)
    Input:
        image: np.array
                if len(image.shape)=2-> gray-scale with shape = (W, H)
                if len(image.shape)=3-> RGB-scale with shape = (W, H, 3)
        threshold: None or float value
            if !=None, then each pixel value is binarized by threshold=a
    Output:
        image_bi: np.array of binarized data
            if gray-scale: shape = (W, H)
            if RGB-scale: shape = (W, H, 3), note that CLAHE is performed individually for each color channel
    '''
    if len(image.shape)==2:
        image_eq =  exposure.equalize_adapthist(image)
        if threshold==None:
            image_bi = image_eq
        else:
            image_bi =  np.where(image_eq > threshold, 1, 0)
    elif len(image.shape)==3:
        image_eq_R =  exposure.equalize_adapthist(image[:,:,0])
        image_eq_G =  exposure.equalize_adapthist(image[:,:,1])
        image_eq_B =  exposure.equalize_adapthist(image[:,:,2])
        image_eq = np.concatenate([image_eq_R[:,:, np.newaxis], image_eq_G[:,:, np.newaxis], image_eq_B[:,:, np.newaxis]], axis = 2)
        if threshold==None:
            image_bi = image_eq
        else:
            image_bi =  np.where(image_eq > threshold, 1, 0)
        
    return image_bi
'''
image_bi = img_equalize_binary(image, threshold=0.5)
# for Gray imgae 
plt.imshow(image_bi, cmap='gray')
plt.show()
# for RGB imgae 
plt.imshow(image_bi*255)# transform {0,1} to {0, 255} binary data so as for RGB visualization
plt.show()
'''

#%%
#                      I. Random Strafified Split for, say, Training / Validatrion / Testing sets
    
#%%
def Split_Data_Strafied(origion_path, Labels, split_ratio = OrderedDict({'train': 0.80, 'test': 0.20}), flag=1, resize = None):
    """
    Input:
        origion_path: data directory (a folder with subfolders of labels)
        Labels: list of labels; e.g., Label0 = "h1975"; Label1 = "or4"; Labels = [Label0,Label1]
        split_ratio: Define the splitted file names and their ratios (sum=1)
        flag: color mode the image should be read; 1 = cv2.IMREAD_COLOR, 0 = cv2.IMREAD_GRAYSCALE
        resize: None (if you don't want to resize input image); o.w., set resize = say, (300, 300),...
    Output:
        split_data: dictionary with key names = each label, values = a dictionary of datasets of each split dataset (e.g., train, test)
        split_index: dictionary with key names = each label, values = a dictionary of index names of each split dataset (e.g., train, test)
    """
    split_data = OrderedDict({}) # to import data per label per split set; e.g., {'label0': data_dict,...} -> data_dict = {'train': data, 'test': data}
    split_index = OrderedDict({}) # # to record indices of data per label per split set
    
    for label in Labels: #os.listdir(origion_path)
        print('Current label:', label)
        image_list = os.listdir(os.path.join(origion_path, label))
        N_label = len(image_list)
    
        i = 0
        split_dict = {}        
        for key, ratio in split_ratio.items():
            if i < len(split_ratio)-1:
                image_sublist = np.random.choice(image_list, int(round(N_label*ratio,0)), replace=False).tolist()
                split_dict[key] = image_sublist
            
                image_list = list(set(image_list) - set(image_sublist))
                i += 1
            else:
                image_sublist = image_list
                split_dict[key] = image_sublist
                
            #print(f'The image files (with number of {len(image_sublist)}) are randomly distributed to {key}-group')
        split_index[label] = split_dict 
        
        data_dict = {}
        for key, image_sublist in split_dict.items():
            #print('Current group:', key)            
        
            raw_data = []
            for image in image_sublist:
                #print('Current image file:', image)
                if flag==1:
                    raw_image = cv2.imread(os.path.join(origion_path, label, image))[:,:,::-1]
                else:
                    raw_image = cv2.imread(os.path.join(origion_path, label, image), flag)
                    
                if resize != None:
                    raw_image = cv2.resize(raw_image, resize)
                raw_data.append(raw_image)                
            raw_data = np.array(raw_data)
            data_dict[key] = raw_data            
                
        split_data[label] = data_dict
    
    return split_data, split_index
'''
#Application

from CNN_Procedure import S1_Prep_RadomSplit as S1

#Define the splitted file names and their ratios (sum=1)
from collections import OrderedDict
split_ratio = OrderedDict({'train': 0.60, 'val': 0.20, 'test': 0.20})

resize = None

split_data, split_index = S1.Split_Data_Strafied(origion_path, Labels, split_ratio, resize = resize) 

#Extract and name corresponding data - hard label ----------------------------------------------------------
train_data = np.concatenate([split['train'] for label, split in split_data.items()], 0)
train_label = np.concatenate([np.zeros(len(split['train']), dtype=np.int_) + Labels_dic[label] for label, split in split_data.items()], 0).astype(np.int_)
train_index = [x  for label, split in split_index.items() for x in split['train']]

val_data = np.concatenate([split['val'] for label, split in split_data.items()], 0)
val_label = np.concatenate([np.zeros(len(split['val']), dtype=np.int_) + Labels_dic[label] for label, split in split_data.items()], 0).astype(np.int_)
val_index = [x  for label, split in split_index.items() for x in split['val']]

test_data = np.concatenate([split['test'] for label, split in split_data.items()], 0)
test_label = np.concatenate([np.zeros(len(split['test']), dtype=np.int_) + Labels_dic[label] for label, split in split_data.items()], 0).astype(np.int_)
test_index = [x  for label, split in split_index.items() for x in split['test']]
'''


