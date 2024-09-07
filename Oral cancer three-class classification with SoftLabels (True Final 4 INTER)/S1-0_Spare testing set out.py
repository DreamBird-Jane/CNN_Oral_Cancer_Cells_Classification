# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:54:35 2022

@author: user
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict

import os

#os.getcwd()
## Analysis directory
directory = "D:\Project_Jane\StemCellReg2021\Oral cancer three-class classification with SoftLabels (Prev INTER without 140 and 410)"
os.chdir(directory)
#%%                                        0. Define data directories
#%%                                      
## Data directory
data_folder = 'oral cancer 0521-0618_tag300_Val'
origion_path = "D:\\Project_Jane\\StemCellReg2021\\Data\\" + data_folder
Label0 = "control"
Label1 = "5nMTG"
Label2 = 'NonCancerHEK293T'
Labels = [Label0, Label1, Label2]

## Data directory: Intermediate(INTER) Data
data_new_folder = 'oral cancer inter stage-all_tag300_Val'
origion_new_path = "D:\\Project_Jane\\StemCellReg2021\\Data\\" + data_new_folder


Labels_INTER = ['TG2-WT1_0709', '20X_TG2-WT1', '20X_TG1-WT2', '20X_TG3-WT1', '20X_TG1-WT3'] 


#%%                                       2. Input testing set
#%%
'''
#Input testing set: INTER data
test_path = pd.read_csv(os.path.join('Testing set', 'test_index_INTERNone_t0.csv'), header=None).values
test_path = [i[0] for i in test_path.tolist()]

test_data= []
for j, i in enumerate(test_path):
    print(j, i)
    if j < 237:
        Label = Labels[0]
    elif j < 474:
        Label = Labels[1]        
    else:
        Label = Labels[2]
    print('current searched folder:', Label)    
    test_image = cv2.imread(os.path.join(origion_path, Label, i))[:,:,::-1]
    if test_image.size ==0:
        break    
    test_data.append(test_image)
test_data= np.array(test_data)


#Input testing set: INTER data
test_new_path = pd.read_csv(os.path.join('Testing set', 'test_index_INTER_INTERNone_t0.csv'), header=None).values
test_new_path = [i[0] for i in test_new_path.tolist()]

test_new_data= []
for j, i in enumerate(test_new_path):
    print(j, i)
    if j < 110:
        Label_INTER = Labels_INTER[0]
    elif j < 202:
        Label_INTER = Labels_INTER[1]
    elif j < 244:
        Label_INTER = Labels_INTER[2]
    elif j < 282:
        Label_INTER = Labels_INTER[3]
    elif j < 323:
        Label_INTER = Labels_INTER[4]        
    else:
        Label_INTER = Labels_INTER[5]
    print('current searched folder:', Label_INTER)    
    test_image = cv2.imread(os.path.join(origion_new_path, Label_INTER, i))[:,:,::-1]
    if test_image.size ==0:
        break    
    test_new_data.append(test_image)
test_new_data= np.array(test_new_data)

'''

#%%                                       2. Move selected files to ouput folder 
#%%
from shutil import move

## Data directory: output of testing set ------------------------------------------------------
data_folder2 = 'oral cancer 0521-0618_tag300_Test'
origion_path2 = "D:\\Project_Jane\\StemCellReg2021\\Data\\" + data_folder2

for Label in Labels:
    if not os.path.exists(os.path.join(origion_path2, Label)):
        os.mkdir(os.path.join(origion_path2, Label))

test_path = pd.read_csv(os.path.join('Testing set', 'test_index_DS_INTEROS_t0.csv'), header=None).values
test_path = [i[0] for i in test_path.tolist()]

for j, i in enumerate(test_path):
    print(j, i)
    if j < 60:
        Label = Labels[0]
    elif j < 120:
        Label = Labels[1]        
    else:
        Label = Labels[2]
    
    source = os.path.join(origion_path, Label, i)
    destination = os.path.join(origion_path2, Label, i)
    move(source, destination)
    

## INTER Data directory: output of testing set ------------------------------------------------------
data_new_folder2 = 'oral cancer inter stage-all_tag300_Test'
origion_new_path2 = "D:\\Project_Jane\\StemCellReg2021\\Data\\" + data_new_folder2

for Label_INTER in Labels_INTER:
    if not os.path.exists(os.path.join(origion_new_path2, Label_INTER)):
        os.mkdir(os.path.join(origion_new_path2, Label_INTER))

test_new_path = pd.read_csv(os.path.join('Testing set', 'test_index_INTER_DS_INTEROS_t0.csv'), header=None).values
test_new_path = [i[0] for i in test_new_path.tolist()]


for j, i in enumerate(test_new_path):
    print(j, i)
    if j < 0:
        Label_INTER = Labels_INTER[0]
    elif j < 40:
        Label_INTER = Labels_INTER[1]
    elif j < 80:
        Label_INTER = Labels_INTER[2]
    elif j < 120:
        Label_INTER = Labels_INTER[3]
    else:
        Label_INTER = Labels_INTER[4]
   
    source = os.path.join(origion_new_path, Label_INTER, i)
    destination = os.path.join(origion_new_path2, Label_INTER, i)
    move(source, destination)

