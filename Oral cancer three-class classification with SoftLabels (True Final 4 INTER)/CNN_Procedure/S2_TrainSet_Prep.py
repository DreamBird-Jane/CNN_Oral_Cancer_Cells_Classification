# -*- coding: utf-8 -*-
# %%
"""
Created on Sun Jul 11 19:23:05 2021

@author: user
"""
import os
from os import listdir, walk
#from os.path import join

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE

#from Image_Oversampling_with_CutMix import CutMix

from collections import Counter
# %%
#                                                        O. Simple preprocessing

# %%
#label_list = [1/3, 2/3, 0] #[4/5, 1/5, 0]
def Select_subset(X, Y, label_list):
    """
    Select subset data of X, accroding to label of Y (Multi-class applicable)
    Input:
        X: general ndarray data
        Y: target (one-hot encoder or soft label), shape=(N,K), where N=sampel size, K=#{class}
        label_list: list with len=K; e.g., K=3 -> [4/5, 1/5, 0]
    Output:
        X: selected
    """
    
    condition = np.zeros((len(Y),))==0
    for i in range(Y.shape[1]):
        condition = condition & (Y[:,i]==label_list[i])             
    
    X = X[condition] 
    
    return X

def Select_subset_flatten(data_list, target_list, label_lists ):
    '''
    Generalized version of Select_subset(X, Y, label_list)
    Input:
        data_list: list of Xs; general ndarray data
        target_list: list of target data ys of Xs (one-hot encoder or soft label), shape per target=(N,K), where N=sampel size, K=#{class}
        label_lists: list of label_list (-> with len=K; e.g., K=3 -> [4/5, 1/5, 0] ) used for subset selection of (X, y)
        
        e.g.,   data_list = [train_skf_data, train_skf_data_INTER, train_skf_data_INTER]
                target_list = [train_skf_label, train_label_INTER, train_label_INTER]
                label_lists = [[1, 0, 0], [3/4,1/4,0], [4/5,1/5,0]] 
    
    Output:
        data_concat_flat: selected, concatednated and flattened data from X based on label_lists
        N_concat: sample size for list of Xs
        target_concat: selected and concatednated data y based on label_lists
                originally, the select target data should be [label_lists[i]*N_concat[i]] for i-th data_list, 
                but we transform it into [[i]*N_concat[i]] (i.e., new label= 0, 1, 2... for different X)
    '''
    data_concat = np.array([]).reshape((0, data_list[0].shape[1], data_list[0].shape[2], data_list[0].shape[3])) 
    target_concat = np.array([]).astype(np.int_)
    N_concat = []
    i = 0
    # data, target, label_list = data_list[i], target_list[i], label_lists[i]
    for data, target, label_list in zip(data_list, target_list, label_lists):
        print(data.shape,"\t", target.shape, "\t", label_list)
        #if data.shape[0] != target.shape[0]:
         #   return print(f"for label_list={label_list}, the sample size for data(={data.shape[0]}) should be equal to that of target(={target.shape[0]})")        
        data_select = Select_subset(data, target, label_list) 
        N = len(data_select)
        label_new = np.array([i]*N).astype(np.int_)
        print(f"for label_list={label_list}:\n the sample size selected is {N}, labeling as {i}")
        
        data_concat = np.concatenate([data_concat, data_select])
        target_concat = np.concatenate([target_concat, label_new])
        N_concat.append(N)
        i+=1
        
    #Flantten the dataset (so as tabular data for machine learning)
    data_concat_flat = data_concat.reshape((len(data_concat), -1))
    
    return data_concat_flat, target_concat, N_concat

#data_concat_flat, target_concat, N_concat =    select_subset_flatten(data_list, target_list, label_lists )
#N_concat =  Counter(target_concat2)

def ReshapeX_TransY(data_concat_flat, target_concat, label_lists, img_shape=(300, 300, 3)):
    '''
    Transform tabular data (X, y) into input ready for CNN analysis
    Input:
        data_concat_flat: flattened data X with shape = (sum(N_concat), img_shape[0]*img_shape[1]*img_shape[2])        
        target_concat: transform lables y (i.e., new label= 0, 1,... for different X)
        label_lists: original labels y (one-hot encoder or soft labels)  ; e.g., label_lists = [[2/3, 1/3, 0], [1/3, 2/3, 0]]      
    Output:
        data_concat: reshaped X with shape = (sum(N_concat), img_shape[0], img_shape[1], img_shape[2]), for input of CNN
        target_concat_m: transformed y back to original labels according to label_lists, with shape = (sum(N_concat), len(label_lists[0]))
        originally, the select target data should be [label_lists[i]*N_concat[i]] for i-th data_list, 
                but we transform it into [[i]*N_concat[i]] 
    '''
    N_concat =  Counter(target_concat) #e.g., Counter({0: 295, 1: 135}) sample size for each label (key)
    data_concat = np.array([]).reshape((0, img_shape[0], img_shape[1], img_shape[2])) 
    target_concat_m = np.array([]).reshape((0, len(label_lists[0]))) #????????????????????????????????????
    for key, value in N_concat.items():
        print("Label: ",key, " transformed into: ", label_lists[key],"\t with N=", value)
                
        data_concat0 = data_concat_flat[target_concat==key]
        data_concat0 = data_concat0.reshape(value, img_shape[0], img_shape[1], img_shape[2])
        target_concat_m0 = np.array([label_lists[key]*value]).reshape((-1, 3))  
                
        data_concat = np.concatenate([data_concat, data_concat0])
        target_concat_m = np.concatenate([target_concat_m, target_concat_m0])
    del data_concat0, target_concat_m0
    return data_concat, target_concat_m

def Resample(input_data, N_exp):
    """
    input data: data array of X (shape = [N0, H, W(, C)])
    N_exp: expected output sample size(>1); notice N_exp can be >, =, or < N0.
    """           
    N0 = len(input_data)
    index = np.arange(N0).tolist()
            
    if N_exp > N0:
        print("N_exp > N0")
        #Random-Oversample (with replacement), 
        index_re = random.choices(index, k=N_exp) 
        output_data = input_data[index_re]
    elif N_exp == N0:
        print("N_exp == N0")
        output_data = input_data
    else:
        print("N_exp < N0")
        #Random-Undersample (without replacement)
        index_re = random.sample(index, N_exp)
        output_data = input_data[index_re]

    return output_data

def Select_Resample(X, Y, label_list, N_exp):
    """
    X: general ndarray data
    Y: target (one-hot encoder or soft label), shape=(N,K), where N=sampel size, K=#{class}
    N_exp: resample size
    label_list=[4/5, 1/5, 0]
    """
    if N_exp >0:
        data_OS = Select_subset(X, Y, label_list=label_list)
        data_OS = Resample(data_OS, N_exp = N_exp)
        label_OS_m = np.array([label_list]*N_exp)
    else:
        data_OS = np.zeros((0,300,300,3))
        label_OS_m = np.zeros((0, 3))
    #output: [data_OS, label_OS_m]
    return data_OS, label_OS_m





# %%
#                                                        I.  CutMix preprocessing

# %%
class CutMix():
    '''
    Source: Hsieh, H. C., Chiu, Y. W., Lin, Y. X., Yao, M. H. & Lee, Y. J., “Local Precipitation Forecast with LSTM for Greenhouse Environmental Control,” 
    presented at The International Conference on Pervasive Artificial Intelligence (ICPAI2020), Taipei, Taiwan 2020.
    
    Input:
        batch_size: number of samples used each time for generateing sythesized data with the same CutMix window
        mode: "Partial" or "Full" (only partial or Full data are sythesized with CutMix)
        Input: X_train data with type='np.ndarray', shape = (N,T,p)  (N=N0+N1)
        target: Y_train data (binary class) with type='np.ndarray', shape = (N, )
            Note: class 1 of target(y) is assumed to be the minor class
    Output:
        X_train_CutMix: Synthesized X_train data with type='np.ndarray', shape = (2*N0,T,p)
        Y_train_CutMix: Synthesized Y_train data (binary class) with type='np.ndarray', shape = (2*N0, )
'''
    def __init__(self, batch_size = 1, mode =  'Partial'):
        self.batch_size = batch_size
        self.mode = mode
        
    def CutMix_batchwise(self, Input, target, Input_s, target_s ): 
        target = target.astype('float64')
        target_s = target_s.astype('float64')
        #print('dtypes of target and target_s: ', target.dtype, target_s.dtype)
        
        N_plus = len(Input)
        scans = int(N_plus/self.batch_size)
        #for X.shape = (N, T, p, 3), W:= T, H:=p
        W= Input.shape[1]
        H= Input.shape[2]
        
        if W==1 and H>1:
            print('In case of W(Input.shape[1])==1 and H(Input.shape[2])>1')
            for s in range(scans):
                lamda = np.random.uniform(0,1)
                #randomly select index in Input[i] (i.e. Input[:, 0, r_y]) as the centor of the window
                r_y = random.randint(0, H-1)
                #randomly define the Height of the row window (for CutMix)
                r_H = H * (1-lamda)
                #define the range (Height) of the window (for CutMix) (i.e. Input[:, 0, y1:y2])
                y1 = int(round(max(0, r_y-r_H/2),0))
                y2 = int(round(min(H, r_y+r_H/2),0))
                
                #Replace the selected window from Input with the one from Input_s
                Input[(s*self.batch_size):((s+1)*self.batch_size), 0, y1:y2, :] = Input_s[(s*self.batch_size):((s+1)*self.batch_size), 0, y1:y2, :]
                #Adjust lambda to the exact area ratio
                lamda = 1-(y2-y1)/H
                #print('{}-th lamda= {}'.format(s, lamda))
                #intrapolate the class from target with the one from target_s based on exact ratio: lambda
                target[(s*self.batch_size):((s+1)*self.batch_size)] = lamda * target[(s*self.batch_size):((s+1)*self.batch_size)] + (1-lamda)* target_s[(s*self.batch_size):((s+1)*self.batch_size)]
                #print(target[(s*self.batch_size):((s+1)*self.batch_size)])
        
            if (scans*self.batch_size) < N_plus:
                Input[((s+1)*self.batch_size):, 0, y1:y2, :] = Input_s[((s+1)*self.batch_size):, 0, y1:y2, :]
                target[((s+1)*self.batch_size):] = lamda * target[((s+1)*self.batch_size):] + (1-lamda)* target_s[((s+1)*self.batch_size):]
            
        elif H==1 and W>1:
            print('In case of W(Input.shape[1])==1 and H(Input.shape[2])>1')
            for s in range(scans):
                lamda = np.random.uniform(0,1)
                #randomly select index in Input[i] (i.e. Input[:, r_x, 0]) as the centor of the window
                r_x = random.randint(0, W-1)
                #randomly define the Width of the column window (for CutMix)
                r_W = W * (1-lamda)
                #define the range (Width) of the window (for CutMix) (i.e. Input[:, x1:x2, 0])
                x1 = int(round(max(0, r_x-r_W/2),0))
                x2 = int(round(min(W, r_x+r_W/2),0))
                
                #Replace the selected window from Input with the one from Input_s
                Input[(s*self.batch_size):((s+1)*self.batch_size), x1:x2, 0, :] = Input_s[(s*self.batch_size):((s+1)*self.batch_size), x1:x2, 0, :]
                #Adjust lambda to the exact area ratio
                lamda = 1-(x2-x1)/W
                #print('{}-th lamda= {}'.format(s, lamda))
                #intrapolate the class from target with the one from target_s based on exact ratio: lambda
                target[(s*self.batch_size):((s+1)*self.batch_size)] = lamda * target[(s*self.batch_size):((s+1)*self.batch_size)] + (1-lamda)* target_s[(s*self.batch_size):((s+1)*self.batch_size)]
                #print(target[(s*self.batch_size):((s+1)*self.batch_size)])
        
            if (scans*self.batch_size) < N_plus:
                Input[((s+1)*self.batch_size):, x1:x2, 0, :] = Input_s[((s+1)*self.batch_size):, x1:x2, 0, :]
                target[((s+1)*self.batch_size):] = lamda * target[((s+1)*self.batch_size):] + (1-lamda)* target_s[((s+1)*self.batch_size):]
                            
        elif W> 1 and H>1:       
            for s in range(scans):
                lamda = np.random.uniform(0,1)
                #randomly select index in Input[i] (i.e. Input[:, r_x, r_y, :]) as the centor of the window
                r_x = random.randint(0, W-1)
                r_y = random.randint(0, H-1)
                #randomly define the Width and Height of the window (for CutMix)
                r_W = W * np.sqrt(1-lamda)
                r_H = H * np.sqrt(1-lamda)
                #define the range (Width and Height) of the window (for CutMix) (i.e. Input[:, x1:x2, y1:y2])
                x1 = int(round(max(0, r_x-r_W/2),0))
                x2 = int(round(min(W, r_x+r_W/2),0))
                y1 = int(round(max(0, r_y-r_H/2),0))
                y2 = int(round(min(H, r_y+r_H/2),0))
                
                #Replace the selected window from Input with the one from Input_s
                Input[(s*self.batch_size):((s+1)*self.batch_size), x1:x2, y1:y2, :] = Input_s[(s*self.batch_size):((s+1)*self.batch_size), x1:x2, y1:y2, :]
                #Adjust lambda to the exact area ratio
                lamda = 1-((x2-x1)*(y2-y1))/(W*H)
                #print('{}-th lamda= {}'.format(s, lamda))
                #print('Before: {}-th target & = \t{} vs {}'.format(s, target[(s*self.batch_size):((s+1)*self.batch_size)], target_s[(s*self.batch_size):((s+1)*self.batch_size)]))
                #intrapolate the class from target with the one from target_s based on exact ratio: lambda
                target[(s*self.batch_size):((s+1)*self.batch_size)] = lamda * target[(s*self.batch_size):((s+1)*self.batch_size)] + (1-lamda)* target_s[(s*self.batch_size):((s+1)*self.batch_size)]
                #print('After: {}-th target= \t{}'.format(s, target[(s*self.batch_size):((s+1)*self.batch_size)]))

        
            if (scans*self.batch_size) < N_plus:
                #print('Before: {}-th target & = \t{} vs {}'.format(s, target[(s*self.batch_size):((s+1)*self.batch_size)], target_s[(s*self.batch_size):((s+1)*self.batch_size)]))
                Input[((s+1)*self.batch_size):, x1:x2, y1:y2, :] = Input_s[((s+1)*self.batch_size):, x1:x2, y1:y2, :]
                target[((s+1)*self.batch_size):] = lamda * target[((s+1)*self.batch_size):] + (1-lamda)* target_s[((s+1)*self.batch_size):]
                #print('After: {}-th target= \t{}'.format(s, target[(s*self.batch_size):((s+1)*self.batch_size)]))
        else:
            print("In case of Input.shape = (N,1,1), the input data (Input and target will be returned as output, which is the same as random Over-Sampling technique to minor class)")
            
        return  Input, target 
            
            
            
    
    def fit_resample(self, X_train, Y_train):   
        if self.mode == 'Partial':                               
            index0 = np.where(Y_train == 0)[0].tolist()
            index1 = np.where(Y_train == 1)[0].tolist()
            N0 = len(index0)
            N1 = len(index1)
            N_plus = N0 - N1 #Since class 1 is assumed a minor class, N0>N1.
            #Random-Oversample the minor class (with replacement), 
            #and Random-Undersample the major class ((without replacement))
            index0_re =  random.sample(index0, N_plus)
            index1_re = random.choices(index1, k=N_plus)
            
            #Output Input, target (for minor class) & Input_s, target_s (for major class)
            Input = X_train[index1_re]
            target = np.ones(N_plus) #Y_train[index1_re]
            Input_s = X_train[index0_re]
            target_s  = np.zeros(N_plus)
            
            #Use CutMix to synthesize new data from Input (target) as well as Input_s (target)
            Input, target = self.CutMix_batchwise( Input, target, Input_s, target_s)             
            
            #Now concate the new synthesized data to the original X_train, Y_train
            X_train_CutMix = np.concatenate([X_train, Input])
            Y_train_CutMix = np.concatenate([Y_train, target])
            
            #Shuffle the order 
            index = np.arange(N0*2)
            random.shuffle(index)            
            X_train_CutMix = X_train_CutMix[index]
            Y_train_CutMix  = Y_train_CutMix[index]
            
        elif self.mode == 'Full':                            
            index0 = np.where(Y_train == 0)[0].tolist()
            index1 = np.where(Y_train == 1)[0].tolist()
            N0 = len(index0)
            N1 = len(index1)
            #Random-Oversample the minor class (with replacement), 
            index1_re = random.choices(index1, k=N0)            
            
            #Output Input, target (for minor class) & Input_s, target_s (for major class)
            Input = np.concatenate([X_train[index0], X_train[index1_re]])
            target = np.concatenate([np.zeros(N0), np.ones(N0)])
            
            #Shuffle the order of Input(target)
            index = np.arange(N0*2)
            random.shuffle(index)
            
            Input_s = Input[index]
            target_s  = target[index]
            
            #Use CutMix to synthesize new data from Input (target) as well as Input_s (target)
            Input, target = self.CutMix_batchwise( Input, target, Input_s, target_s) 
            #Now concate the new synthesized data to the original X_train, Y_train
            X_train_CutMix = Input
            Y_train_CutMix = target            
            
        else:
            print('Error message: Please specify the argument "mode" either by "Partial" or "Full".')
            X_train_CutMix = []
            Y_train_CutMix = []  
                                    
        return X_train_CutMix, Y_train_CutMix


def CutMix_Prep(train_skf_data, train_skf_label, state, model_resize=(300,300)):
    """
    This is specifically designed for "binary classification!"
    train_skf_data, train_skf_label: training data with its labels (X, y)
    state: = {0,1,2} Shuffle Method 0~2
        (1) 0: Typical CutMix 
        (2) 1: Same label of data is CutMix-operated with same label
        (3) 2: Each label of data is CutMix-operated with Opposite label
        
    Source paper (for typical CutMix): 
        Yun, Sangdoo, et al. "Cutmix: Regularization strategy to train strong classifiers with localizable features." 
        Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    """
    #Important:label must be float rather that int to implement CutMix
    train_skf_label = train_skf_label.astype('float') 
    
    # Extract data for each label (0, 1)
    train_0_data = train_skf_data[train_skf_label==0]
    train_1_data = train_skf_data[train_skf_label==1]


# %% 0. Please select only 1 out of 3 shuffling methods of CutMix operation (and comment out the rest of methods)

    # %% Shuffle Method 0 (random sample CutMix-operated)
    if state == 0:
        index = np.arange(len(train_skf_data))
        random.shuffle(index)
        #random.shuffle(index)
        train_skf_data_s = train_skf_data[index]
        train_skf_label_s = train_skf_label[index]

    # %% Shuffle Method 1 ( Same label of data is CutMix-operated with same label)
    elif state == 1:
        train_skf_data = np.concatenate([train_0_data, train_1_data], 0)
        train_skf_label = np.concatenate([np.zeros(len(train_0_data)), np.ones(len(train_1_data))]) #default - dtype: float64
        
        #1. shuffle order of train_0_data [label=0] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_0_data))
        random.shuffle(index)
        train_0_data_s = train_0_data[index]

        #2. shuffle order of train_1_data [label=1] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_1_data))
        random.shuffle(index)
        train_1_data_s = train_1_data[index]

        train_skf_data_s = np.concatenate([train_0_data_s, train_1_data_s], 0)
        train_skf_label_s = train_skf_label

    # %% Shuffle Method 2 ( Each label of data is CutMix-operated with Opposite label)
    elif state == 2:
        train_skf_data = np.concatenate([train_0_data, train_1_data], 0)
        train_skf_label = np.concatenate([np.zeros(len(train_0_data)), np.ones(len(train_1_data))], 0) #default - dtype: float64
        
        #1. shuffle order of train_0_data [label=0] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_0_data))
        random.shuffle(index)
        train_0_data_s = train_0_data[index]

        #2. shuffle order of train_1_data [label=1] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_1_data)) 
        random.shuffle(index)
        train_1_data_s = train_1_data[index]

        train_skf_data_s = np.concatenate([train_1_data_s, train_0_data_s], 0)
        train_skf_label_s = np.concatenate([np.ones(len(train_1_data)), np.zeros(len(train_0_data))], 0) #Default - dtype: np.float64
        
    else:
        return print(f"value of state must be 0~3")

#train_skf_data = data_CutMix 
#train_skf_label = label_CutMix
#Generalized version of function "CutMix_Prep" (i.e., label is not necessary {0, 1}, but any 2 different numbers)
def CutMix_Prep_gen(train_skf_data, train_skf_label, state, model_resize=(300,300)):
    """
    This is specifically designed for "binary classification!"
    train_skf_data, train_skf_label: training data with its labels (X, y)
    state: = {0,1,2} Shuffle Method 0~2
        (1) 0: Typical CutMix 
        (2) 1: Same label of data is CutMix-operated with same label
        (3) 2: Each label of data is CutMix-operated with Opposite label
        
    Source paper (for typical CutMix): 
        Yun, Sangdoo, et al. "Cutmix: Regularization strategy to train strong classifiers with localizable features." 
        Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
    """
    #Important:label must be float rather that int to implement CutMix
    train_skf_label = train_skf_label.astype('float') 
    
    # extract unique labeling values
    labels = np.unique(train_skf_label)
    
    if len(labels)!= 2:
        return print("number of labels should only be two (since CutMix is for binary data)")
    
    # Extract data for each label (0, 1)
    train_0_data = train_skf_data[train_skf_label==labels[0]]
    train_1_data = train_skf_data[train_skf_label==labels[1]]


# %% 0. Please select only 1 out of 3 shuffling methods of CutMix operation (and comment out the rest of methods)

    # %% Shuffle Method 0 (random sample CutMix-operated)
    if state == 0:
        index = np.arange(len(train_skf_data))
        random.shuffle(index)
        #random.shuffle(index)
        train_skf_data_s = train_skf_data[index]
        train_skf_label_s = train_skf_label[index]

    # %% Shuffle Method 1 ( Same label of data is CutMix-operated with same label)
    elif state == 1:
        train_skf_data = np.concatenate([train_0_data, train_1_data], 0)
        train_skf_label = np.concatenate([np.zeros(len(train_0_data))+labels[0], np.zeros(len(train_1_data))+labels[1]]) #default - dtype: float64
        
        #1. shuffle order of train_0_data [label=labels[0]] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_0_data))
        random.shuffle(index)
        train_0_data_s = train_0_data[index]

        #2. shuffle order of train_1_data [label=1] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_1_data))
        random.shuffle(index)
        train_1_data_s = train_1_data[index]

        train_skf_data_s = np.concatenate([train_0_data_s, train_1_data_s], 0)
        train_skf_label_s = train_skf_label

    # %% Shuffle Method 2 ( Each label of data is CutMix-operated with Opposite label)
    elif state == 2:
        train_skf_data = np.concatenate([train_0_data, train_1_data], 0)
        train_skf_label = np.concatenate([np.zeros(len(train_0_data))+labels[0], np.zeros(len(train_1_data))+labels[1]], 0) #default - dtype: float64
        
        #1. shuffle order of train_0_data [label=0] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_0_data))
        random.shuffle(index)
        train_0_data_s = train_0_data[index]

        #2. shuffle order of train_1_data [label=1] (for random sampling without replacement before CutMix)
        index = np.arange(len(train_1_data)) 
        random.shuffle(index)
        train_1_data_s = train_1_data[index]

        train_skf_data_s = np.concatenate([train_1_data_s, train_0_data_s], 0)
        train_skf_label_s = np.concatenate([np.zeros(len(train_1_data))+labels[1], np.zeros(len(train_0_data))+labels[0]], 0) #Default - dtype: np.float64
        
    else:
        return print(f"value of state must be 0~3")


# %% 1. CutMix Operation
    

    train_skf_data0, train_skf_label0 = train_skf_data.copy(), train_skf_label.copy()

    
    CutMix_P = CutMix(batch_size = 1, mode =  'Partial')
    train_skf_data_CutMix, train_skf_label_CutMix = CutMix_P.CutMix_batchwise(train_skf_data0, train_skf_label0, train_skf_data_s, train_skf_label_s)
    #train_skf_label_CutMix element shows the proportion of label 1 (i.e.,train_N_label) on the image
    
    # Resize images to match the requirement of the subsequent modeling
    if model_resize != None:
        train_skf_data_CutMix2 = []
        for img in train_skf_data_CutMix:  
            img = cv2.resize(img, model_resize)
            train_skf_data_CutMix2.append(img)     
        train_skf_data_CutMix = np.array(train_skf_data_CutMix2)
        

    '''
    np.save(os.path.join(train_path, f'train_skf_data_with_CutMix_{str(state)}.npy'), train_skf_data_CutMix)
    np.save(os.path.join(train_path, f'train_skf_label-{label1}_with_CutMix_{str(state)}.npy'), train_skf_label_CutMix)
    #load npy file
    train_skf_data_CutMix = np.load('train_skf_data_with_CutMix.npy')
    train_skf_label_CutMix = np.load(f'train_skf_label-{label1}_with_CutMix.npy)
    '''

    return train_skf_data_CutMix, train_skf_label_CutMix
# %%









# %%
#                                                        II.  Puzzle preprocessing

# %%

# %% Function 1. Puzzle Preprocessing for each image
# %%
#image = train_skf_data[0]
#len_crop = 400

def Puzzle(image, len_crop):
    """
    input: np.array; original image (image), and length of the puzzle
    output: np.arry; puzzled image which has the same size as original one
    """
    raw_size = image.shape[:2] #(height, width)
    if raw_size[0] % len_crop == 0:
        height_index = int(raw_size[0]/ len_crop)
    else:  
        print("The residual of height divided by len_crop must be 0!") # return
        #break
        
    if raw_size[1] % len_crop == 0:
        width_index = int(raw_size[1]/ len_crop)
    else:
        print("The residual of width divided by len_crop must be 0!") 
        #break
        
    img_crop = []
    
    for i in range(height_index):
        for j in range(width_index):
            #print(i, j)
            crop = image[len_crop*i:len_crop*(i+1), len_crop*j:len_crop*(j+1)]
            img_crop.append(crop)
            
    img_crop = np.array(img_crop)
            
    #Randomize all crops in img_crop
    index = np.arange(len(img_crop))
    random.shuffle(index)
    img_crop = img_crop[index]
    
    #Paste puzzled crops to the image
    index = 0
    image_puzzled = image.copy()
    for i in range(height_index):
        for j in range(width_index):
            image_puzzled[len_crop*i:len_crop*(i+1), len_crop*j:len_crop*(j+1)] = img_crop[index]
            index += 1
                        
    return image_puzzled

# %% Function 2. Puzzle Preprocessing for training set
# %%
def Puzzle_Prep(train_skf_data, len_crop, model_resize=(300,300)):
    """
    @Purpose: Puzzle operation to the training dataset (X, y) = (train_skf_data, train_skf_label)
    Input:
        len_crop: length (height as well as width) of puzzle pieces
        resize: if to resize the Puzzle-operated training dataset for output; say, (300, 300) for modeling, or None
    Output:
        train_skf_data_puz: training dataset after Puzzle operation (note that train_skf_label_puz = train_skf_label)
    """
    train_skf_data_puz = []
    for img in train_skf_data:        
        img_puz = Puzzle(img, len_crop)
        if model_resize != None:
            img_puz = cv2.resize(img_puz, model_resize)
            
        train_skf_data_puz.append(img_puz)
        
    train_skf_data_puz = np.array(train_skf_data_puz)
    
    return train_skf_data_puz
# %%


# %%
#                                  III. preprocessing with SMOTE (with Mixedup): to build artificial soft labels

# %%
from sklearn.neighbors import NearestNeighbors

def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    y: np.array, label (target) with shape = (N,)
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    H, W, C = X.shape[1:]
        
    N = X.shape[0]
    
    X = X.reshape(N,-1)

    #X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
    euclidean,indices= nbs.kneighbors(X)
    '''
    euclidean: distances of each instance of X to its (5) nearest neighbors; shape = (sample_size, n_neighbors(5))
    indices: indices of each instance of X to its (5) nearest neighbors; shape = (sample_size, n_neighbors(5))
    '''
    return indices

    
    
def cross_nearest_neighbour(X_0, X_1):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X_0, X_1: numpy.ndarray with shape = (N, H, W, C); X_0 are (reference) data for synthesis and X_1 are for finding neighbors
    y_0, y_1: numpy.ndarray, label (target) with shape = (N,)
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    H, W, C = X_0.shape[1:]    
    N_0 = X_0.shape[0];
    N_1 = X_1.shape[0];
    
    X_0 = X_0.reshape(N_0,-1)
    X_1 = X_1.reshape(N_1,-1)

    #X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X_1)
    euclidean,indices= nbs.kneighbors(X_0)
    """
    euclidean: distances of each instance of X to its (5) nearest neighbors; shape = (sample_size, n_neighbors(5))
    indices: indices of each instance of X to its (5) nearest neighbors; shape = (sample_size, n_neighbors(5))
    """

    return indices


'''
X = data_SMOTE
y = label_SMOTE
n_sample = 120
'''
def SMOTE_SoftLabel(X, y, state, n_sample="default"):
#if True:
    """
    Give the augmented data using SMOTE algorithm, and use Mixup to synthesize target y
    
    args
    X: pandas.DataFrame, input vector DataFrame X=[X_0; X_1] where X_0 are (reference) data for synthesis and X_1 are for finding neighbors
    y: pandas.DataFrame, feature vector dataframe y=[y_0; y_1]
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    H, W, C = X.shape[1:]
    #Important:label must be float rather that int to implement CutMix
    y = y.astype('float') 
    
    if state == 0:
        print("Generate synthetic data with conventional SMOTE (target synthesized with all neighbors)")
        
        indices = nearest_neighbour(X)
        n = len(indices) # len(X)
        '''
        plt.hist(indices.flatten(), bins = 192)
        plt.show()
        '''       
        if n_sample=="default":
            n_sample = n
        new_X = np.zeros((n_sample, H, W, C))
        target = np.zeros((n_sample))
        for i in range(n_sample):
            reference = random.randint(0,n-1) # random select ane instance from X -> called reference
            neighbour = random.choice(indices[reference,1:]) # find the indices of (5-1) NNs from the reference (excluding itself), then randomly pick one index (NN) out
            all_point = indices[reference] # pick all indices of (5) NNs from the reference (including itself)
            nn_array = y[all_point] # pick corresponding targets (y) from all indices of (5) NNs from the reference (including itself)
            target[i] = nn_array.mean()

            ratio = random.random()
            gap = X[neighbour,:] - X[reference,:] #corrected already!
            new_X[i] = np.array(X[reference,:] + ratio * gap) #x + r*(x_nn - x) = r*x_nn + (1-r)*x 
    
    elif state == 1:
        print("Generate synthetic data across distince labels (so as to derive soft labels)")
        # extract unique labeling values
        labels = np.unique(y)
    
        # Extract data for each label (0, 1)
        X_0 = X[y==labels[0]]; y_0 = y[y==labels[0]]
        X_1 = X[y==labels[1]]; y_1 = y[y==labels[1]]
    
        indices = cross_nearest_neighbour(X_0, X_1)
        n = len(indices) # len(X)

        if n_sample=="default":
            n_sample = n
        new_X = np.zeros((n_sample, H, W, C))
        target = np.zeros((n_sample))
        for i in range(n_sample):
            reference = random.randint(0,n-1) # random select ane instance from X -> called reference
            neighbour = random.choice(indices[reference,0:]) # find the indices of (5-0) NNs from the reference (excluding itself), then randomly pick one index (NN) out
            ratio = random.random()
            gap_X = X_1[neighbour,:] - X_0[reference,:] 
            gap_y = y_1[neighbour] - y_0[reference] 

            new_X[i] = np.array(X_0[reference,:] + ratio * gap_X) #x + r*(x_nn - x) = r*x_nn + (1-r)*x                 
            target[i] = np.array(y_0[reference] + ratio * gap_y)  #y + r*(y_nn - y) = r*y_nn + (1-r)*y   

    return new_X, target



# %%
#                                                    DownSampling(DS) Preprocessing 

# %%
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks    
# %% 1. Select subset data (X, y) (could be multi-class) and apply DS
def DS_Prep(data_list, target_list, label_lists, img_shape=(300,300,3), sampling_strategy="all", n_neighbors=3):
    '''
    Default DS: ENN (Edited Nearest Neighbours)
    Input: e.g.,
        data_list = [train_skf_data_INTER, train_skf_data_INTER]
        target_list = [train_label_INTER, train_label_INTER] (one-hot encoder or soft label array)
        label_lists = [[2/3, 1/3, 0], [1/3, 2/3, 0]] #selected data based on label lists
    Output: the down-sampled data, of which is transformed into input ready for CNN analysis
    '''
    #Select assigned data
    data_concat_flat, target_concat, N_concat = Select_subset_flatten(data_list, target_list, label_lists )
    print("Before:", Counter(target_concat))
    ## Under-Sampling: ENN
    undersampler = EditedNearestNeighbours(sampling_strategy=sampling_strategy, n_neighbors=3)                #{1: N_exp + len(data_SMOTE01)}
    #undersampler = S2.TomekLinks()
    data_concat_flat, target_concat = undersampler.fit_resample(data_concat_flat, target_concat)
    N_concat = Counter(target_concat)
    print("After:", N_concat)
    #reshape data X for CNN input
    data_concat, target_concat_m = ReshapeX_TransY(data_concat_flat, target_concat, label_lists, img_shape=img_shape)
    
    return data_concat, target_concat_m, N_concat
