# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 12:50:08 2021

@author: Jane (Hsing-Chuan) Hsieh
"""

import cv2
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

#math
import math
import random

from sklearn.base import BaseEstimator

class IScoreCL(BaseEstimator):
    """
    @Purpose: build Interaction-based Convolution Layer (CL) (i.e., CNN conbined with I-Score feature evaluation)
    
    Parameters:
        kernel_size: size of kernel / filter with shape = (kernel_size, kernel_size)
        stride: the number of pixels shifts over the input matrix
        start_point: filter start at firtst elemtnt of array, i.e., (0, 0)
        
    Input:
        X_sub: 2-D array; tabular data of selected exploratory variables from X
        y: 1-D array; tabular data of responses y
        
        
    Output:
        X_dagger_Conv: the interaction-based (I-Score) convolution layers, from each of input data X;
                            shape = (N, H_out, W_out)
        
    Attribute:
        self.IScore_kernel: the trained kernels/filters, 
                            with each kernel recording the seleted X indices info. ('k_index') 
                            and the average of Y in each partition of seleted Xs ('Partitions')
        self.exception: # of times that the produced kernel values from Interaction-based CLs that is out of the range of partition elements;
                        i.e., # of times that 'kernel values = Mean_y '
    """
    
    def __init__(self, kernel_size = 2, stride = 2, start_point = 1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.start_point = start_point
        

    def I_Score(self, X_sub, y):
        '''
        Purpose: Compute I-score according to X_sub
        input:
            X_sub: 2-D array; tabular data of selected exploratory variables from X
            y: 1-D array; tabular data of responses y
        output:
            I-score: standardized influential score measuring the impact of selected X (X_sub) to y
            summary: record sample size ('N_part') and Mean(y) ('Mean_y_part') for each conditional partition of X_sub
        '''
        N = len(X_sub)
        Mean_y = np.mean(y)
        Var_y = np.var(y)
    
        df = pd.DataFrame( np.hstack([X_sub, y[:, np.newaxis]]) )
            
        summary = df.groupby(df.columns[:-1].tolist()).agg([np.size,np.mean])
        summary = summary.astype('float64')
        summary.columns = ['N_part', 'Mean_y_part']
        summary['I_term'] = (summary['N_part']**2) * (summary['Mean_y_part']-Mean_y)**2  #!!!!!!!!!!!!!!!!! precision error?
        # standardized I-Score
        I_Score = summary['I_term'].sum() * (1/(N*Var_y))
    
        return I_Score, summary[['N_part', 'Mean_y_part']]

    def BDA(self, X_sub, y):
        """
        Purpose: Backward Dropping Algorithm for featur selection)
        Input:
            X_sub: 2-D numpy.ndarray; tabular data of selected exploratory variables from X
            y: 1-D numpy.ndarray; 1-D array; tabular data of responses y
        Output:
            record: dictionary; the selected best influential variables indices ('X_s') in a list, as well as it's (maximum) I-Score
        """
        N, k = X_sub.shape # the number of exploratory variables X's, a subset of X={X_1, ..., X_{H*W}} 
        k_index = list(range(k))
        
        df_record = pd.DataFrame([], columns=['X_s', 'Iscore'])

        while len(k_index)>=1:
            # 1 Compute I-score according to X_sub
            Iscore, _ = self.I_Score(X_sub[:, k_index], y)
            
            record = {'X_s': k_index, 'Iscore': Iscore}
            df_record = df_record.append([record], ignore_index=True)
           
            # 2 Drop random 1 X
            k_index = list( set(k_index) - set(random.sample(k_index, 1)) )
            
        #output selected variables
        print('The greedy selection process:\n', df_record)
        argmax = df_record.Iscore.idxmax()
        
        k_index = df_record.X_s[argmax]
        Iscore = df_record.Iscore[argmax]
    
        return k_index, Iscore


    #def I_Score_Conv(self, X, y):  #original name
    def fit(self, X, y):
        """
        Purpose: Train for I-Score Convolution Layers
        input: 
            X: numpy.ndarray; image data (explainatory variables) with size = (N,H,W,3) (if RGB) or (N,H,W) (if Gray-Scale)
            y: 1-D numpy.ndarray; response data for image classification (y:must be binary {0,1}) or for regression (y: discrete or continuous)
            
        Output:
            IScore_kernel: the trained kernels/filters, 
                            with each kernel recording the seleted X indices info. ('k_index') 
                            and the average of Y in each partition of seleted Xs ('Partitions')
                        
        """
    
        # input: X wiht shape=(N, H, W, 1) 
        N, self.H, self.W  = X.shape
        self.Mean_y = np.mean(y)
    
        self.H_out = math.floor((self.H-self.kernel_size -self.start_point+1)/self.stride + 1)
        self.W_out = math.floor((self.W-self.kernel_size -self.start_point+1)/self.stride + 1)

        IScore_kernel = OrderedDict({})
        I_Heatmap = np.empty([self.H_out, self.W_out])

        for i in range(self.H_out):
            for j in range(self.W_out):
                print(f'process: ({i}, {j})-th out of ({self.H_out-1}, {self.W_out-1})')
                #print(f"{(self.stride*i)}:{(self.stride*i+self.kernel_size)}, {(self.stride*j)}:{(self.stride*j+self.kernel_size)}") 
        
                #1 build filter/kernel input for variables selection
                X_sub = X[:, (self.stride*i):(self.stride*i+self.kernel_size), (self.stride*j):(self.stride*j+self.kernel_size)].reshape((N, -1)) # default: row-wise flattening        
        
                #2 Backward Dropping Algorithm for featur selection)        
                k_index, _ = self.BDA(X_sub, y) #;print('selected X variables (indices): ', k_index)        

                #3 train filter/kernel & save results of each (i,j)
                IScore, Partitions = self.I_Score(X_sub[:, k_index], y)        
                ## save trained results for each (i,j)
                IScore_kernel[(i,j)] = OrderedDict({'IScore': IScore, 'k_index': k_index, 'Partitions': Partitions})
                I_Heatmap[i, j] = IScore_kernel[(i, j)]['IScore']        
        
        self.IScore_kernel = IScore_kernel
        self.I_Heatmap = I_Heatmap
        
        '''
        plt.imshow(self.I_Heatmap ) #, cmap='gray'
        plt.colorbar()
        plt.show()
        '''        
        return  self     #self.IScore_kernel , self.Mean_y 

    
        
    def pred(self, X): 
        """
        Purpose: Predict I-Score Convolution Layer
        Input: 
            X: numpy.ndarray; image data (explainatory variables) with size = (N,H,W,3) (if RGB) or (N,H,W) (if Gray-Scale)

        Output:
            X_dagger_Conv: the interaction-based (I-Score) convolution layers, from each of input data X;
                            shape = (N, H_out, W_out)
                        
        """
        # input: X wiht shape=(N, H, W, 1) 
        N, H_1, W_1  = X.shape   
    
        if (self.H != H_1) or (self.W != W_1):
            return print(f'The expected X shape is ({self.H}, {self.W}), which is incompatible with current input data with shape ({H_1}, {W_1})')
    

        exception = 0  
        data = []
    
        for n in range(N):
        
            X_dagger_Conv = np.empty([self.H_out, self.W_out])
            for i in range(self.H_out):
                for j in range(self.W_out):    
                    print(f'process of sample {n}: ({i}, {j})-th out of ({self.H_out-1}, {self.W_out-1})')                    
                    #1 build filter/kernel input for variables selection
                    X_sub = X[n, (self.stride*i):(self.stride*i+self.kernel_size), (self.stride*j):(self.stride*j+self.kernel_size)].reshape((1, -1)) # default: row-wise flattening  
                
                    k_index = self.IScore_kernel[(i, j)]['k_index']
                    k_index_y = tuple( list(X_sub[0, k_index]) )
                
                    try:
                        X_dagger = self.IScore_kernel[(i, j)]['Partitions'].loc[ k_index_y, 'Mean_y_part']
                    except:
                        X_dagger = self.Mean_y
                        exception += 1 
                    
                    X_dagger_Conv[i, j] = X_dagger
        
            data.append(X_dagger_Conv)
        
        data = np.array(data)
        self.exception = exception
    
        return data
    
    def save_weights(self, path):
        '''
        @Purpose: to save the traind model
        input:
            path: (output directory +) filename; e.g., os.path.join(savepath, 'IScore_kernel.json')
        '''
        Dict = OrderedDict({'IScore_kernel': self.IScore_kernel, 'Mean_y': self.Mean_y})
        with open(path, 'wb') as outF:
            
            pickle.dump(Dict, outF)
            
    def pred_new(self, X, Mean_y, IScore_kernel): 
        """
        Purpose: Load training model (i.e., dictionary of {'IScore_kernel', 'Mean_y'}) & Predict I-Score Convolution Layer for new data
        Input: 
            X: numpy.ndarray; image data (explainatory variables) with size = (N,H,W,3) (if RGB) or (N,H,W) (if Gray-Scale)

        Output:
            X_dagger_Conv: the interaction-based (I-Score) convolution layers, from each of input data X;
                            shape = (N, H_out, W_out)
                        
        """
        # input: X wiht shape=(N, H, W, 1) kernel_size=2, stride=2
        N, H, W  = X.shape

        H_out = math.floor((H-kernel_size -start_point+1)/stride + 1)
        W_out = math.floor((W-kernel_size -start_point+1)/stride + 1)

        H_out_0, W_out_0 = list(IScore_kernel.keys())[-1]
        H_out_0 +=1
        W_out_0 +=1    
    
        if (H_out_0 != H_out) or (W_out_0 != W_out):
            return print(f'The expected Conv shape is ({H_out_0}, {W_out_0}) while deirving ({H_out}, {W_out}), which is incompatible with input data and kernel settings')

        exception = 0  
        data = []
    
        for n in range(N):
        
            X_dagger_Conv = np.empty([self.H_out, self.W_out])
            for i in range(self.H_out):
                for j in range(self.W_out):    
                    print(f'process of sample {n}: ({i}, {j})-th out of ({self.H_out-1}, {self.W_out-1})')                    
                    #1 build filter/kernel input for variables selection
                    X_sub = X[n, (self.stride*i):(self.stride*i+self.kernel_size), (self.stride*j):(self.stride*j+self.kernel_size)].reshape((1, -1)) # default: row-wise flattening  
                
                    k_index = self.IScore_kernel[(i, j)]['k_index']
                    k_index_y = tuple( list(X_sub[0, k_index]) )
                
                    try:
                        X_dagger = self.IScore_kernel[(i, j)]['Partitions'].loc[ k_index_y, 'Mean_y_part']
                    except:
                        X_dagger = self.Mean_y
                        exception += 1 
                    
                    X_dagger_Conv[i, j] = X_dagger
        
            data.append(X_dagger_Conv)
        
        data = np.array(data)
    
        return data


"""
#%%                                      II-1. Train for I-Score Convolution Layers
#%%
kernel_size = 2
stride = 2
start_point = 1
'''
savepath = os.path.join(savepath, f'w{kernel_size}-s{stride}-sp{start_point}-DataAugment')
if not os.path.exists(savepath):
    os.mkdir(savepath) 
'''
# model training for ICL   
model = S3_ICL.IScoreCL(kernel_size = kernel_size, stride = stride, start_point = start_point)    
model.fit(train_data, train_label)

#IScore_kernel: the trained kernels/filters after fitting
IScore_kernel = model.IScore_kernel

## Illustration of I-Score Heatmap trained from training dataset(after fitting)
I_Heatmap = model.I_Heatmap
plt.imshow(I_Heatmap ) #, cmap='gray'
plt.colorbar()
plt.title('I-Score Heatmap')
plt.show()
plt.savefig(os.path.join(savepath, 'I-Score_Heatmap.png'), transparent=True)


## Save training weights - dictionary of {'IScore_kernel', 'Mean_y'}
model.save_weights(os.path.join(savepath, 'Mean_y and IScore_kernel.json'))


#%%                                      II-2. prediction with built filters 
#%%
# produce Interaction-based CL 
train_Conv1 = model.pred(train_data)
val_Conv1 = model.pred(val_data)
test_Conv1 = model.pred(test_data)



#%% 
#                                       IV. Model prediction on new testing set

#%%
## Load training weights - dictionary of {'IScore_kernel', 'Mean_y'}
import pickle
with open(os.path.join(savepath, 'Mean_y and IScore_kernel.json'), 'rb') as inF:
    weights = pickle.load(inF)

IScore_kernel = weights['IScore_kernel']
Mean_y = weights['Mean_y']

model2 = S3_ICL.IScoreCL(kernel_size = 2, stride = 2, start_point = 1)  
train_Conv2 = model2.pred_new(test_data[:2], Mean_y, IScore_kernel)

## Save dictionary - IScore_kernel
with open(os.path.join(savepath, 'IScore_kernel.json'), 'wb') as outF:
    pickle.dump(IScore_kernel, outF)

"""
  

