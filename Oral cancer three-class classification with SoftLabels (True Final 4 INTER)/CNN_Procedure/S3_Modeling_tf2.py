# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Jul 12 17:24:27 2021

@author: user
"""
import numpy as np


## for modeling (updated)
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import efficientnet.tfkeras as efn
from tensorflow.keras import backend as K


# For learning rate scheduling
from tensorflow.keras.callbacks import LearningRateScheduler

'''
#How to fix "AttributeError: module 'tensorflow' has no attribute 'get_default_graph'"?
https://stackoverflow.com/questions/55496289/how-to-fix-attributeerror-module-tensorflow-has-no-attribute-get-default-gr

Replace all keras.something.something with tensorflow.keras.something, and use:

import tensorflow as tf
from tensorflow.keras import backend as k

'''



# %%
#                                                0. create model structure

# %%
def create_EfficientNetB3(n_class, Loss='categorical_crossentropy', Activation ='softmax'):
    """
    Loss: loss function; e.g., 'categorical_crossentropy', (S3.)FocalLoss(gamma, alpha_list)
    activation: 'softmax'
    """
    pre_model = efn.EfficientNetB3(include_top = False, weights='imagenet', pooling='avg')
    x = pre_model.output
    output_layer = Dense(n_class, activation = Activation)(x) #
    model = Model(inputs = pre_model.input, outputs = output_layer)
    #parallel_model = multi_gpu_model(model, gpus = 2, cpu_merge = False)

    for layer in model.layers[:-2]:
        layer.trainable = True
    for layer in model.layers[-2:]:
        layer.trainable = True 

    # print out model structure if needed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    for x in model.trainable_weights:
        print(x.name)
    '''
    #model.summary() 
    
    #parallel_model = multi_gpu_model(model, gpus = 2, cpu_merge = False)
    #parallel_model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(), loss = Loss, metrics = ['accuracy'])
    
    return model



# %%
#                                  I. for LearningRateScheduler (inserted during model.fit / model.fit_generator)

# %%
def step_decay(epoch):     
    if (epoch < 5):
        return(7e-5)
    elif (epoch < 12):
        return(3e-5)
    elif (epoch < 20):
        return(1e-5)
    else:
        return(1e-6)
        

def call_LearningRateScheduler(step_decay, verbose=1):
    """
    #Purpose: for model fitting: generate data sequences

    Input:
        step_decay: function to define learning rate schedule
        verbose: =1 -> broadcast on IPython console
    Output:
        callbacks_list: List; callback list for following input:
            
                model_history = model.fit_generator(
                        datagen.flow(train_skf_data, train_skf_label, batch_size = batch_size),
                        steps_per_epoch = len(train_skf_indices) / batch_size,
                        validation_data = (val_skf_data, val_skf_label),
                        epochs = epochs,
                        verbose = 1,
                        callbacks = callbacks_list
                        )
    """    
    lrate = LearningRateScheduler(step_decay, verbose=verbose)
    callbacks_list = [lrate] 
    return callbacks_list
     
     
        



# %%
#                                  II. for custom loss function

# %%
#gamma = 2
#alpha_list = [0.95, 0.05] #[0.05, 0.55, 0.4]

def FocalLoss(gamma, alpha_list):
    def focal_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        weight_array = np.multiply(y_true, tf.convert_to_tensor(alpha_list))
        return K.sum(weight_array * K.pow(1-y_pred, gamma) * -K.log(y_pred))
    
    return focal_loss









"""
#%% codes in  main.py
    #%% IV. Model Training
    # call model
    model = create_EfficientNetB3()
    
    # for model fitting: generate data sequences
    lrate = LearningRateScheduler(step_decay, verbose=1)
    callbacks_list = [lrate]
    
    datagen = ImageDataGenerator(
                                width_shift_range = 0.0,     # 水平平移
                                height_shift_range = 0.0,    # 垂直平移
                                rotation_range = 90,         # 0-180 任一角度旋轉
                                horizontal_flip = True,      # 任意水平翻轉
                                vertical_flip = True,        # 任意垂直翻轉
                                fill_mode = "constant",      # 在旋轉或平移時，有空隙發生，則空隙補常數
                                cval = 0                     # 設定常數值為 0
                                )
  
    # Preprocess (if any) to k-fold validation dataset
    val_skf_data = S1.Resize_Prep(train_data[val_indices], model_resize) # ready for model input 
    val_skf_label = np.eye(2)[train_label[val_indices]] 
    
    # for model fitting: training with data sequences
    model_history = model.fit_generator(
            datagen.flow(train_skf_data, train_skf_label, batch_size = batch_size),
            steps_per_epoch = len(train_skf_indices) / batch_size,
            validation_data = (val_skf_data, val_skf_label),
            epochs = epochs,
            verbose = 1,
            callbacks = callbacks_list
            )
"""

# %%
