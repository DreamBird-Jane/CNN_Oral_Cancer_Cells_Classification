# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Aug  2 16:44:19 2021

@author: user

@Resource:
    1. to load model with customized loss functions:
        https://stackoverflow.com/questions/57982158/valueerror-unknown-loss-functionfocal-loss-fixed-when-loading-model-with-my-cu
        https://github.com/keras-team/keras/issues/5916
    2. **Grad-CAM source code:
        https://github.com/eclique/keras-gradcam
        https://github.com/eclique/keras-gradcam/blob/master/gradcam_vgg.ipynb
"""




# %% O. Grad-CAM & Guided Grad-CAM
!/usr/bin/python -m pip install --upgrade pip
# !pip install -U scikit-learn 
# !pip install Keras==2.2.4
# !pip install tensorflow==1.13.1
# !pip install tensorflow-gpu==1.13.1
# !pip install opencv-python-headless # !pip install opencv-python
# %% O.0  Build model
# !pip install fastai
# !pip install joblib
# !pip install python-gdcm
# !pip install scikit-image
# %%
# !pip install efficientnet
# troubleshooting: compatible with keara 2.2.4
# !pip install 'h5py==2.10.0'

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
#from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.python.framework import ops
# %%
def build_model(path, modelh5):
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    Input:
        path: folder where the model is located
        modelh5: model file name (.h5); 'EfficientNetB3_NoneNone_b16_e5_t1_K3_a0.82_g5.h5'
    """ 
    # model use some custom objects, so before loading saved model
    # import module your network was build with
    # e.g. import efficientnet.keras / import efficientnet.tfkeras
    import efficientnet.keras as efn #import customed model to prevent issues-"valueerror: unknown activation function:swish"
    from keras.models import load_model
    # default
    model = load_model(os.path.join(path, modelh5))
    '''
    # for model with customized loss
    from CNN_Procedure import S3_Modeling as S3
    import keras.losses
    # For customized loss funciton (before loading model)
    gamma = 5                                    #customized !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    alpha_list =  [0.692, 0.154, 0.154]
    
    keras.losses.focal_loss = S3.FocalLoss(gamma, alpha_list)
    
    model = load_model(os.path.join(output_path, modelh5))
    #or default options
    #model = load_model(os.path.join(output_path, 'EfficientNetB3_NoneNone_b16_e5_t1_K3_a0.82_g5.h5'), custom_objects={'FocalLoss':S3.FocalLoss(gamma, alpha_list)})
    '''
    """
    if you wish to just perform inference with your model and not further optimization or training your model,
    you can simply wish to ignore the loss function like this: model_2
    """
    #model_2 = load_model(os.path.join(output_path, 'EfficientNetB3_NoneNone_b16_e5_t1_K3_a0.82_g5.h5'), compile=False)

    return model #VGG16(include_top=True, weights='imagenet')

# %% O.1  Utility functions
# %%
def load_image(path, resize, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=resize)
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    '''
    else:
        x = image.img_to_array(x)
        x= np.expand_dims(x, axis=0)
    '''
    return x

def load_image_cv(path, resize, preprocess=True):
    """Load and preprocess image."""
    x = cv2.imread(path)[:,:,::-1]
    x = cv2.resize(x, resize)
    
    if preprocess:
        x = preprocess_input(x)
        
    x = np.expand_dims(x, axis=0)

    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


# %% O.2  Grad-CAM
# %%
'''
#test
input_model = model #=build_model(output_path, 'EfficientNetB3_NoneNone_b16_e5_t1_K0_a0.82_g4.h5')
img = img
cls = 0 #class index, e.g., 0 or 1 for binary
layer_name = 'top_conv'
'''

def grad_cam(input_model, img, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]       #.output: <tf.Tensor 'dense_8/Softmax:0' shape=(?, 2) dtype=float32>
    conv_output = input_model.get_layer(layer_name).output #<tf.Tensor 'top_conv/convolution:0' shape=(?, ?, ?, 1536) dtype=float32>
    grads = K.gradients(y_c, conv_output)[0] #> a list type with only one element (as below)
    #image has to be same same shape with input_model.input > <tf.Tensor 'gradients/AddN:0' shape=(?, ?, ?, 1536) dtype=float32>

    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])
    #input_model.input: <tf.Tensor 'input_8:0' shape=(?, ?, ?, 3) dtype=float32>


    output, grads_val = gradient_function([img]) #output, grads_val: np.ndarray, shape = (1, 10, 10, 1536)
    output, grads_val = output[0, :], grads_val[0, :, :, :] ##output, grads_val: np.ndarray, shape = (10, 10, 1536)

    weights = np.mean(grads_val, axis=(0, 1))
    '''
    grads_val has 3 axes, summarize axis 0,1 but 2 > gets  1536 weights, each from a feature map
    '''    
    cam = np.dot(output, weights) #>shape = (10, 10)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR) ##resize # cam > shape = (W, H) e.g.= (300,300) via upsampling cv2.INTER_LINEAR
    cam = np.maximum(cam, 0)  ##ReLu activation to cam
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    #plt.imshow(cam)
    return cam
    
def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return new_cams

# %% O.3  Guided Backprop
# %%
def build_guided_model(model):
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = model #build_model()                                
    return new_model


'''
#test
input_model = guided_model #= build_guided_model(model)
img_path = path  #= os.path.join(origion_path, label_name, i)
images = load_image(img_path, resize)
layer_name = 'top_conv'
'''
def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input #shape= TensorShape([Dimension(None), Dimension(None), Dimension(None), Dimension(3)])
    layer_output = input_model.get_layer(layer_name).output #shape = TensorShape([Dimension(None), Dimension(None), Dimension(None), Dimension(1536)])
    grads = K.gradients(layer_output, input_imgs)[0]
    #same shape as input_imgs:TensorShape([Dimension(None), Dimension(None), Dimension(None), Dimension(3)])
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0] #shape= (1, 300, 300, 3)
    return grads_val




"""
to-do:
    search and modify code:　decode_predictions
"""
"""

Label0 = "h1975"
Label1 = "or4"
Label2 = 'NonCancerHEK293T'
Labels = [Label0, Label1, Label2]
model = model 
guided_model = guided_model 
img_path = path  
layer_name= "top_activation" 
cls=-1
true_cls= i
outpath = output_path2_FPred
"""

def compute_saliency(model, guided_model, Labels, img_path, outpath, layer_name='top_activation', cls=-1, true_cls=0):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class)
    """
    preprocessed_input = load_image_cv(img_path, resize, preprocess=False)  #for prediction
    #plt.imshow(preprocessed_input[0])

    predictions = model.predict(preprocessed_input)          #e.g., Binary: array([[0.8924582 , 0.10754178]], dtype=float32)
    #top_n = 1
    #top = decode_predictions(predictions, top=top_n)[0]        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #classes = np.argsort(predictions[0])[-top_n:][::-1]        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('Model prediction:')
    for i, (c, p) in enumerate(zip(Labels, predictions.tolist()[0])):
        print('\t{:15s}\t({})\twith probability {:.3f}'.format(c, i, p))
    '''
    for c, p in zip(classes, top):
        print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    '''    
    preprocessed_input = load_image(img_path, resize)  #for Grad-CAM implimentation
    
    if cls == -1:
        cls_pred = np.argmax(predictions)
        class_name = Labels[cls_pred] 
        print("Explanation for '{}'".format(class_name))
    
        gradcam = grad_cam(model, preprocessed_input, cls_pred, layer_name) #shape = (300, 300)
        gb = guided_backprop(guided_model, preprocessed_input, layer_name) #shape = (1, 300, 300, 3)
        #plt.imshow(gb[0])
        guided_gradcam = gb * gradcam[..., np.newaxis] #(1, 300, 300, 3) * (300, 300, 1) >  (1, 300, 300, 3)
        #plt.imshow(guided_gradcam[0])
    
        if cls_pred == true_cls:
            describe = "correct prediction- "
        else:
            describe = "false prediction- "

        #visualization:
        plt.figure() #figsize=(15, 15)
        #plt.subplot(131)
        '''
        describe_full = 'GradCAM- '+ describe + '\n' + class_name + ' with prob ' + str(round(predictions[0,cls_pred],3))
        plt.title(describe_full)
        '''
        plt.axis('off')
        plt.imshow(load_image(img_path, resize, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)  
        plt.savefig(os.path.join(outpath, file[:-4] +'- '+ 'GradCAM- '+ describe[:-2] + '.tiff'), transparent =True, dpi=400)
        plt.show()
        #plt.close()
    
        plt.figure() #figsize=(15, 15)
        #plt.subplot(131)
        '''
        describe_full = 'Guided Backprop- '+ describe + class_name
        plt.title(describe_full)
        '''
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))    
        plt.savefig(os.path.join(outpath, file[:-4] +'- ' + 'Guided Backprop' + '.tiff'), transparent =True, dpi=400)
        plt.show()
        #plt.close()    

        plt.figure() #figsize=(15, 15)
        #plt.subplot(131)
        '''
        describe_full = 'Guided GradCAM- '+ describe + '\n' + class_name + ' with prob ' + str(round(predictions[0,cls_pred],3))
        plt.title(describe_full)
        '''
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.savefig(os.path.join(outpath, file[:-4] +'- ' + 'Guided GradCAM- '+ describe[:-2] + '.tiff'), transparent =True, dpi=400)
        plt.show()
        #plt.close() 
        
    else:
        cls_pred = cls
        class_name = Labels[cls] 
        print("Explanation for '{}'".format(class_name))
    
        gradcam = grad_cam(model, preprocessed_input, cls, layer_name) #shape = (300, 300)
        gb = guided_backprop(guided_model, preprocessed_input, layer_name) #shape = (1, 300, 300, 3)
        #plt.imshow(gb[0])
        guided_gradcam = gb * gradcam[..., np.newaxis] #(1, 300, 300, 3) * (300, 300, 1) >  (1, 300, 300, 3)

        describe = "target- "

        #visualization:
        plt.figure() #figsize=(15, 15)
        #plt.subplot(131)
        '''
        describe_full = 'GradCAM- '+ describe + '\n' + class_name + ' with prob ' + str(round(predictions[0,cls],3))
        plt.title(describe_full)
        '''
        plt.axis('off')
        plt.imshow(load_image(img_path, resize, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)  
        plt.savefig(os.path.join(outpath, file[:-4] +'- '+ 'GradCAM- ' + describe + class_name + '.tiff'), transparent =True, dpi=400)
        plt.show()
        plt.close()
    
        plt.figure() #figsize=(15, 15)
        #plt.subplot(131)
        '''
        describe_full = 'Guided Backprop- '+ describe + class_name
        plt.title(describe_full)
        '''
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))    
        plt.savefig(os.path.join(outpath, file[:-4] +'- ' + 'Guided Backprop- ' + '.tiff'), transparent =True, dpi=400)
        plt.show()
        plt.close()    

        plt.figure() #figsize=(15, 15)
        #plt.subplot(131)
        '''
        describe_full = 'Guided GradCAM- '+ describe + '\n' + class_name + 'with prob ' + str(round(predictions[0,cls],3))
        plt.title(describe_full)
        '''
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.savefig(os.path.join(outpath, file[:-4] +'- ' +'Guided GradCAM- ' + describe + class_name + '.tiff'), transparent =True, dpi=400)
        plt.show()
        plt.close() 
        
    return gradcam, gb, guided_gradcam, cls_pred
'''    
    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, resize, preprocess=False)) / 2
        cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))
    
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, resize, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))
        
        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()
'''        






# %% I Load Testing set & trained model

# %%
import os
from os import listdir, walk
#import cv2
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import time

start_time = time.time()

#os.getcwd()
## Analysis directory
# directory = "D:\Project_Jane\StemCellReg2021\Oral cancer three-class classification_K-fold CV_SoftLabels" #"D:\Project_Jane\Grad-CAM\Input data without circle masks (Loss-MSE)"#os.chdir(directory)
directory = "/home/u4574403/Project_Jane/StemCellReg2021/Oral cancer three-class classification with SoftLabels (True Final 4 INTER)"

# Load Testing set & trained model
## Data directory
data_folder = 'oral cancer 0521-0618_tag300_Test'
data_ver = "Hard with Soft_true_DS-OS"
origion_path = os.path.join('/home/u4574403/Project_Jane/StemCellReg2021/Data', data_folder)

# For custom loss
Loss_name = 'mse' 

# for repetiontion
Trials = 1
K_fold = 5

last_layer = "top_activation" #'top_conv'

output_path = os.path.join(directory, f"output_{data_ver}_{Trials}-trials_{K_fold}-fold_{Loss_name}")
output_path2 = os.path.join(output_path, f"Grad-CAM_t0_k4")
if not os.path.exists(output_path2):
    os.mkdir(output_path2)
    


Label0 = "control"
Label1 = "5nMTG"
Label2 = 'NonCancerOral'
Labels = [Label0, Label1, Label2]



# %% I.1 Load trained model
#note: function build_model(path, modelh5) has to be modified if alpha0, gamma values are changed
model = build_model(output_path, 'EfficientNetB3_DS_INTEROS_t0_K4.h5')

guided_model = build_guided_model(model) 


# %% I.2 Load all testing set
H, W =  300, 300 # 224,224# Input shape, defined by the model (model.input_shape)
resize = (H, W) 

test_index = pd.read_csv(os.path.join(output_path, 'test_index_DS_INTER-OS_t0.csv'), header=None).values
test_index = [i[0] for i in test_index.tolist()]


All_path = {}
for i in range(len(Labels)):
    All_path[f"All_{i}_path"] = os.listdir(os.path.join(origion_path, Labels[i]))

test_path = {}
for i in range(len(Labels)):
    test_path[f"test_{i}_path"] = list(set(test_index).intersection(All_path[f"All_{i}_path"])) #or list(set(test_index) - set(All_1_path))



FPred_dic = {}
FPred_dic["FPred_0_cases"] = [] #True Positive class := 0
FPred_dic["FPred_1_cases"] = [] #True Positive class := 1
FPred_dic["FPred_2_cases"] = [] #True Positive class := 2

'''
import random
n = 3
random.seed(223)
'''
CPred_dic = {}
for i in range(len(Labels)):
    #CPred_dic[f"CPred_{i}_samples"] = random.sample(  list(set(test_path[f"test_{i}_path"] ) - set(FPred_dic[f"FPred_{i}_cases"])), n )
    CPred_dic[f"CPred_{i}_samples"] =  list( set(test_path[f"test_{i}_path"]) - set(FPred_dic[f"FPred_{i}_cases"]) )

'''
CPred_dic["CPred_0_samples"] = ['control_65.tif']
CPred_dic["CPred_1_samples"] = ['5nMTG_Oraldrug190.tif']
CPred_dic["CPred_2_samples"] = []
'''

# %%
i=2
CPred_dic[f"CPred_{i}_samples"]

# %%
#                                              II. Computing saliency
# %% [markdown]
# Label_indice = list(range(len(Labels)))
#
# '''
# #%% FPred_{label}_cases
# for i in range(len(Labels)):
#     if FPred_dic[f"FPred_{i}_cases"] == []:
#         continue
#     
#     label_name = Labels[i]
#
#     output_path2_FPred = os.path.join(output_path2, f"FPred_{i}");print(output_path2_FPred)
#     if not os.path.exists(output_path2_FPred):
#         os.mkdir(output_path2_FPred)
#     
#     for file in FPred_dic[f"FPred_{i}_cases"]:
#         path = os.path.join(origion_path, label_name, file);print(path)
#         #img = load_image_cv(path, resize, preprocess=False) #>shape=(1,300,300,3) #np.expand_dims(x, axis=0)
#         #img_prob = model.predict(img);print(img_prob)
#
#         gradcam, gb, guided_gradcam, cls_pred = compute_saliency(model, guided_model, Labels, path, layer_name=last_layer, 
#                                                        cls=-1, true_cls=i, outpath=output_path2_FPred)  
#         gradcam, gb, guided_gradcam, _ = compute_saliency(model, guided_model, Labels, path, layer_name=last_layer, 
#                                                        cls=i, true_cls=i, outpath=output_path2_FPred)
#         
#         Label_indice2 =list( set(Label_indice) - set([i]) - set([cls_pred]))
#         for j in Label_indice2:
#             gradcam, gb, guided_gradcam, _ = compute_saliency(model, guided_model, Labels, path, layer_name=last_layer, 
#                                                               cls=j, true_cls=i, outpath=output_path2_FPred) 
#  '''     
# Label_indice =
#

# %% CPred_ {"incorrectly_encoded_metadata": "{label}_cases"}
#for i in range(len(Labels)):
# for i in range(2,len(Labels))
for i in range(len(Labels)-1):
    if CPred_dic[f"CPred_{i}_samples"] == []:
        continue
    
    label_name = Labels[i]; print("current label:", label_name)
    output_path2_CPred = os.path.join(output_path2, f"CPred_{i}");print(output_path2_CPred)
    if not os.path.exists(output_path2_CPred):
        os.mkdir(output_path2_CPred)    
    
    for file in CPred_dic[f"CPred_{i}_samples"]:
        path = os.path.join(origion_path, label_name, file);print(path)
        #img = load_image_cv(path, resize, preprocess=False) #>shape=(1,300,300,3) #np.expand_dims(x, axis=0)
        #img_prob = model.predict(img);print(img_prob)
        
        gradcam, gb, guided_gradcam, _ = compute_saliency(model, guided_model, Labels, path, layer_name=last_layer, 
                                                       cls=-1, true_cls=i, outpath=output_path2_CPred) 
        
        '''
        Label_indice2 =list( set(Label_indice) - set([i]) )
        for j in Label_indice2:
            gradcam, gb, guided_gradcam, _ = compute_saliency(model, guided_model, Labels, path, layer_name=last_layer, 
                                                           cls=j, true_cls=i, outpath=output_path2_CPred) 
        '''


end_time = time.time()
print("Output completed, time spent (in hours):", (end_time - start_time)/3600)



# %%
