# -*- coding: utf-8 -*-
# %%
"""
Created on Sun Jul 11 19:34:18 2021

@author: user
"""
#import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict

#import os
#from os import listdir, walk

# for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# from keras.utils import np_utils

# %%
#                                              Binary Classification Metrics
# %%
# For evaluation of the trained model (metrics:　tn, fn, fp, tp, Accuracy, Sencitivity, Specificity, fp_cases, fn_cases)    
# y_pred: val_predict_label, y_true: train_label[val_indices], y_index_names: np.array(train_index)[val_indices]   
def Evaluated_metrics(y_pred, y_true, y_index_names): #Evaluated_metrics(val_predict_label, train_label[val_indices], np.array(train_index)[val_indices])
    """
    y_pred, y_true: 1-D np.arrarys with predicted v.s. true labels (y) of input data X
    y_index_names:　list or np.arrary of index names of all labels of y; 
                    e.g., if X are images: y_index_names = ['Image_18757.tif', 'Image_18758.tif', ..., 'Image_18826.tif']
    """
    cm = confusion_matrix(y_pred , y_true)
    tn, fn, fp, tp = cm.ravel()  
    
    Accuracy = (tn+tp)/(tn+fp+fn+tp)
    Sencitivity = tp/(tp+fn)
    Specificity = tn/(fp+tn)
    
    ## Record the indices of the false positive/false negative cases   ----> MakeFunction???????????????
    is_fp = ((y_true == 0) * (y_pred == 1))*1
    fp_index = np.where(is_fp == 1)[0]
    fp_cases = [np.array(y_index_names)[i] for i in fp_index]

    is_fn = ((y_true == 1) * (y_pred == 0))*1
    fn_index = np.where(is_fn == 1)[0]
    fn_cases = [np.array(y_index_names)[i] for i in fn_index]
    
    return tn, fn, fp, tp, Accuracy, Sencitivity, Specificity, fp_cases, fn_cases


# %% Plot ROC curve and calculate AUC score

# #val_label = train_label[val_indices]
# def ROC_AUC(val_label, val_predict_prob):
#     """
#     Input:
#         val_label: np.array of true labels of input data X (i.e., y_true); e.g.,  array([0, 0, ..., 0, 1, 1, ..., 1])
#         val_predict_prob: 2-D array of predicted probabilities of X, of which each row = [Pr(0), Pr(1)]; 
#                          e.g., array([[0.61831874, 0.3816812 ], ..., [0.15863289, 0.84136707]], dtype=float32)
#     Output: Given that label i is considered positive (i =  0, 1, "micro"),
#         fpr[i] =  Increasing false positive rates such that element j is the false positive rate of predictions with score >= thresholds[i][j]
#         tpr[i] =  Increasing true positive rates such that element j is the true positive rate of predictions with score >= thresholds[i][j]
#         thresholds[i] = Decreasing thresholds on the decision function used to compute fpr and tpr. 
#                         thresholds[i][0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1 (y_score = val_predict_prob[i])
    
#         roc_auc[i]: auc
#     """
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     thresholds = dict()
#     roc_auc = dict()
#     n_classes= len(np.unique(val_label, return_counts=False)) #count the number of unique elements in val_label
    
#     # for one-hot encoder of val_label
#     val_label2=np_utils.to_categorical(val_label)
    
#     for i in range(n_classes):
#         # label i is considered positive in current loop
#         fpr[i], tpr[i],thresholds[i]= roc_curve(val_label2[:, i], val_predict_prob[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
        
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(val_label2.ravel(), val_predict_prob.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
#     return fpr, tpr, thresholds, roc_auc

# %%
#                                              Multi-class Classification Metrics
# %%
'''
y_pred_prob = val_predict_prob
y_pred_label = val_predict_label
y_true = val_label
y_index_names = val_index
target_names = Labels
'''

def Multiclass_metrics_OVR(y_pred_prob, y_pred_label, y_true, y_index_names, target_names):
    """
    # Evaluation metrics for multi-class classification; one over rest (OVR) case
    Input:
    - y_pred_prob: 2-D array with shape = (N, num_classes)
    - y_pred_label, y_true: 1-D np.arrarys with predicted v.s. true labels (y) of input data X; shape = (N, )
    - y_index_names:　list or np.arrary of index names of all labels of y; length = N
                    e.g., if X are images: y_index_names = ['Image_18757.tif', 'Image_18758.tif', ..., 'Image_18826.tif']
    - target_names: list of label names (starting from class 0 , 1, 2,...); length=num_classes >=2
                    e.g., ['h1975', 'or4', 'NonCancerHEK293T']
                    
    Output:
    - report: pd.DataFrame of confusion matrix, true_n of each class,  OVR metrics for each class (precision/recall/f1/specificity)
    - overall:　overall report for accuracy, and precision, recall, f1-score, AUC under macro/weighted averaging
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score

    cm = confusion_matrix(y_pred_label, y_true)

    # %% OVR (One versus Rest) metrics
    multi_metrics = classification_report(y_true, y_pred_label, target_names=target_names, digits=4, output_dict=True)

    report = pd.DataFrame(cm, columns = ['pred_n_'+label for label in target_names], index = target_names)
    report['true_n'] = [multi_metrics[label]['support'] for label in target_names]
    report['precision_OVR'] = [multi_metrics[label]['precision'] for label in target_names]
    report['recall_OVR'] = [multi_metrics[label]['recall'] for label in target_names]
    report['f1_OVR'] = [multi_metrics[label]['f1-score'] for label in target_names]
    
    AUC_bi = []
    f_cases_list = []
    f_cases_cols = ['f_cases_'+str(i) for i in range(len(target_names)-1)]
    for i, label in enumerate(target_names):
        # For AUC of each class
        y_true_bi = np.where(y_true==i,1,0)
        y_pred_prob_bi = y_pred_prob[:, i] 
        AUC_bi.append( roc_auc_score(y_true_bi, y_pred_prob_bi))
        
        # For false classification cases of each class
        cls_set = set( range(len(target_names)) )
        cls_rest = cls_set - {i}
        f_cases_dict = OrderedDict({})
        ii = 0
        
        for j in cls_rest:            
            # Record the indices of the false predicted cases from class i
            is_f = ((y_true == i) * (y_pred_label == j))*1
            f_index = np.where(is_f == 1)[0]
            f_cases = [np.array(y_index_names)[i] for i in f_index]
            f_cases_dict[f_cases_cols[ii]] =  f_cases #.append(f_cases)                              #!!!!!!!!!!!!!!!
            ii += 1
        f_cases_list.append(f_cases_dict) #[label] = f_cases_dict
        
    report['AUC_OVR'] = AUC_bi
    report = pd.concat([report, pd.DataFrame(f_cases_list, index = target_names)], axis = 1)


    # %% Overall metrics
    overall=OrderedDict({})
    
    overall['accuracy'] = multi_metrics['accuracy']
    
    for key, value in multi_metrics['macro avg'].items():
        if key == 'support':
            continue
        overall[key+'_macro'] = value
 
    overall['AUC_macro'] = roc_auc_score(y_true, y_pred_prob, average="macro", multi_class="ovr")

    for key, value in multi_metrics['weighted avg'].items():
        if key == 'support':
            continue
        overall[key+'_weighted'] = value
          
    overall['AUC_weighted'] = roc_auc_score(y_true, y_pred_prob, average="weighted", multi_class="ovr")
    
    return report, overall


"""
# For validation set
val_predict_prob = model.predict(val_data)
val_predict_label = np.argmax(val_predict_prob,axis=1)
   
report_val, overall_val = Multiclass_metrics_OVR(y_pred_prob = val_predict_prob, y_pred_label = val_predict_label, 
                                                y_true = val_label, y_index_names = val_index, target_names = Labels)

# For testing set
test_predict_prob = model.predict(test_data)
test_predict_label = np.argmax(test_predict_prob,axis=1)
report_test, overall_test = Multiclass_metrics_OVR(y_pred_prob = test_predict_prob, y_pred_label = test_predict_label, 
                                                y_true = test_label, y_index_names = test_index, target_names = Labels)
"""
# %%
