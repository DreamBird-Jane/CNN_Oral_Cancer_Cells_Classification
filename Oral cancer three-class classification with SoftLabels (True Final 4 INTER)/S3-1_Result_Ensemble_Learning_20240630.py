#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Jul 15 14:50:36 2021

@author: jane_hsieh
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# %% Change directory if necessary
#os.getcwd()
## Analysis directory
directory = "/home/u4574403/Project_Jane/StemCellReg2021/Oral cancer three-class classification with SoftLabels (True Final 4 INTER)"
#directory = "/Users/jane_hsieh/Desktop/current work/CNN" # for mac

os.chdir(directory)


# import evaluation results data
sub_dir = "output_Hard only_1-trials_5-fold_mse" 
#"output_Hard only_1-trials_5-fold_mse"

#"output_Hard only_1-trials_5-fold_mse"              #M1: Prep = None , other_info =  None
#"output_Hard with Soft_1-trials_5-fold_mse"         #M2: Prep = "INTER" , other_info =  None
#"output_Hard with Soft_B-SMOTE_1-trials_5-fold_mse" #M3: Prep = "Bord_SMOTE_INTER", other_info =  None
#"output_Hard with Soft_OS_1-trials_5-fold_mse"      #M4: Prep = "OS_INTER", other_info = None
#"output_Hard with Soft_true_DS-OS_1-trials_5-fold_mse" #M5: Prep= "DS_INTER", other_info = "OS"

Prep= None#"Bord_SMOTE_INTER"#"DS_INTER"#None #"Bord_SMOTE_INTER" # 
other_info= None #"OS" #

trial = 0
batch_size = 16
epochs = 10

# %% Import Data
# true testing set labels
test_label = np.load(os.path.join(sub_dir, 'test_label.npy'))
test_label_INTER =np.load(os.path.join(sub_dir, 'test_label_INTER.npy'))

# predicted testing set probabilities
test_data_pred = pd.read_csv(os.path.join(sub_dir, f'Evaluation_{Prep}{other_info}_TestPred_batch-{batch_size}_epoch-{epochs}.csv'), index_col=0, float_precision='round_trip')
test_data_INTER_pred = pd.read_csv(os.path.join(sub_dir, f'Evaluation_{Prep}{other_info}_TestPred_inter_batch-{batch_size}_epoch-{epochs}.csv'), index_col=0, float_precision='round_trip')
## select data of the trial
test_data_pred = test_data_pred[test_data_pred['Trial']==trial]
test_data_INTER_pred = test_data_INTER_pred[test_data_INTER_pred['Trial']==trial]
# testing index(file) names
test_index = pd.read_csv(os.path.join(sub_dir, f'test_index_{Prep}-{other_info}_t{trial}.csv'), index_col = None, header=None).values.tolist()
test_index = [i[0] for i in test_index]
test_index_INTER = pd.read_csv(os.path.join(sub_dir, f'test_index_INTER_{Prep}-{other_info}_t{trial}.csv'), index_col = None, header=None).values.tolist()
test_index_INTER = [i[0] for i in test_index_INTER]

test_index_INTER2 = [np.int(i[5:8]) for i in test_index_INTER]

#test_index_{Prep}-{other_info}_t{trial}.csv
#'test_index_INTER_{Prep}-{other_info}_t{trial}.csv'



# %% Ensemble Learning (EL) results
Labels = ["control", "5nMTG", 'NonCancerOral']
## Hard ======================================================================================
#test_data_pred.columns
#1. simple ensemble learning
EL_simple = test_data_pred[Labels].groupby([test_data_pred.index]).mean()

#2. weighted ensemble learning (by acc)
w_tot_acc = test_data_pred.loc[test_data_pred.index==0,'accuracy'].sum()
weights = test_data_pred['accuracy']/w_tot_acc
EL_weighted_acc = test_data_pred[Labels].mul( weights, axis=0).groupby(test_data_pred.index).sum() #weighted sum of K folds by acc

#3. weighted ensemble learning (by f1)
w_tot_f1 = test_data_pred.loc[test_data_pred.index==0,'f1-score_macro'].sum()
weights = test_data_pred['f1-score_macro']/w_tot_f1
EL_weighted_f1 = test_data_pred[Labels].mul( weights, axis=0).groupby(test_data_pred.index).sum() #weighted sum of K folds by f1

## INTER ======================================================================================
#test_data_INTER_pred.columns
#1. simple ensemble learning
EL_INTER_simple = test_data_INTER_pred[Labels].groupby(test_data_INTER_pred.index).mean()

#2. weighted ensemble learning (by acc)
w_tot_acc = test_data_INTER_pred.loc[test_data_INTER_pred.index==0,'accuracy'].sum()
weights = test_data_INTER_pred['accuracy']/w_tot_acc
EL_INTER_weighted_acc = test_data_INTER_pred[Labels].mul( weights, axis=0).groupby(test_data_INTER_pred.index).sum() #weighted sum of K folds by acc

#3. weighted ensemble learning (by f1)
w_tot_f1 = test_data_INTER_pred.loc[test_data_INTER_pred.index==0,'f1-score_macro'].sum()
weights = test_data_INTER_pred['f1-score_macro']/w_tot_f1
EL_INTER_weighted_f1 = test_data_INTER_pred[Labels].mul( weights, axis=0).groupby(test_data_INTER_pred.index).sum() #weighted sum of K folds by acc

# %% Evaluation for metrics

from CNN_Procedure import S4_Evaluation as S4
import math

# For classification: confution matrix and Accuracy, sensitivity, specificity / RMSE (INTER) ==============================================
#1. simple ensemble learning
pred_label_simple = np.argmax(EL_simple.values,axis=1)
report_simple, overall_simple = S4.Multiclass_metrics_OVR(y_pred_prob = EL_simple.values, y_pred_label = pred_label_simple, 
                                                    y_true = test_label, y_index_names = test_index, target_names = Labels)
report_simple['ensemble'] = 'simple'

## RMSE for INTER testing set
RMSE = math.sqrt( ((test_label_INTER - EL_INTER_simple.values)**2).sum()/len(EL_INTER_simple) )
overall_simple['RMSE_INTER'] = RMSE


#2. weighted ensemble learning (by acc)
pred_label_weighted_acc = np.argmax(EL_weighted_acc.values,axis=1)
report_weighted_acc, overall_weighted_acc = S4.Multiclass_metrics_OVR(y_pred_prob = EL_weighted_acc.values, y_pred_label = pred_label_weighted_acc, 
                                                    y_true = test_label, y_index_names = test_index, target_names = Labels)
report_weighted_acc['ensemble'] = 'weighted_acc'

## RMSE for INTER testing set
RMSE = math.sqrt( ((test_label_INTER - EL_INTER_weighted_acc.values)**2).sum()/len(EL_INTER_weighted_acc) )
overall_weighted_acc['RMSE_INTER'] = RMSE


#3. weighted ensemble learning (by f1)
pred_label_weighted_f1 = np.argmax(EL_weighted_f1.values,axis=1)
report_weighted_f1, overall_weighted_f1 = S4.Multiclass_metrics_OVR(y_pred_prob = EL_weighted_f1.values, y_pred_label = pred_label_weighted_f1, 
                                                    y_true = test_label, y_index_names = test_index, target_names = Labels)
report_weighted_f1['ensemble'] = 'weighted_f1'

## RMSE for INTER testing set
RMSE = math.sqrt( ((test_label_INTER - EL_INTER_weighted_f1.values)**2).sum()/len(EL_INTER_weighted_f1) )
overall_weighted_f1['RMSE_INTER'] = RMSE


# concatenation -------------------------------
report = pd.concat([report_simple, report_weighted_acc, report_weighted_f1], axis =0)
report["Trial"] = trial

overall = pd.DataFrame([overall_simple, overall_weighted_acc, overall_weighted_f1])
overall['ensemble']  = ['simple', 'weighted_acc', 'weighted_f1']
overall["Trial"] = trial

with open(os.path.join(sub_dir, f'Ensemble_TestPred_overall_batch-{batch_size}_epoch-{epochs}.csv'), "a", newline='\n') as evalF_overall:
    overall.to_csv(evalF_overall, mode='a',header = evalF_overall.tell()==0, index=False)
                
with open(os.path.join(sub_dir , f'Ensemble_TestPred_batch-{batch_size}_epoch-{epochs}.csv'), "a", newline='\n') as evalF:
    report.to_csv(evalF, mode='a', header = evalF.tell()==0, index=True)

# For calculating mean prob. for differenct INTER data ===============================================================

#1. simple ensemble learning
EL_INTER_simple['C0>C1'] = (EL_INTER_simple.iloc[:,0] > EL_INTER_simple.iloc[:,1])*1  
EL_INTER_simple['C0<C1'] = 1-EL_INTER_simple['C0>C1']
EL_INTER_simple['INTER_type'] = test_index_INTER2
M_INTER_simple = EL_INTER_simple.groupby(['INTER_type']).mean()
M_INTER_simple['ensemble'] = 'simple'

#2. weighted ensemble learning (by acc)
EL_INTER_weighted_acc['C0>C1'] = (EL_INTER_weighted_acc.iloc[:,0] > EL_INTER_weighted_acc.iloc[:,1])*1  
EL_INTER_weighted_acc['C0<C1'] = 1-EL_INTER_weighted_acc['C0>C1']
EL_INTER_weighted_acc['INTER_type'] = test_index_INTER2
M_INTER_weighted_acc = EL_INTER_weighted_acc.groupby(['INTER_type']).mean()
M_INTER_weighted_acc['ensemble'] = 'weighted_acc'

#3. weighted ensemble learning (by f1)
EL_INTER_weighted_f1['C0>C1'] = (EL_INTER_weighted_f1.iloc[:,0] > EL_INTER_weighted_f1.iloc[:,1])*1  
EL_INTER_weighted_f1['C0<C1'] = 1-EL_INTER_weighted_f1['C0>C1']
EL_INTER_weighted_f1['INTER_type'] = test_index_INTER2
M_INTER_weighted_f1 = EL_INTER_weighted_f1.groupby(['INTER_type']).mean()
M_INTER_weighted_f1['ensemble'] = 'weighted_f1'

M_INTER = pd.concat([M_INTER_simple, M_INTER_weighted_acc, M_INTER_weighted_f1], axis=0)
M_INTER["Trial"] = trial

del M_INTER_simple, M_INTER_weighted_acc, M_INTER_weighted_f1


with open(os.path.join(sub_dir, f'Ensemble_TestPred_inter_M-{batch_size}_epoch-{epochs}.csv'), "a", newline='\n') as evalF:
    M_INTER.to_csv(evalF, mode='a',header = evalF.tell()==0, index=True)

# %%
test_data_INTER_pred

# %%
test_label_INTER 

# %%
test_index_INTER

# %%
EL_INTER_simple 

# %%
test_index_INTER2
