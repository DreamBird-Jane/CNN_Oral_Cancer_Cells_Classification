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


# %% Change directory if necessary
#os.getcwd()
## Analysis directory
# directory = "D:\Project_Jane\StemCellReg2021\Oral cancer three-class classification with SoftLabels (Prev INTER without 140 and 410)"
directory = "/home/u4574403/Project_Jane/StemCellReg2021/Oral cancer three-class classification with SoftLabels (True Final 4 INTER)"

os.chdir(directory)



# %% import evaluation results data
sub_dir = "output_Hard with Soft_1-trials_5-fold_mse"
print(f"Folder:\t{sub_dir}")
#"output_Hard only_1-trials_5-fold_mse"              #M1: Prep = None , other_info =  None
#"output_Hard with Soft_1-trials_5-fold_mse"         #M2: Prep = "INTER" , other_info =  None
#"output_Hard with Soft_B-SMOTE_1-trials_5-fold_mse" #M3: Prep = "Bord_SMOTE_INTER", other_info =  None
#"output_Hard with Soft_OS_1-trials_5-fold_mse"      #M4: Prep = "OS_INTER", other_info = None
#"output_Hard with Soft_true_DS-OS_1-trials_5-fold_mse" #M5: Prep= "DS_INTER", other_info = "OS"


Prep= "INTER"#None #"DS_INTER"# "OS_INTER"#"Bord_SMOTE_INTER" #
other_info= None#"OS" #

trial = 0
batch_size = 16
epochs = 10



#Evaluation_{Prep}{other_info}_overall_batch-{batch_size}_epoch-{epochs}.csv
#'Evaluation_{Prep}{other_info}_inter_M_batch-{batch_size}_epoch-{epochs}.csv'

## 1. without preprocess
df_None = pd.read_csv(os.path.join("./",sub_dir, f'Evaluation_{Prep}{other_info}_overall_batch-{batch_size}_epoch-{epochs}.csv'), index_col = None, float_precision='round_trip')
#df_None.columns
#df_None = df_None[['Focal_gamma', 'Focal_alpha0', 'Trial', 'K-fold', 'val_Acc', 'val_Sens', 'val_Spec', 'val_auc', 'test_Acc', 'test_Sens', 'test_Spec', 'test_auc']]
df_None = df_None[['accuracy', 'recall_macro', 'precision_macro', 'f1-score_macro', 'AUC_macro', 
                   'dataset', 'Trial', 'K-fold']] #'Focal_gamma', 'Focal_alpha0', 'Focal_alpha1', 'N1', 'N2', 'N3', 'N4'

#df_None_m = df_None.groupby(['dataset', 'Focal_alpha0', 'Focal_alpha1', 'N0', 'N1' ]).mean() #'Trial'
#df_None_se = df_None.groupby(['dataset', 'Focal_alpha0', 'Focal_alpha1', 'N0', 'N1']).std() # 'Trial'
df_None_m = df_None.groupby(['dataset', 'Trial' ]).mean() #'Trial'
df_None_se = df_None.groupby(['dataset', 'Trial']).std() # 'Trial'

'''
for col in df_None_m.columns:
    print(f"Mean of {col}: \n", round(df_None_m[col]*100,4), '\n\n')

for col in df_None_se.columns:
    print(f"Mean of {col}: \n", round(df_None_m[col]*100,4), '\n\n')
'''
'''
print("Mean of val:\n", round(df_None_m.loc[('val', 0.45, 0.45, 240, 240)]*100, 2) )
print("SE of val:\n", round(df_None_se.loc[('val', 0.45, 0.45, 240, 240)]*100, 2) )
print("Mean of test:\n", round(df_None_m.loc[('test', 0.45, 0.45, 240, 240)]*100, 2) )
print("SE of test:\n", round(df_None_se.loc[('test', 0.45, 0.45, 240, 240)]*100, 2) )
'''
print("Mean of val:\n", round(df_None_m.loc[('val',trial)]*100, 2) )
print("SE of val:\n", round(df_None_se.loc[('val', trial)]*100, 2) )
print("Mean of test:\n", round(df_None_m.loc[('test', trial)]*100, 2) )
print("SE of test:\n", round(df_None_se.loc[('test', trial)]*100, 2) )

## 2. For intermediate data
df_inter = pd.read_csv(os.path.join("./",sub_dir, f'Evaluation_{Prep}{other_info}_inter_batch-{batch_size}_epoch-{epochs}.csv'), index_col = None, float_precision='round_trip')
df_inter.columns #= ['Focal_gamma', 'Focal_alpha0', 'Focal_alpha1', 'Trial', 'K-fold', 'Mean_prob_label0', 'Mean_prob_label1', 'Mean_prob_label2']

df_inter_m = df_inter.groupby(['Trial']).mean(); 
print(round(df_inter_m.loc[(trial), ['RMSE_val', 'RMSE_test']], 4))
df_inter_se = df_inter.groupby(['Trial']).std();   
print(round(df_inter_se.loc[(trial), ['RMSE_val', 'RMSE_test']], 4))
'''
df_inter_m = df_inter.mean(); 
print(round(df_inter_m[['RMSE_val', 'RMSE_test']], 4))
df_inter_se = df_inter.std(); 
print(round(df_inter_se[['RMSE_val', 'RMSE_test']], 4))
'''

## 2. For intermediate data - mean(se) probability of each type of INTER data
df_inter2 = pd.read_csv(os.path.join("./",sub_dir, f'Evaluation_{Prep}{other_info}_inter_M_batch-{batch_size}_epoch-{epochs}.csv'), index_col = None, float_precision='round_trip')
#df_inter2.columns
df_inter2 = df_inter2[['INTER_type', 'Pr_control', 'Pr_5nMTG', 'Pr_NonCancerOral', 
                       'C0>C1', 'C0<C1', 'dataset', 'Trial']]#'N1', 'N2', 'N3', 'N4'
df_inter2_m = df_inter2.groupby(['dataset','Trial', 'INTER_type']).mean() #'Trial'
df_inter2_se = df_inter2.groupby(['dataset','Trial', 'INTER_type']).std() # 'Trial'

list1 = ['Pr_control', 'Pr_5nMTG', 'Pr_NonCancerOral']
list2 = ['C0>C1', 'C0<C1']

print("Mean of val:\n", round(df_inter2_m.loc[('val',trial),list1]*100, 2) )
print("SE of val:\n", round(df_inter2_se.loc[('val',trial),list1]*100, 2) )
print("Mean of test:\n", round(df_inter2_m.loc[('test',trial),list1]*100, 2) )
print("SE of test:\n", round(df_inter2_se.loc[('test',trial),list1]*100, 2) )

print("Mean of val:\n", round(df_inter2_m.loc[('val',trial),list2]*100, 2) )
print("SE of val:\n", round(df_inter2_se.loc[('val',trial),list2]*100, 2) )
print("Mean of test:\n", round(df_inter2_m.loc[('test',trial),list2]*100, 2) )
print("SE of test:\n", round(df_inter2_se.loc[('test',trial),list2]*100, 2) )
