# Research Background

This is the work for the paper "**Deep learning-based automatic image classification of oral cancer cells acquiring chemoresistance in vitro**" (under publishing by **PLOS ONE**)

## Abstract
Cell shape reflects the spatial configuration resulting from the equilibrium of cellular and environmental signals and is considered a highly relevant indicator of its function and biological properties. For cancer cells, various physiological and environmental challenges, including chemotherapy, cause a cell state transition, which is accompanied by a continuous morphological alteration that is often extremely difficult to recognize even by direct microscopic inspection. To determine whether deep learning-based image analysis enables the detection of cell shape reflecting a crucial cell state alteration, we used the oral cancer cell line resistant to chemotherapy but having cell morphology nearly indiscernible from its non-resistant parental cells. We then implemented the automatic approach via deep learning methods based on EfficienNet-B3 models, along with over- and down-sampling techniques to determine whether image analysis of the Convolutional Neural Network (CNN) can accomplish three-class classification of non-cancer cells vs. cancer cells with and without chemoresistance. We also examine the capability of CNN-based image analysis to approximate the composition of chemoresistant cancer cells within a population. We show that the classification model achieves at least 98.33% accuracy by the CNN model trained with over- and down-sampling techniques. For heterogeneous populations, the best model can approximate the true proportions of non-chemoresistant and chemoresistant cancer cells with Root Mean Square Error (RMSE) reduced to 0.16 by Ensemble Learning (EL). In conclusion, our study demonstrates the potential of CNN models to identify altered cell shapes that are visually challenging to recognize, thus supporting future applications with this automatic approach to image analysis.

# Data
Due the the large volume of the datasets, it's stored in Kaggle repository "[oral cancer cells with chemoresistance in vitro](https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro)."

## I. HOME dataset
There are 3 types of oral cells:
1. Oral cancer cells: untreated OECM1 cells (“PARENTAL”), N=300.
   - folder name: 'control' (do not mis-understand it with normal oral cells)
2. Chemoresistant oral cancer cells::, chemoresistant OECM1 cells (“RESISTANT”), N=300.
   - folder name: '5nMTG'
3. Normal oral cells: normal oral keratinocytes (“CONTROL”), N=300.
   - folder name: 'nonCancerOral'

These 3 categories of data are split into validation and testing sets, which are located in:
1. Validation set folder: [oral cancer 0521-0618_tag300_Val]([https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro](https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro))
2. Testing set folder: [oral cancer 0521-0618_tag300_Test]([https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro](https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro))

![Fig 1. Three classes of cells (with labeling), named as HOMO data. Each
class has equally 300 images. Notice that the morphological features of three classes of
cells have only subtle differences that the expert might be able to distinguish.](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/attachments/Fig1.png)
   
## II. HETERO dataset
To simulate heterogeneity within tumors, PARENTAL and RESISTANT cells were harvested, mixed in a 1:2, 2:1, 1:3, or 3:1 ratio. The images of mixed cells (i.e., featuring soft labels) were classified as the heterogeneous data set (HETERO).
There are hence 4 types of mixed oral cells:
1. PARENTAL : RESISTANT = 1:2 (N=200)
   - folder name: '20X_TG2-WT1'  
2. PARENTAL : RESISTANT = 2:1 (N=200)  
   - folder name: '20X_TG1-WT2'  
3. PARENTAL : RESISTANT = 1:3 (N=200)  
   - folder name: '20X_TG3-WT1'  
5. PARENTAL : RESISTANT = 3:1 (N=200)  
   - folder name: '20X_TG1-WT3'  
  
These 4 categories of data are split into validation and testing sets, which are located in:
1. Validation set folder: [oral cancer inter stage-all_tag300_Val](https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro)
2. Testing set folder: [oral cancer inter stage-all_tag300_Test](https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro)


200 images of each ratio were collected, resulting in a total of 800 images. 

# Codes
## I. Model Training: Three-class classification results of oral cancer cells
Refer to folder "[Oral cancer three-class classification with SoftLabels (True Final 4 INTER)](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/tree/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER))."

There are 5 models for CNN classification with soft labels:
1. M1. Only the HOMO data set was used for model training.
   - [S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M1.ipynb](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M1.ipynb)
2. M2. Both the HOMO and the HETERO data sets were used for model training.
   - [S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M2.ipynb](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M2.ipynb)
3. M3. Both the HOME and the HETERO data sets were used. For the HETERO data set, we additionally performed Borderline-SMOTE to generate more synthetic HETERO data for model training.
   - [S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M3.ipynb](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M3.ipynb)
4. M4. Both the HOME and the HETERO data sets were used. For the HETERO data set, we additionally performed Random OS to generate more HETERO data for model training.
   - [S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M4.ipynb](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M4.ipynb)
5. M5. Both the HOME and the HETERO data sets were used. Additionally, we performed the ENN-DS method to delete some ambiguous cell images in HETERO data set first and then used the Random OS to generate more HETERO data.
   - [S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M5.ipynb](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S-All_Image_Multi-Class_Classification_TestOut_K-foldCV_SoftLabels_20240624(oral)_M5.ipynb)
   - The example results can be found in folder "[output_Hard with Soft_true_DS-OS_1-trials_5-fold_mse](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/tree/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER))"

In addition, there are two subsequent Python files for summarizing the model training results:
1. Summarizing model training results from 5-fold cross-validating(CV): [S3-1_Result_Statistics_Multiclass_20240630.py](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S3-1_Result_Statistics_Multiclass_20240630.py)
2. Deriving ensembling results for testing sets: [S3-1_Result_Ensemble_Learning_20240630.py](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER)/S3-1_Result_Ensemble_Learning_20240630.py)


## II. Grad-CAM Visualization
The Grad-CAM results were used as a visual explanation to reveal how deep learning algorithms make decisions, aiming to elucidate the interpretability of our results in the context of chemoresistant oral cell morphology.

The codes to generate Grad-CAM pictures can be found in folder "[Grad-CAM](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/tree/main/Grad-CAM)."

### Input
1. Trained model in .h5 format: It would be generated after running codes in folder "[Oral cancer three-class classification with SoftLabels (True Final 4 INTER)](https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/tree/main/Oral%20cancer%20three-class%20classification%20with%20SoftLabels%20(True%20Final%204%20INTER))." 
  - E.g., "EfficientNetB3_DS_INTEROS_t1_K3.h5"
Any medical image (in .tiff format) from out HOMO test sets

### Output
It specifically generates 3 kinds of pictures per medical image:
1. GradCAM  <img src="https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/attachments/control_Oral17-%20GradCAM-%20correct%20prediction.png" alt="GradCAM" width="500"/>
2. Guided Backprop <img src="https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/attachments/control_Oral17-%20Guided%20Backprop.png" alt="Guided Backprop" width="500"/>
1. Guided GradCAM <img src="https://github.com/DreamBird-Jane/CNN_Oral_Cancer_Cells_Classification/blob/main/attachments/control_Oral17-%20Guided%20GradCAM-%20correct%20prediction.png" alt="Guided GradCAM" width="500"/>


