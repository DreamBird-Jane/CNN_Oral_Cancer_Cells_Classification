# Research Background

This is the work for the paper "**Deep learning-based automatic image classification of oral cancer cells acquiring chemoresistance in vitro**" (under publishing by **PLOS ONE**)

## Abstract
Cell shape reflects the spatial configuration resulting from the equilibrium of cellular
and environmental signals and is considered a highly relevant indicator of its function
and biological properties. For cancer cells, various physiological and environmental
challenges, including chemotherapy, cause a cell state transition, which is accompanied
by a continuous morphological alteration that is often extremely difficult to recognize
even by direct microscopic inspection. To determine whether deep learning-based image
analysis enables the detection of cell shape reflecting a crucial cell state alteration, we
used the oral cancer cell line resistant to chemotherapy but having cell morphology
nearly indiscernible from its non-resistant parental cells. We then implemented the
automatic approach via deep learning methods based on EfficienNet-B3 models, along
with over- and down-sampling techniques to determine whether image analysis of the
Convolutional Neural Network (CNN) can accomplish three-class classification of
non-cancer cells vs. cancer cells with and without chemoresistance. We also examine
the capability of CNN-based image analysis to approximate the composition of
chemoresistant cancer cells within a population. We show that the classification model
achieves at least 98.33% accuracy by the CNN model trained with over- and
down-sampling techniques. For heterogeneous populations, the best model can
approximate the true proportions of non-chemoresistant and chemoresistant cancer cells
with Root Mean Square Error (RMSE) reduced to 0.16 by Ensemble Learning (EL). In
conclusion, our study demonstrates the potential of CNN models to identify altered cell
shapes that are visually challenging to recognize, thus supporting future applications
with this automatic approach to image analysis.

# Data
Due the the large volume of the datasets, it's store in Kaggle repository "[oral cancer cells with chemoresistance in vitro](https://www.kaggle.com/datasets/janehsieh/oral-cancer-cells-with-chemoresistance-in-vitro)"


# Codes
## Three-class classification results of oral cancer cells
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

   


## Grad-CAM Visualization
