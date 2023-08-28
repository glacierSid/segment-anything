# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:37:12 2023

@author: Siddharth Shankar
Generating F1 score and confusion matrix for the results from 
Segment Anything Model (SAM) when compared with the manual labeled
ground truth data.

Covering different sensors and feature types with this metric to provide a 
comprehensive comparison of SAM in the polar sciences.

F1-score/Dice-score, confusion matrix, precision and recall, IoU/Jaccard score.
"""

import imageio
from fnmatch import fnmatch
import os
from datetime import datetime as dt
from tqdm import tqdm
import pandas as pd
from osgeo import gdal
import rasterio as ras
import cv2
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report, f1_score, jaccard_score
from PIL import Image
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt

# TRUE_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\metrics\model_metric\true_2'
TRUE_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\metrics\with_prompt_metric\true'
PREDICT_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\metrics\with_prompt_metric\predict'
#PREDICT_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\metrics\model_metric\predict_2'

true_pattern = '*.png'
predict_pattern = '*.png'
# True Label list 1D
trueLabel = []
predictLabel = []
for _,dirs,files in os.walk(TRUE_PATH,topdown=True): 
    dirs.clear() #excludes the files in subdirectories
    for name in files:   
        if fnmatch(name,true_pattern):
            trueLabel.append(name)
trueLabel = sorted(trueLabel)
# print(trueLabel)

# Predicted Label list 1D
predictLabel = []

for _,dirs,files in os.walk(PREDICT_PATH,topdown=True): 
    dirs.clear() #excludes the files in subdirectories
    for name in files:   
        if fnmatch(name,predict_pattern):
            predictLabel.append(name)


predictLabel = sorted(predictLabel)
# print(predictLabel)


df_list = []
for num,true in enumerate(trueLabel):
    df = pd.DataFrame()
    trueArr = asarray(Image.open(os.path.join(TRUE_PATH,true))).flatten()
    predArr = asarray(Image.open(os.path.join(PREDICT_PATH,predictLabel[num]))).flatten()
    
    class_report = classification_report(trueArr, predArr)
    # print('Classification Report: ',class_report)
    
    conf_matrix = confusion_matrix(trueArr, predArr,normalize='true')
    # print('Confusion Matrix: ',conf_matrix)

    cmd = ConfusionMatrixDisplay(conf_matrix)
    
    cmd.plot(cmap='viridis')
    plt.title('Image: %s'%(true))
    cmd.figure_.savefig('%s_WP_CM.png'%(true),dpi=300)
    # print('F1 score for %s : '%(true),f1_score(trueArr, predArr))
    
    df['image'] = [true]
    df['F1_score'] = [(f1_score(trueArr, predArr))]
    # df['IoU'] = jaccard_score(trueArr, predArr)
    print(df)
    df_list.append(df)
    
df_final = pd.concat(df_list)
# df_final.to_csv(os.path.join(PREDICT_PATH,'f1_score_no_prompt_v2.csv'))
