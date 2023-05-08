# -*- coding: utf-8 -*-
"""
The program converts .tif to .jpg/.png and also generates a .gif animation.

@author: Siddharth Shankar
"""
import imageio
from fnmatch import fnmatch
import os
from datetime import datetime as dt
from tqdm import tqdm
import pandas as pd
from osgeo import gdal
import rasterio as ras
images = []

# DIRECTORY STRUCTURE for Labels and Images
# BASE_PATH = r'C:/ICEYE/2023/subsets/subset_0_of_ICEYE_X7_GRD_SM_59797_20210620T052843_M_256'
BASE_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\Figure4_features\terminus'#r'C:\Users\shank\Downloads'
# BASE_PATH = r'C:\ICEYE\2023\clipped'
pattern = '*S2B_MSIL1C_20200826T141739_N0209_R096_T24WWU_20200826T144937_terminus.tif'

# Setting up path structure based on fjord and satellite sensor used         

fileNames = []
for _,dirs,files in os.walk(BASE_PATH,topdown=True): 
    dirs.clear() #excludes the files in subdirectories
    for name in files:   
        if fnmatch(name,pattern):
            fileNames.append(name)
options_ = [
            '-ot Byte',
            '-of JPEG',
            '-b 1 -b 2 -b 3',
            '-scale'#' #0 255', #120 150
        ]           

options_string = " ".join(options_)

# Defining output path directory for all sensor .jpg created         
out_path = BASE_PATH#+'/jpgs'

# Converting each geotiff into .jpg         
for file_ in fileNames:
    
    outfile = file_.split('.')[0]+'.jpg' 
    print(outfile)
    img = gdal.Open(os.path.join(BASE_PATH,file_))
    gdal.Translate(os.path.join(out_path,outfile),img,format='JPEG',
                   options=options_string)




