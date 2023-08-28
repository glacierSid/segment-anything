# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:19:47 2023

@author: shank
"""

from PIL import Image
import os
from fnmatch import fnmatch
from osgeo import gdal
import numpy as np
import re
import rasterio as ras

root = r'C:/segment-anything/images/testing/final_manuscript_images/Landsat/subsets'
tif_path = os.path.join(root,'tif')#r'D:\trackBergs\Track_Helheim_Melange_2020_testing\helheim_geotiff_clip_2022\png'
png_path = os.path.join(root,'png')
output_path = os.path.join(png_path,'png_to_tif')



tif_pattern = "*.tif"

outfile = {}

# TIF Files
all_tifs = []
for _,dirs,tif_files in os.walk(tif_path,topdown=True): 
    dirs.clear() #excludes the files in subdirectories
    for tif in tif_files:   
        if fnmatch(tif,tif_pattern):
            all_tifs.append(tif)

# PNG Files
all_pngs = []
png_pattern = '*.jpg'
for _,dirs,png_files in os.walk(png_path,topdown=True):
    dirs.clear() # excludes sub-directories
    for png in png_files:
        if fnmatch(png,png_pattern):
            all_pngs.append(png)


for tiffile in all_tifs:
    print('TIF files: \n',tiffile)
    with ras.open(os.path.join(tif_path,tiffile),count=1) as tif_src:
        tif_profile = tif_src.profile
        tif_crs = tif_src.crs

for pngfile in all_pngs:
    print('PNG files: \n',pngfile)
    with ras.open(os.path.join(png_path,pngfile),count=1) as png_src:
        png_profile = png_src.profile
        png_profile.update(
            driver = "GTiff",
            dtype = "uint16",
            transform = tif_src.transform,
            crs=tif_src.crs,
            width = tif_src.width,
            height = tif_src.height
            )
        with ras.open(os.path.join(output_path,"%s.tif"%(pngfile.split('.')[0])),'w',**png_profile) as dest:
            dest.write(png_src.read(1),1)
         


