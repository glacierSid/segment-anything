'''
This code creates a prompt based generation of masks
using Segment Anything Model (SAM).
'''


from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Define the main working directories, models, and paths
# ********************************************************************************
FEATURE_OF_INTEREST = 'crevasse' #icebergs,crevasse,terminus,supraglacial_lakes
MODEL_TYPE = 'vit-b'
MODEL_WEIGHTS = 'sam_vit_b_01ec64.pth' 
OUTPUT_FOLDER = 'predict_no_prompt' # predict_with_prompt, predict_no_prompt

BASE_PATH = r'C:/segment-anything/images/testing/testing_data/%s'%(FEATURE_OF_INTEREST)
OUTPUT_PATH = os.path.join(BASE_PATH,'%s'%(OUTPUT_FOLDER))
fileName = 'S1B_IW_GRDH_1SDH_20190502T091100_20190502T091125_016063_01E364_ADEC_subset.png'    

# *********************************************************************************



# Setup the image and the model checkpoints
sam = sam_model_registry["%s"%(MODEL_TYPE)](checkpoint="C:/segment-anything/images/testing/models/%s"%(MODEL_WEIGHTS))

# Enable for GPU
# device = "cuda"
# sam.to(device=device)

image = cv2.imread('C:/segment-anything/images/testing/%s/%s'%(FEATURE_OF_INTEREST,fileName))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = SamPredictor(sam)
predictor.set_image(image)



#*************************************************************************************** 

# Setup the prompt coordinates on the image for the model
# Foreground label is 1 and Background label is 0
# There can be multiple foreground and background defined for the model.

input_points = np.array([[268,301],[476,362],[625,119],[58,121]])
input_label = np.array([1,1,1,1])


masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_label,
    multimask_output=True,
)
# *************************************************************************************

# Enable if need to auto-generate the masks from default settings
#-----------------------------------------------------------------
'''
mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
'''

# Enable for more control on the IoU and Stability score
#--------------------------------------------------------
'''
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.7,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

masks2 = mask_generator_2.generate(image)
'''

# Plot the image
'''
for num in range(len(masks)):
    im = Image.fromarray(masks[num]['segmentation'])
    im.save('terminus_test1_%s.png'%(num))
'''


binary_masks = []
for num in range(len(masks)):
    binary_masks.append(masks[num])#['segmentation'])

final_binary_mask = sum(binary_masks)

# Enable to get binary classification
# final_binary_mask[final_binary_mask>0]=255
final_binary_image = Image.fromarray(final_binary_mask)

# output_path = r'C:\segment-anything\images\icebergs\prompt_predict'
final_binary_image.save(os.path.join(OUTPUT_PATH,'%s_predict_%s.png'%(fileName.split('.')[0],MODEL_TYPE)))
# cv2.imwrite('test.jpg',final_binary_mask)
