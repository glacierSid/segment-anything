'''
This code generates masks for images of icebergs, terminus, crevasses, and supraglacial lakes
in optical and SAR imagery using Segment Anything Model (SAM).

It is a NON-prompt based generation of masks.
'''


from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch

# Define the main working directories, models, paths, and fileNames
# ********************************************************************************
FEATURE_OF_INTEREST = 'terminus' #icebergs,crevasse,terminus,supraglacial_lakes
MODEL_TYPE = 'vit_l'
MODEL_WEIGHTS = 'sam_vit_l_0b3195.pth' # sam_vit_b_01ec64.pth,sam_vit_h_4b8939.pth,sam_vit_l_0b3195.pth
OUTPUT_FOLDER = 'predict_no_prompt' # predict_with_prompt, predict_no_prompt

BASE_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\Figure4_features/%s'%(FEATURE_OF_INTEREST)
OUTPUT_PATH = os.path.join(BASE_PATH,'%s'%(OUTPUT_FOLDER))
fileName = 'S2B_MSIL1C_20200826T141739_N0209_R096_T24WWU_20200826T144937_terminus_subset.jpg'    

# *********************************************************************************


# fileName = 'S1B_IW_GRDH_1SDH_20190502T091100_20190502T091125_016063_01E364_ADEC_subset.png'    
# sam = sam_model_registry["vit_h"](checkpoint="C:/segment-anything/sam_vit_h_4b8939.pth")
sam = sam_model_registry["%s"%(MODEL_TYPE)](checkpoint="C:/segment-anything/images/testing/models/%s"%(MODEL_WEIGHTS))

# device = "cuda"
# sam.to(device=device)
# image = cv2.imread('C:/segment-anything/images/icebergs/%s'%(fileName))
image = cv2.imread(os.path.join(BASE_PATH,r'%s'%fileName))

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor = SamPredictor(sam)
predictor.set_image(image)


# Enable if need to auto-generate the masks from default settings
#-----------------------------------------------------------------

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

# Enable for more control on the IoU and Stability score
#--------------------------------------------------------

# mask_generator_2 = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.6,
#     stability_score_thresh=0.7,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )

# masks2 = mask_generator_2.generate(image)


# Plot the image

for num in range(len(masks)):
    im = Image.fromarray(masks[num]['segmentation'])
    im.save(os.path.join(OUTPUT_PATH,'terminus_test2_%s.png'%(num)))

'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 
# torch.cuda.empty_cache()


binary_masks = []
for num in range(len(masks)):
    binary_masks.append(masks[num]['segmentation'])

final_binary_mask = sum(binary_masks)

# Enable to get binary classification
# final_binary_mask[final_binary_mask>0]=255
final_binary_image = Image.fromarray(final_binary_mask)

# output_path = r'C:\segment-anything\images\icebergs\predict'
# final_binary_image.save(os.path.join(output_path,'%s_predict_vit_h_defaultsetting.png'%(fileName.split('.')[0])))
final_binary_image.save(os.path.join(OUTPUT_PATH,'%s_predict_%s_%s.png'%(fileName.split('.')[0],MODEL_TYPE,OUTPUT_FOLDER)))

# cv2.imwrite('test.jpg',final_binary_mask)
'''