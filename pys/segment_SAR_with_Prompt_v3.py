'''
This code generates masks for images of icebergs, terminus, crevasses, and supraglacial lakes
in optical and SAR imagery using Segment Anything Model (SAM).
sam_vit_h_4b8939.pth
It is a prompt based generation of masks where a point or a polygon or both
can be used to tell the model potential region of interest and assist it in 
generating the masks.

'''


from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import geopandas as gpd
# ********************************************************************************
# Define the main working directories, models, paths, and fileNames
# ********************************************************************************
FEATURE_OF_INTEREST = 'planet' #icebergs,crevasse,terminus,supraglacial_lakes,planet, sentinel-2, sentinel-1, timelapse
MODEL_TYPE = 'vit_l'
MODEL_WEIGHTS = 'sam_vit_l_0b3195.pth' # sam_vit_b_01ec64.pth,sam_vit_h_4b8939.pth,sam_vit_l_0b3195.pth
OUTPUT_FOLDER = 'predict_with_prompt' # predict_with_prompt, predict_no_prompt

BASE_PATH = r'C:\segment-anything\images\testing\final_manuscript_images\Figure2_openwater/%s'%(FEATURE_OF_INTEREST)
OUTPUT_PATH = os.path.join(BASE_PATH,'%s'%(OUTPUT_FOLDER))
fileName = '20220714_151057_24_227b_3B_AnalyticMS_SR_clip_subset.jpg'   

# *********************************************************************************


# Setup the image and the model checkpoints
sam = sam_model_registry["%s"%(MODEL_TYPE)](checkpoint="C:/segment-anything/images/testing/models/%s"%(MODEL_WEIGHTS))

# Enable for GPU
# device = "cuda"
# sam.to(device=device)

# image = cv2.imread(r'C:/segment-anything/images/testing/testing_data/%s/%s'%(FEATURE_OF_INTEREST,fileName))
image = cv2.imread(os.path.join(BASE_PATH,r'%s'%fileName))

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = SamPredictor(sam)
predictor.set_image(image)


#*************************************************************************************** 
# PROMPTS
# **********
# Setup the prompt coordinates on the image for the model
# Foreground label is 1 and Background label is 0
# There can be multiple foreground and background defined for the model.

input_points = np.array([[316,156],[32,90],[17,94],[737,17],[474,257],
                          [111,159],[193,65],[283,490],[660,228],[456,429],
                         [23,87],[146,78],[223,71],[329,152],[273,508],
                          [456,436],[465,258],[622,206],[723,12],[41,94]
                         ])
input_label = np.array([1,1,1,1,1,
                        1,1,1,1,1,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        ]) # Either 1 or 0


masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_label,
    multimask_output=True,
)



if FEATURE_OF_INTEREST == 'terminus':
    for num in range(len(masks)):
        im = Image.fromarray(masks[num]['segmentation'])
        im.save(os.path.join(OUTPUT_PATH,fileName.split('.')[0]+'%s_predict.png'%(num)))

else:
    binary_pred_zeros = np.zeros_like(masks[1])
    for num in range(len(masks)):
        # 25% or higher number of pixels are True, that means it is a potential representation of background
        # and not of icebergs
        if np.count_nonzero(masks[num])>(0.25*(masks[num]).size):
            continue
        else:               
            binary_pred_zeros[masks[num]==1]=1
    im = Image.fromarray(binary_pred_zeros)
    im.save(os.path.join(OUTPUT_PATH,fileName.split('.')[0]+'_predict_20pt_adjacent.png'))





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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)* color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image,alpha=0.6)
    show_mask(mask, plt.gca())
    show_points(input_points, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
  




'''
binary_masks = []
for num in range(len(masks)):
    binary_masks.append(masks[num])#['segmentation'])

final_binary_mask = sum(binary_masks)

# Enable to get binary classification
# final_binary_mask[final_binary_mask>0]=255
final_binary_image = Image.fromarray(final_binary_mask)

final_binary_image.save(os.path.join(OUTPUT_PATH,'%s_predict_%s_%s.png'%(fileName.split('.')[0],MODEL_TYPE,OUTPUT_FOLDER)))
'''