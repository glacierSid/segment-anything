from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

fileName = 'S1B_IW_GRDH_1SDH_20190502T091100_20190502T091125_016063_01E364_ADEC_subset.png'    
sam = sam_model_registry["vit_h"](checkpoint="C:/segment-anything/sam_vit_h_4b8939.pth")
# device = "cuda"
# sam.to(device=device)
image = cv2.imread('C:/segment-anything/images/icebergs/%s'%(fileName))
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
'''
for num in range(len(masks)):
    im = Image.fromarray(masks[num]['segmentation'])
    im.save('terminus_test1_%s.png'%(num))
'''

binary_masks = []
for num in range(len(masks)):
    binary_masks.append(masks[num]['segmentation'])

final_binary_mask = sum(binary_masks)

# Enable to get binary classification
# final_binary_mask[final_binary_mask>0]=255
final_binary_image = Image.fromarray(final_binary_mask)

output_path = r'C:\segment-anything\images\icebergs\predict'
final_binary_image.save(os.path.join(output_path,'%s_predict_vit_h_defaultsetting.png'%(fileName.split('.')[0])))
# cv2.imwrite('test.jpg',final_binary_mask)
