from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     polygons = []
#     color = []
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#         ax.imshow(np.dstack((img, m*0.35)))
        
sam = sam_model_registry["vit_h"](checkpoint="C:/segment-anything/sam_vit_h_4b8939.pth")
image = cv2.imread('C:/segment-anything/subset_0_of_ICEYE_X7_GRD_SM_59797_20210620T052843_CLIPPED_crop4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor = SamPredictor(sam)
predictor.set_image(image)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
# masks, _, _ = predictor.predict(<input_prompts>)

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

final_binary_image.save('helheim_iceye_SA_FAIR4_255.png')
# cv2.imwrite('test.jpg',final_binary_mask)
