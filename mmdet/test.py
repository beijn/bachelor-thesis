# %% [markdown]
# # MMDetection Framework - CenterNet
# %%

import sys; sys.path += ['..']  # NOTE find shared modules
from util.preprocess import *
from util.plot import *


from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import cv2
import mmcv


import torch

# %%


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')


# NOTE: kernel needs to be restarted after changing this
config = './centernet-update_r50-caffe_fpn_ms-1x_coco.py'
checkpoint = './weights/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth'
model = init_detector(config, checkpoint, device=device)


visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta


img = skimage.io.imread('../data/park.jpg')

result = inference_detector(model, img)



visualizer.add_datasample(
name='result',
image=img,
data_sample=result,
draw_gt=False,
pred_score_thr=0.3,
show=False)

img = visualizer.get_image()

plot_image(img);



# %%
