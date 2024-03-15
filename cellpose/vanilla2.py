# %% [markdown]
# # Cellpose 
# Just for a high resolution prediction on all images we have annotations for.
# %% 

from cellpose import models
from cellpose.io import imread
import skimage

import sys; sys.path += ['..']

from util.plot import *

image = imread('../data/third/1.jpg')

model = models.Cellpose(gpu=True, model_type='cyto', net_avg=True)

masks, _flows, _styles, diams = model.eval([image], diameter=None, channels=[[0, 0]])
print("Automatically estimated diameter:", diams)

plot_image(skimage.color.label2rgb(masks[0], image, bg_color=None, alpha=0.5, saturation=1, colors = colors), scale=5) 
  # %%



images = [imread(f'../data/third/{i}.jpg') for i in [1,2,4]]

masks, _flows, _styles, _diams = model.eval(images, diameter=None, channels=[[0, 0]]*3)


M = 0
for m in masks: M += m.max()

print(f"Total number of cells: {M}")
print(f"Individual number of cells: {[m.max() for m in masks]}")
# %%
