# %% [markdown]
# # [Cellpose](https://cellpose.readthedocs.io/en/latest)
# see https://cellpose.readthedocs.io/en/latest/notebook.html
# and https://cellpose.readthedocs.io/en/latest/outputs.html#plotting-functions

# %%
import sys, os; sys.path += ['..']  # NOTE find shared modules

import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread
import pandas as pd

from util.plot import *

import skimage

# %% 

dataset = 'third'
data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+').query('objective == "40x"')


imgids = data['imgid'].unique()


data
# %%

model_type = 'cyto'
model = models.Cellpose(model_type=model_type, gpu=True, net_avg=True)
images = [imread(f'../data/third/{i}.jpg') for i in imgids]

masks, _flows, _styles, diams = model.eval(images, diameter=None, channels=[[1,0]]) # we select the green channel because it seemed to be the sharpest

# %%

fig, axs = mk_fig(3,2, shape=images[0].shape[:2])

for img, ax, mask, d, i in zip(images, axs.flat, masks, diams, imgids):
  plot_image(skimage.color.label2rgb(mask, img, bg_color=None, alpha=0.5, saturation=1, colors=colors), ax=ax)
  ax.set_title(f'cellpose (vanilla): {dataset}/{i}.jpg (auto cell diameter: {d:.0f}px)')

# %%
