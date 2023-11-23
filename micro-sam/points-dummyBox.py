# %% [markdown]
# # µSAM dummy point prompts from point prompts
# Stuart in his third datarelease provided only points for cells.. Lets see what we can do with it

# %%
import sys, os; sys.path += ['..']  # NOTE find shared modules

import os
import itertools as it
import json

import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt

import micro_sam
from micro_sam import instance_segmentation, util

from util.plot import *
from util.cache import *
from util.preprocess import *


# %%
points = preprocess_point('../data/third/annotations.json')


dataset_id = 'third'

model_type = 'vit_b'
iou_thresh = 0.88

cache_dir = mk_cache(f"micro-sam/{dataset_id}", dirs='embed masks')

imgid = '1'


pImage = f"../data/{dataset_id}/{imgid}.jpg"
pEmbed = f"{cache_dir}/embed/{imgid}-{model_type}.zarr"

image = skimage.io.imread(pImage)

# %%

predictor = util.get_sam_model(model_type=model_type)

embeddings = util.precompute_image_embeddings(
  predictor, image, ndim = 2, save_path=pEmbed,
  tile_shape=(tile:=1024, tile), halo=(halo:=tile//4, halo)
)


# %%

r = 25 # make dummy boxes with this radius around the points

y,x = points.T
boxes = np.array([x-r, y-r, np.ones_like(x)*r*2, np.ones_like(x)*r*2]).T

insts_dummybox = [
  micro_sam.prompt_based_segmentation.segment_from_box(
    predictor=predictor,
    image_embeddings=embeddings,
    box=np.array([y, x, y+h, x+w]), 
  )[0] for x,y,w,h in boxes
]

masks = np.zeros(image.shape[:2], dtype=np.uint16)
for i, inst in enumerate(insts_dummybox):
  masks[inst] = i+1

# %%

fig, ax = mk_fig(shape=image.shape)

out = skimage.color.label2rgb(
  masks, image, saturation=1, bg_color=None, alpha=0.5, colors=mk_colors(boxes))

plot_image(out, ax=ax, title=f"µSAM (vanilla {model_type}) - promted by dummy boxes with radius {r}px from Stuart\'s points: third/1.jpg")

plot_boxes(boxes, ax)
