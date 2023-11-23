# %% [markdown]
# # µSAM - point prompts
# Stuart in his third datarelease provided only points for cells.. Lets see what we can do with it

# %%
import sys, os; sys.path += ['..']  # NOTE find shared modules

import os
import itertools as it
import json

import numpy as np
import skimage
import matplotlib.pyplot as plt

import torch
import micro_sam
from micro_sam import instance_segmentation, util

from util.points2masks_tiled import *
from util.plot import *
from util.preprocess import *


dataset_id = 'third'
setup_cache(dataset_id, f'R=None')
_, TILES = make_tiles_dummyBox(R=None)

cache_dir = mk_cache(f"micro-sam/{dataset_id}/tiled-points", dirs='embed masks', clear=True)

imgid = '1'
model_type = 'vit_b'
iou_thresh = 0.88

predictor = util.get_sam_model(model_type=model_type)
tiled_masks = []

for tile in TILES:
  img, _, pts, pts_bg = load_tile(tile)
  pEmbed = f'{cache_dir}/embed/{tile}.zarr'
  
  print('LOADING'  if os.path.exists(pEmbed) else
        'WRITING', f'embedding cache at {pEmbed}')
  embeddings = util.precompute_image_embeddings(
    predictor, img, ndim = 2, save_path = pEmbed,
    # NOTE: we dont use µSAM built in tiling because its bad with out huge multi-tile BG
  )

  points = np.concatenate([pts_bg, pts])
  labels = np.concatenate([np.ones(len(pts_bg)), np.zeros(len(pts))])

  mask = micro_sam.prompt_based_segmentation.segment_from_points(
      predictor=predictor,
      image_embeddings=embeddings,
      points=points,
      labels=labels,
    )[0]

  tiled_masks.append(mask)

plot_tiles(imgid, posts=[
  lambda ax, X: ax.imshow(ski.color.label2rgb(tiled_masks[X['idx']], X['img'], saturation=1, bg_color=None, alpha=0.5, colors=[[.25,0,.5]])),
], 
suptitle=f"µSAM (vanilla {model_type}) - BG/FG from point promts: {dataset_id}/{imgid}.jpg")
 


# %% [markdown]
# ## Result Notes
# - as expected its best to have positve points for intended object (BG) and the cells as negatives
# # TODO
# - use different methods to smoothen the masks
# - https://docs.monai.io/en/stable/transforms.html#fillholes for holes
# - ?? for cracks?
