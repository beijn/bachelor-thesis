# %% [markdown]
# # Train cellpose on point heatmaps
# %%

import skimage, scipy

import matplotlib.pyplot as plt

import sys; sys.path += [".."]  # NOTE find shared modules

from util.preprocess import *
from util.plot import *

import json
import pandas as pd



# %%

dataroot = './data'
dataset = 'third'
imgid = '1'

data = pd.read_csv(f'{dataroot}/{dataset}/data.csv', sep='\s+')
points = json.load(open(f'{dataroot}/{dataset}/points.json'))

load_image = lambda i: skimage.io.imread(f"{dataroot}/{dataset}/{i}.jpg").astype(np.float32) / 255


__annot_id = 0
coco = dict(
  info = dict(
    year = 2023,
    version = 231210,
    description = 'EVOS with Stuarts constant settings. Incomplete points by Stuart.',
    contributor = 'Stuart Fawke',
  ),
  images = [(
    S := load_image(ID).shape[:2], 
    dict(
      id = ID,
      width = S[1],
      height = S[0],
      file_name = f'{dataset}/{ID}.jpg',
    ))[-1] for ID in data['img']
  ],
  annotations = [(
    W := A['original_width'],
    H := A['original_height'],
    w := A['value']['width']/100 * W,
    h := w,
    x := A['value']['x']/100 * W - w/2,
    y := A['value']['y']/100 * H - h/2,
    dict(
      id = (__annot_id := __annot_id+1),
      img_id = 1,
      category_id = 1,
      segmentation = [],
      area = w*h,
      bbox = [x,y,w,h],
      iscrowd = 0,
    ))[-1] for A in points[0]['annotations'][0]['result']
  ],
  categories = [dict(
    id = 1,
    name = 'cell',
    supercategory = 'cell',
  )]
)
# TODO keypoints? 

json.dump(coco, open(f'{dataroot}/{dataset}/points-coco.json', 'w'), indent=2)
# %%
