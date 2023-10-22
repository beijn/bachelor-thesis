# %% [markdown]
# # Plot the example annotation(s)
#
#
# ## Notes on the format
#
# see `annotation-outline.txt` and `out.json`
#
# - some annotations have multiple results
# - all masks 4 color channels are identical (re-evaluate on final data)
#

# %%
import sys, os; sys.path += [os.path.join(os.path.expanduser('~'), 'thesis')]  # NOTE hardcoded project root to find shared util modules

import matplotlib.pyplot as plt
import numpy as np
import skimage
import json
import pandas as pd

# %%
image = plt.imread("data/third/1.jpg")

_annot = json.load(open('data/third/annotations.json'))

# %%
from util.label_studio_converter__brush import *

def preprocess_point(annot):
  '''
    returns: dict: image → numpy instance segmentation mask
  '''

  out = pd.DataFrame()

  for result in annot[0]['annotations'][0]['result']:
    h,w = result['original_height'], result['original_width']
    # rot = result['image_rotation]

    x = result['value']['x']/100 * w
    y = result['value']['y']/100 * h
    s = result['value']['width']

    i = result['to_name']

    out = pd.concat([out, pd.DataFrame(dict(
      x=[x],
      y=[y],
    ))], ignore_index=True)

  return out
annot = preprocess_point(_annot); annot

# %%

def plot_points(image, points):
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(13,10))
  plt.tight_layout()
  ax.axis('off')

  ax.imshow(image)
  ax.scatter(points.x, points.y, s=10, c='red', marker='o')

plot_points(image, annot)
