# %% [markdown]
# # Plot the annotated data
# ## Notes
# - the points are incomplete and inconsistently placed
# - there are only half of the boxes annotated
# %%
import sys, os; sys.path += ['..']  # NOTE find shared modules

import matplotlib.pyplot as plt
import numpy as np
import skimage
import json
import pandas as pd

from util.label_studio_converter__brush import *
from util.plot import *
from util.preprocess import *

# %%
image = skimage.io.imread("./data/third/1.jpg")

points = preprocess_point("./data/third/points.json")

points.shape

# %%

def plot_points(image, points):
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(13,10))
  plot_image(image, ax)
  ax.scatter(points[:,1], points[:,0], s=10, c='red', marker='o')

plot_points(image, points)
points.shape

# %%

boxes = preprocess_bbox("data/third/boxes.json")

fig, ax = mk_fig(shape=image.shape, scale=5)

plot_image(image, ax)
plot_boxes(boxes, ax)

boxes.shape

# %%

G = image[...,1]
G = ((G-G.min())/(G.max()-G.min())*255).astype(np.uint8)

X = 1450; D = 132; Y = 16
G = G[Y:Y+D, X:X+D]

f = 5

o = np.zeros((G.shape[0]*f, G.shape[1]*5), dtype=np.uint8)
for x in range(G.shape[0]):
  for y in range(G.shape[1]):
    o[x*f:x*f+f, y*f:y*f+f] = G[x,y]

plot_image(o)
skimage.io.imsave('1-G.jpg', o)
# %%
