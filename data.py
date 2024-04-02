# %% [markdown]
# # Plot the annotated data
# - there are only half of the boxes annotated
# %%
import sys, os; sys.path += ['..']  # NOTE find shared modules

import matplotlib.pyplot as plt
import skimage

from util.label_studio_converter__brush import *
from util.plot import *
from util.preprocess import *

# %%
image = skimage.io.imread("./data/third/1.jpg")

points = preprocess_point("./data/third/points.json")

def plot_points(image, points):
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(13,10))
  plot_image(image, ax)
  ax.scatter(points[:,0], points[:,1], s=10, c='#7700ff', marker='s')

plot_points(image, points)
points.shape

