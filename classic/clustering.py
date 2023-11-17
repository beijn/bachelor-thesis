
# %% [markdown]
# # Clustering
# - https://www.geeksforgeeks.org/k-means-clustering-with-scipy/
# %%

import skimage as ski
from skimage import filters as f


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import scipy as sc

import sys; sys.path += ['..']  # NOTE find shared modules
from util.preprocess import *
from util.plot import *

import colorcet
import itertools as it

# %%


def imshow(img, pos=None, neg=None, ax=None, scale=20):
  if ax is None:
    fig, ax = plt.subplots(1,1, figsize=(4*scale, 3*scale))

  ax.imshow(img, cmap='gray')
  if pos is not None: ax.scatter(pos[:,1], pos[:,0], s=scale*4, marker='x', c='cyan')
  if neg is not None: ax.scatter(neg[:,1], neg[:,0], s=scale*4, marker='D', c='red' )
  ax.axis('off')
  plt.tight_layout()
  return ax


img_rgb = ski.io.imread('../data/third/1.jpg')

img = ski.color.rgb2gray(img_rgb)
pos = preprocess_point('../data/third/annotations.json')
neg = preprocess_point('../data/third/annot-bg.json')

colors = colorcet.m_glasbey.colors
colors = [c for c,_ in zip(it.cycle(colors), pos)]

imshow(img)

# %% [markdown]
# ## Clustering by k-means based on instensity and selecting the pixels from lowest cluster for nucleus detection
# - seems like it would also benefit from something relative because in some conglomerates theres less contrast
# %%

# compute the distance tranform of an image where every pos point is marked
import scipy.ndimage

points = np.ones_like(img)
points[pos[:,0], pos[:,1]] = 0

dist = scipy.ndimage.distance_transform_edt(points)
imshow(dist, pos)

# stack the image and the distance transform
data = np.stack([img, dist], axis=-1)

from scipy.cluster.vq import whiten, kmeans, vq, kmeans2

data = whiten(data.reshape(-1,2)).reshape(*data.shape)

imshow(data[...,0], pos)



# %% 

centroids, labels = kmeans2(data.reshape(-1,2), 2, iter=50, minit='++')
labels = labels.reshape(data.shape[:2])

imshow(labels, pos)

imshow(ski.color.label2rgb(labels, img, colors=colors), pos)

# %% 

centroids, labels = kmeans2(data.reshape(-1,2), 3, iter=50, minit='++')
labels = labels.reshape(data.shape[:2])

imshow(labels, pos)

imshow(ski.color.label2rgb(labels, img, colors=colors), pos)

# %%

centroids, labels = kmeans2(data.reshape(-1,2), 4, iter=50, minit='++')
labels = labels.reshape(data.shape[:2])

imshow(labels, pos)

imshow(ski.color.label2rgb(labels, img, colors=colors), pos)

# %%

centroids, labels = kmeans2(data.reshape(-1,2), 20, iter=50, minit='random')
labels = labels.reshape(data.shape[:2])

imshow(labels, pos)

imshow(ski.color.label2rgb(labels, img, colors=colors), pos)


# %% [markdown]
# # Notes
# - Vornoi by it self is just useless unless as a very bad stupid pseudo target or prior for segmentation methods
# - Clustering by intensities and selecting the lowest cluster is same as thresholding
#   - would need relative / local minima...
# - kmeans is very subsceptible to minit (worst fort k=2) and looks different every time
# 
# ## Whats the possible uses of clustering with intesities AND distances?
# - generate a pseudo target for segmentation
# - but not to transfer as is  in a setting where we only have intensities