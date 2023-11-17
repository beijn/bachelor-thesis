
# %% [markdown]
# # Thresholding, Histogram clustering ...
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

def imshow(img, pos=None, neg=None, ax=None, scale=20):
  if ax is None:
    fig, ax = plt.subplots(1,1, figsize=(4*scale, 3*scale))

  ax.imshow(img, cmap='gray')
  if pos is not None: ax.scatter(pos[:,1], pos[:,0], s=scale*4, marker='x', c='cyan')
  if neg is not None: ax.scatter(neg[:,1], neg[:,0], s=scale*4, marker='D', c='red' )
  ax.axis('off')
  plt.tight_layout()
  return ax
# %%


img_rgb = ski.io.imread('../data/third/1.jpg')

img = ski.color.rgb2gray(img_rgb)
pos = preprocess_point('../data/third/annotations.json')
neg = preprocess_point('../data/third/annot-bg.json')

colors = colorcet.m_glasbey.colors
colors = [c for c,_ in zip(it.cycle(colors), pos)]

imshow(img)

# %% [markdown]
# ## Thresholding
# - definitely correlated but missing much stuff in some regions and oversensitive in others because of local contrast ratios
#   - => need a localized extremum detection
# %%
thresh = ski.color.gray2rgb(img)
thresh[img<=0.42] = colors[0]

imshow(thresh, pos);

# %% [markdown]
# ## Thresholding on local minima
# %%

# determine local minima

# first compute divergence
img_dx, img_dy = np.gradient(img)
img_d = np.sqrt(img_dx**2 + img_dy**2)

imshow(img_d, pos)

bla = (1-img_d) * img
imshow(bla)
imshow(img)

# %%
thresh2 = ski.color.gray2rgb(bla)
thresh2[bla<=0.42] = colors[0]
imshow(thresh2, pos)

# %%

def norm(x): return (x-x.min())/(x.max()-x.min())


#f.apply_hysteresis_threshold(img, 0.42, 0.5),


x = f.butterworth(img, 0.2)[10:-10,10:-10]  # we cut of edge artifacts so that normalization is more dirable
imshow(1-norm(x))

x = f.butterworth(x, 0.18, high_pass=False)[10:-10,10:-10] 
imshow(1-norm(x))


# %% 


from skimage import data
from skimage.morphology import disk
from skimage.filters import rank


_mean = rank.mean(img, footprint=disk(3))


from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float

im = img_as_float(img)

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.minimum_filter(im, size=5, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(-im, min_distance=20)

# display results
fig, axes = plt.subplots(1, 3, figsize=(40,40), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()

plt.show()


# %% [markdown]
# # Notes
# . Gradient: Have kind of reinvented Sobel filter?
# - Butterworth: super at filtering out low frequency noise (eg background and PC artificats) 
# 
# # TODO
# - maybe use wavelet transform for local minima smoothing
# %%
