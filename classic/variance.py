# %% [markdown]
# # Second round of filtered images
# Inspired by Olavi's success with variance images in the red component.
# The YU stuff wtih Sten's bf2pc model seems too fancy.
# But the gist is - there is a possibly simple 1-D color space projection that contains all necessary info / two with complementary info.
# -> why use RGB channels - find a transform that incorporates reasonable priors
# ## Notes
# - Olavi observed: R and G color components (cc) are similar (~ in the same phase, same type of interference image).
#   - Blue channel looked different and seemed to be rather in the opposite phase compared to Red and Green.
#   - Blue channel was also out of focus, probably due to shorter wavelength.
# - G channel seems sharper and has less artifacts than R channel.
# %%

import sys; sys.path += [".."]  # NOTE find shared modules

from util.preprocess import *
from util.plot import *

import skimage
import numpy as np
import pandas as pd

img = "third/1.jpg"
RGB = skimage.io.imread(f"../data/{img}").astype(float) / 255

imshow(RGB);

# %%

def show_channels(X, info, scale=2):
  C = X.shape[-1]
  fig, axs = plt.subplots(nrows=C, ncols=1, figsize=(4*scale, 3*scale*C))

  for ax,c in zip(axs.flat, range(C)):
    imshow(X[:,:,c], ax=ax, scale=scale)

  fig.suptitle(f"Color components {img}: {info}")
  plt.tight_layout()

show_channels(RGB, "RGB");

R_G = RGB[:,:,0] - RGB[:,:,1]
imshow(R_G, title=f"Difference between R and G channel {img}");

# %% [markdown]
# ## Reproducing Olavi's RGB-Red Variance results
# - improved rectangular filter with circular filter
# - found out that the gauss results in inverting the variance and thus in identity
# %%

G = RGB[:,:,1]  


from scipy.ndimage import generic_filter


circular_mask = lambda d: (r:=d//2,
  np.array([[1 if (x**2 + y**2) <= r**2 else 0 for x in range(-r,r+1)] for y in range(-r,r+1)])
)[-1]


gaussian_mask = lambda d: (r:=d//2,
  np.array([[np.exp(-(x**2 + y**2) / (2*(sigma:=1)**2)) for x in range(-r,r+1)] for y in range(-r,r+1)])
)[-1]



variance_square = lambda x: np.sqrt(np.sum((x - (M:=np.mean(x)))**2) / (x.size - 1)) / M

variance_circle = lambda x_: (
  mask := circular_mask( int(np.sqrt(x_.size)) ).ravel(),
  x := x_ * mask,
  4 / np.pi * np.sqrt(np.sum((x - (M:=np.mean(x)))**2) / (mask.sum() - 1)) / M
)[-1]

variance_gauss = lambda x_: (
  mask := gaussian_mask( int(np.sqrt(x_.size)) ).ravel(),
  x := x_ * mask,
  np.sqrt(np.sum((x - (M:=np.mean(x)))**2) / (mask.sum() - 1)) / M
)[-1]

identity = lambda x: x


filters = (
  variance_square,
  variance_circle,
)

variances = pd.DataFrame(columns=['filter', 'kernel_size', 'result', 'channel'])

kernels = [9, 18]#[3, 5, 7, 9, 11, 15, 17, 23]

scale=5
fig, axs = plt.subplots(ncols=len(filters), nrows=len(kernels), figsize=(4*len(filters)*scale, 3*len(kernels)*scale))

for col, method in enumerate(filters):
  for ax,kernel_size in zip(axs[:,col], kernels):
    out = generic_filter(G, method, size=kernel_size)
    variances.loc[len(variances.index)] = [method.__name__, kernel_size, out, 'G']
    imshow(out, ax=ax, title=f"Variance of G channel {img}, kernel size {kernel_size}")

for method in filters:
  for k in kernels:
    out = generic_filter(RGB[:,:,0], method, size=k)
    variances.loc[len(variances.index)] = [method.__name__, k, out, 'R']
# %%


# select the entry where filter is square and kernel size is 9

r = variances.query("filter == 'variance_square' and kernel_size == 9 and channel == 'R'").result.values[0]

g = variances.query("filter == 'variance_square' and kernel_size == 9 and channel == 'G'").result.values[0]


threshold = 0.1

# apply the threshold to make r into a binary mask
r = (r > threshold).astype(int)
g = (g > threshold).astype(int)



fig, (ax_r, ax_g) = mk_fig(1,2, shape=RGB.shape, scale=15)


# draw the binary mask in ws as overlay with color and alpha
ax_r.imshow(skimage.color.label2rgb(RGB, r, saturation=1, bg_color=None, alpha=0.5, colors=colors[1])
, alpha=0.5)
ax_r.imshow(skimage.segmentation.mark_boundaries(out, g, color=colors[1]))

ax_g.imshow(skimage.color.label2rgb(RGB, g, saturation=1, bg_color=None, alpha=0.5, colors=colors[2])
, alpha=0.5)
ax_g.imshow(skimage.segmentation.mark_boundaries(out, r, color=colors[2]))



