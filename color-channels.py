# %% [markdown]
# # Separate the color channels
# - different color channels have different focal planes
# %%
import sys; sys.path += ['..']  # NOTE find shared modules

from util.plot import *
from util.preprocess import *

import numpy as np

import skimage
image = skimage.io.imread("./data/third/1.jpg")

# %%

I = image[450:750, 700:1000]

fig, axs = mk_fig(2,2, scale=0.6)


axs[0].scatter(-10,-10, label='PC microscopy image', color='white')
plot_image(I, axs[0])
axs[0].legend()
  
def norm(X):
  return (X-X.min())/(X.max()-X.min())

def do(A, C, c, cn):
  X = np.stack([I[:,:,C]]*3, axis=-1)
  ax = axs[A]
  ax.scatter(-10,-10, label=f'{cn} component', color=c)
  plot_image(X, ax)
  ax.legend()
  

do(1,2, 'b', 'Blue')
do(2,1, 'g', 'Green')
do(3,0, 'r', 'Red')




# %%
