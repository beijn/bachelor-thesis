# %% [markdown]
# # Different Channels have different clarity
# %%

import skimage

import matplotlib.pyplot as plt

import sys, os

sys.path += [".."]  # NOTE find shared modules
from util.preprocess import *
from util.plot import *

import pandas as pd


def norm(x):
  return (x - x.min()) / (x.max() - x.min())


dataset = 'third'

data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+')
data

# %%


fig, axs = mk_fig(3,6, shape=skimage.io.imread(f'../data/{dataset}/1.jpg').shape[:2])
axs = axs.ravel()

for i, (_, row) in enumerate(data.query('objective == "40x"').iterrows()):
  I = skimage.io.imread(f'../data/{dataset}/{row["imgid"]}.jpg')
  
  rx, gx, bx = axs[3*i:3*i+3]

  plot_image(I[:,:,0], rx)
  plot_image(I[:,:,1], gx)
  plot_image(I[:,:,2], bx)

  # # write the green channel
  # os.system('mkdir -p yellow')

  # # scale the green channel to 0-255
  # Y = I.copy()
  # Y[:,:,2] = Y
  # Y = (norm(Y) * 255).astype(np.uint8)
  # Y[:,:,2] = 255
  # skimage.io.imsave(f"yellow/{row['imgid']}-Y.png", Y)


# %%
