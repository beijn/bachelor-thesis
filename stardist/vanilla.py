# %% [markdown]
# # [Stardist](https://github.com/stardist/stardist)

# %%
import sys, os; sys.path += [os.path.join(os.path.expanduser('~'), 'thesis')]  # NOTE hardcoded project root to find shared util modules
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

import skimage

from util.plot import *


model = StarDist2D.from_pretrained('2D_versatile_he')

images = [f'../data/third/{i}.jpg' for i in [1,2,3,4]]

fig, axs = plt.subplots(2, 2, figsize=(13, 10))
plt.tight_layout()

for ax, image, imgid in zip(axs.flat, images, [1,2,3,4]):
  image = skimage.io.imread(image)

  masks, _ = model.predict_instances(normalize(image))
  ax.set_title(f"Stardist (vanilla): third/{imgid}.jpg")

  plot_mask(masks, image, 'red', ax=ax);
