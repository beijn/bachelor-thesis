# %% [markdown]
# # Interactive Butterworth
# - Interactively find a spatial frequency band pass filter
# - Same Idea as my interactive fourier-filter but less flexible and better implemented
# - Desgined to be run in an interactive session
# %%

import sys; sys.path += [".."]  # NOTE find shared modules
import os 

from util.preprocess import *
from util.plot import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import skimage, scipy

%matplotlib ipympl
# %%

from skimage.filters import scharr as edge  # farif is best but more expensive


norm = lambda x: (x - x.min()) / (x.max() - x.min())
load_image = lambda i: norm(skimage.io.imread(f"../data/third/{i}.jpg")).astype(np.float32)
X = load_image(1)[...,1][:512,:512]

fig, (ax,bx) = mk_fig(2,1)

plot_image(X, ax)
artist = ax.imshow(X)

plot_image(edge(X), bx)
brtist = bx.imshow(edge(X))


_mk = lambda y,t,a,b,i: Slider(plt.axes([0.1, 0.09-i*0.04, 0.8, 0.03]), t, a, b, valinit=i)

Ss = [
  Shi  := _mk(0,'above',0.001,0.499,0.001),
  Slo  := _mk(1,'below',0.001,0.499,0.499),
  Sord := _mk(2,'order',0,10,1),
]

block = False
def replot(val):
  global block
  if block: return
  block = True

  butt = lambda x,c,hl: skimage.filters.butterworth(x, c, high_pass = hl,
    order = Sord.val, squared_butterworth = True, npad = 32)
  
  artist.set_data(b := norm(butt(butt(X,Shi.val,1),Slo.val,0)))
  brtist.set_data(norm(edge(b)))

  block = False

[S.on_changed(replot) for S in Ss]

%matplotlib ipympl
plt.show()

# %%