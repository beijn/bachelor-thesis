import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorcet
import pandas as pd
import skimage
import itertools as it
from matplotlib.patches import Rectangle


colors = colorcet.m_glasbey.colors
__SCALE__ = 18



def mk_fig(x=1, y=1, shape=(1,1), scale=__SCALE__, flat=True, **subplot_args):
  shape = np.array(shape)[:2]
  shape = shape / shape.min()

  fig, axs = plt.subplots(nrows=y, ncols=x, figsize=(shape[1]*scale*x, shape[0]*scale*y), **subplot_args)
  plt.tight_layout()
  
  if x == 1 and y == 1: return fig, axs

  if x == 1: axs = [axs]
  if y == 1: axs = [[ax] for ax in axs]

  axs = np.array(axs)
  if flat: axs = axs.ravel()
  return fig, axs


def imshow(im, ax=None, fig=None, kw_subplot={}, kw_imshow={}, scale=__SCALE__, title=""):

  kw_subplot = {
    'figsize': (4*scale,3*scale), 
    **kw_subplot
  }

  kw_imshow = {
    'cmap': 'gray',
    **kw_imshow
  }

  if ax is None: 
    fig, ax = plt.subplots(1,1, **kw_subplot)
  
  ax.imshow(im, **kw_imshow)
  ax.set_title(title)
  ax.axis('off')
  plt.tight_layout()

  return fig, ax
  


def plot_image(image, ax=None, scale=__SCALE__, title=""):
  if ax is None:
    fig, ax = mk_fig(scale=scale)

  ax.imshow(image, cmap='gray')
  ax.set_title(title)
  ax.axis('off')
  plt.tight_layout()
  return ax


def mk_colors(iter, scale=float):
  assert scale in (float, int)
  t = (lambda x: int(x*255)) if scale == int else (lambda x: x)
  cs = ((t(r), t(g), t(b)) for (r,g,b),_ in zip(it.cycle(colors), iter))
  return cs

def plot_mask(mask, image=None, color='red', ax=None, **figargs):
  if not ax: fig, ax = plt.subplots(1,1, **figargs)
  ax.axis('off')

  if isinstance(image, np.ndarray): ax.imshow(image)

  def mk_cmap():
    cmap = plt.cm.__dict__[color.capitalize()+'s']
    cmap = cmap(np.arange(N:=cmap.N))
    cmap[:,-1] = np.linspace(0, 0.5, N)
    return matplotlib.colors.ListedColormap(cmap)

  ax.contourf(mask, cmap=mk_cmap())

  return ax

def plot_boxes(boxes, ax):
  cs = mk_colors(boxes)
  for (x,y,w,h), c in zip(boxes, cs):
    rect = Rectangle((x,y), w, h, linewidth=1, edgecolor=c, facecolor='none')
    ax.add_patch(rect)


def plot_instseg(image=None, results=pd.DataFrame(), dims=['Var 1', 'Var 2'], what='Undefined'):
  global colors

  fig, axs = plt.subplots(len(results.columns), len(results.index), figsize=(10, 10))
  if len(results.index)   <= 1: axs = [axs]
  if len(results.columns) <= 1: axs = [[ax] for ax in axs]

  plt.tight_layout()

  for row, Y in zip(axs, results.index):
    for ax, X in zip(row, results.columns):
      ax.set_title(f"{what}: {dims[0]}={X}, {dims[1]}={Y}")
      ax.axis('off')

      insts = results[X][Y]

      mask = np.zeros(image.shape[:2])
      for i, inst in enumerate(insts):
        mask[inst['segmentation']] = i+1

      ax.imshow(skimage.color.label2rgb(
        mask, image, saturation=1, bg_color=None, alpha=0.5, colors=colors)
      )

      for color, inst in zip(colors, insts):
        ax.contour(inst['segmentation'], colors=[color], linewidths=2)

  return fig, axs
