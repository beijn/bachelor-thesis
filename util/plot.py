import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorcet
import pandas as pd
import skimage


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


def plot_instseg(image=None, results=pd.DataFrame(), dims=['Var 1', 'Var 2'], what='Undefined'):
  colors = colorcet.m_glasbey.colors

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
