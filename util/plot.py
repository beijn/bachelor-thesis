import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
