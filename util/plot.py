import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import colorcet
import pandas as pd
import skimage
import itertools as it
from matplotlib.patches import Rectangle
import torch

colors = colorcet.m_glasbey.colors

__SCALE__ = 6
__SHAPE__ = (1,1)


def plot_set_scale(scale):
  global __SCALE__
  __SCALE__ = scale

def plot_set_shape(shape):
  global __SHAPE__
  m = min(shape)
  __SHAPE__ = tuple([s/m for s in shape])


def mk_fig(x=1, y=1, shape=__SHAPE__, scale=1, flat=True, **subplot_args):
  scale = scale * __SCALE__

  m = min(shape)
  shape = tuple([s/m for s in shape])

  fig, axs = plt.subplots(nrows=y, ncols=x, figsize=(shape[1]*scale*x, shape[0]*scale*y), **subplot_args)
  plt.tight_layout()
  
  if x == 1 and y == 1: return fig, axs

  if x == 1: axs = [axs]
  if y == 1: axs = [[ax] for ax in axs]

  axs = np.array(axs)
  if flat: axs = axs.ravel()
  return fig, axs

def plot_point(points, ax, c=None):
  if isinstance(points, torch.Tensor):
    points = points.detach().cpu().numpy()

  ax.scatter(points[:,0], points[:,1],
             c=mk_colors(points) if c is None else c, s=18/2)
  return ax

def plot_image(image, ax=None, scale=1, title=None, cmap='gray'):
  assert len(image.shape) in (2,3,4), f"Unknown image shape: {image.shape}, expected 2, 3, or 4 dims."

  if ax is None: fig, ax = mk_fig(scale=scale)

  if isinstance(image, torch.Tensor): image = image.detach().cpu().numpy()

  if len(image.shape) == 4: image = image[0]  # scrap batch dim
  if len(image.shape) == 3 and image.shape[0] == 1: image = image[0]  # scrap the channel dim if it was one
  if len(image.shape) == 3 and image.shape[0] <= 4: image = np.moveaxis(image, 0, -1)  # make the (RGBA) channel last if it was first

  ax.imshow(image, cmap=cmap)
  if title: ax.set_title(title)
  ax.axis('off')
  plt.tight_layout()
  return ax


def mk_colors(iter, scale=float):
  assert scale in (float, int)
  
  t = (lambda x: int(x*255)) if scale == int else (lambda x: x)
  cs = ((t(r), t(g), t(b)) for (r,g,b),_ in zip(it.cycle(colors), iter))
  return list(cs)


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




def plot_animation(snapshots, export=False, logify_scalars=False):
  from matplotlib.animation import FuncAnimation, FFMpegWriter
  import matplotlib.pyplot as plt
  import numpy as np

  fig, ax = mk_fig(1,1)

  FRAMES = len(LOG)

  H = target.shape[-2]

  losses, lrs = [ (
    y := np.array([x[dim] for x in LOG]),
    y := np.log(y) if logify_scalars else y,
    m := y.min() if logify_scalars else 0,
    H - (y - m) / (y.max() - m) * H
    )[-1]
    
    for dim in ['loss', 'lr']
  ]
 
  def animate(i):
    epoch, loss, lr, pred = [LOG[i][k] for k in 'epoch loss lr pred'.split()]
    pred = pred[0]  # only one channel in this case

    H,W = pred.shape

    ax.clear()
    xs = np.linspace(0, W*(i-1)/FRAMES, i)
    ax.plot(xs, losses[:i], label=f"{'log' if logify else ''} loss")
    ax.plot(xs, lrs[:i], label=f"{'log' if logify else ''} lr")

    #plot_image(pred, ax=ax0, title=f"Predicted Heatmap")
    title = F"Predicted Heatmap Overlayed on Image"
    heat = norm(np.stack([pred, Z:=np.zeros_like(pred),Z,pred], axis=-1))
    plot_image(inp, ax=ax, title=title)
    plot_image(heat, ax=ax, title=title)
    ax.legend();


  anim = FuncAnimation(fig, animate, frames=FRAMES, interval=100)
 
  if export:  
    import os
    os.system("module load ffmpeg || echo could not: module load ffmpeg")
    writervideo = FFMpegWriter(fps=30) 
    anim.save('../runs/smp/unet-heatmap/training-convergence.mp4', writer=writervideo) 
  else:
    from IPython import display
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()
