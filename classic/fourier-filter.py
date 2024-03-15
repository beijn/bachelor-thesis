# %% [markdown]
# # Something in Fourier Domain
# NOTE designed to run only locally interactively
# NOTE hangs up when a pressed mousebutton leaves the figure area
# %%

import sys; sys.path += [".."]  # NOTE find shared modules

from util.preprocess import *
from util.plot import *

import skimage
from scipy.fft import fft2, ifft2, fftshift
import cv2

def norm(x): return (x - x.min()) / (x.max() - x.min())


polar = cv2.linearPolar #cv2.logPolar

dataset = 'third'
imgid = 1

SLICE = 512  # TODO use the largest square slice

image = norm(skimage.io.imread(f"../data/{dataset}/{imgid}.jpg")[..., 1])
image = image[:SLICE, :SLICE]

X = image
#
W, H = X.shape[:2]
F = np.log(fftshift(fft2(X)))

G = ifft2(fftshift(np.exp(F)))

fig, (a,b,c,d) = mk_fig(2,2)
plot_image(X, a, title=f"Green Channel of {dataset}/{imgid}")
plot_image(np.abs(F), b, title=f"Fourier Transform")
plot_image(G.real, c, title=f"Reversed Fourier Transform")

P = polar(np.abs(F),(H/2, H/2), H/2, 0)  # NOTE for some reason H/... must be adapted with image size  #256-.4.5, 512-6.2, with Linear /2 always works
plot_image(P, d, title="Log(!) Polar Transformed FT")

p = norm(P.sum(axis=0))
d.plot(H-p*H, label="Spectral Sum")
d.legend();


# %% 
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
import numpy as np


fig, (ax, ax_diff, ax_mult, ax_mult2) = mk_fig(2,2)
ax.plot(H-p*H, label="Spectral Sum")
plot_image(X, title=f"Filter with Spatial Frequency Filter", ax=ax);
plot_background = ax.imshow(X)

plot_image(X, ax_diff, title="Difference between Filtered and Original")
plot_diff = ax_diff.imshow(X)

plot_image(X, ax_mult, title="Image Amplified by Difference")
plot_mult = ax_mult.imshow(X)


plot_image(X, ax_mult2, title="Difference between Amplified and Original")
plot_mult2 = ax_mult2.imshow(X)



%matplotlib ipympl


class PointEditor():
  def __init__(self, fig, ax, width, height, **kwargs):
    [setattr(self, k, v) for k,v in {**dict(
      xmin = 0,
      xmax = width,
      ymin = 0,
      ymax = height,
      points_n = 10,
      fig = fig,
      ax = ax,
      width = width,
      height = height,
      active_point = None,
      interp = lambda x,y: scipy.interpolate.CubicHermiteSpline(x,y, dydx=np.zeros_like(x)),
      interp_n = 100,
    ), **kwargs}.items()]

    self.points_x = np.linspace(self.xmin, self.xmax, self.points_n) 
    self.points_y = np.ones(self.points_n) * height / 2
    
    self.plot_points, = self.ax.plot(self.points_x, self.points_y, linestyle='none',marker='o',markersize=8)
    self.plot_interp, = self.ax.plot(*self.interpolate(), 'r-', label='interpolation')

    [self.fig.canvas.mpl_connect(id, fun) for id, fun in [
      ('button_press_event', self.callback_press),
      ('button_release_event', self.callback_release),
      ('motion_notify_event', self.callback_move)
    ]]


  def interpolate(self, interp_n=None):
    if not interp_n: interp_n = self.interp_n
    interp_x = np.linspace(self.xmin, self.xmax, self.interp_n) 
    return interp_x, self.interp(self.points_x, self.points_y)(interp_x)

  def set_point(self, x,y, index):
    # clamp to valid range 
    self.points_x[index] = np.clip(x, self.xmin, self.xmax)
    self.points_y[index] = np.clip(y, self.ymin, self.ymax)


    # sort the points according to their x-value
    indices = np.argsort(self.points_x)
    self.points_x = self.points_x[indices]
    self.points_y = self.points_y[indices]

    # update the index of the active point
    self.active_point = np.where(indices == index)[0][0]

    self.plot_points.set_xdata(self.points_x)
    self.plot_points.set_ydata(self.points_y)

    self.plot_interp.set_ydata(self.interpolate()[1])
    self.fig.canvas.draw_idle()

    self.updater(self.interp(self.points_x, self.points_y))

  def callback_press(self, event):
    if event.inaxes is None \
    or event.button != 1: return

    self.active_point = self.which_point_clicked(event)

  def which_point_clicked(self, event):
    t = self.ax.transData.inverted()
    tinv = self.ax.transData 
    xy = t.transform([event.x,event.y])
    xr = np.reshape(self.points_x,(np.shape(self.points_x)[0],1))
    yr = np.reshape(self.points_y,(np.shape(self.points_y)[0],1))
    xy_vals = np.append(xr,yr,1)
    xyt = tinv.transform(xy_vals)
    xt, yt = xyt[:, 0], xyt[:, 1]
    d = np.hypot(xt - event.x, yt - event.y)
    indseq, = np.nonzero(d == d.min())
    ind = indseq[0]
    if d[ind] >= 8: ind = None   # the distance in pixel which is too far from the point
    return ind
  
  def callback_release(self, event):
    if event.button != 1: return
    self.active_point = None

  def callback_move(self, event):
    if self.active_point is None \
    or event.inaxes is None \
    or event.button != 1:
      return
  
    self.set_point(event.xdata, event.ydata, index=self.active_point)


def update_image(interpolator):
  x = np.linspace(0,W,W)
  y = 1-norm(interpolator(x))
  P = np.stack([y for _ in range(H)])
  Q = polar(P, (H/2, H/2), H/2, cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS).T
  
  f = F * Q


  R = norm(ifft2(fftshift(np.exp(f))).real)
  plot_background.set_data(R + 0.3*Q)
    
  D = norm(np.abs(norm(X)-norm(R)))
  plot_diff.set_data(D)
  plot_mult.set_data(norm(X*D))
  plot_mult2.set_data(norm(X-(X*D)))
    
  for artist in [plot_background, plot_mult, plot_mult2, plot_diff]:
    artist.set_clim(0,1)

    
editor = PointEditor(fig, ax, width=W, height=H, updater=update_image)

plt.show()