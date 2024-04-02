# %%

import scipy as sc
import numpy as np


#points = preprocess_point("../data/third/points.json")


targ = np.load('../runs/smp/unet-watershed/targ.npy')
pred = np.load('../runs/smp/unet-watershed/pred.npy')
#targ2 = np.load('../runs/smp/unet-watershed/targ2.npy')
pred2 = np.load('../runs/smp/unet-watershed/pred2.npy')
#targ4 = np.load('../runs/smp/unet-watershed/targ4.npy')
pred4 = np.load('../runs/smp/unet-watershed/pred4.npy')


from scipy.ndimage import label
from skimage.color import label2rgb

connect = np.ones((3, 3), dtype=int)  # this defines the connection filter

"""
dos = [
  ("train image ", targ, pred,  1766),
  ("test image 1", targ2, pred2, 1198),
  ("test image 2", targ4, pred4,  980)
]


for desc, targ, pred, refcount in dos:
  targ = (targ < 0.5).astype(int)
  pred = (pred < 0.5).astype(int) 

  label_targ, ncomps_targ = label(targ, connect)
  label_pred, ncomps_pred = label(pred, connect)

  small_targ = 0
  small_pred = 0

  # filter connected components that are greater than 10 pixels
  for i in range(1, ncomps_targ+1):
    if (label_targ == i).sum() < 100:
      label_targ[label_targ == i] = 0
      small_targ += 1

  background = ncomps_targ - small_targ - refcount

  for i in range(1, ncomps_pred+1):
    if (label_pred == i).sum() < 100:
      label_pred[label_pred == i] = 0
      small_pred += 1

  print(f"{desc}: annotated: {refcount}, background: {background}, objects: {ncomps_targ}, small objects: {small_targ}, corrected cell count: {ncomps_targ - background - small_targ}")
""";

dos = [
  ("train image ", pred, 1766),
  ("test image 1", pred2, 1198),
  ("test image 2", pred4,  980)
]

for desc, pred, refcount in dos:
  pred = (pred < 0.5).astype(int) 

  label_pred, ncomps_pred = label(pred, connect)

  small_pred = 0

  # filter connected components that are greater than 10 pixels
  for i in range(1, ncomps_pred+1):
    if (label_pred == i).sum() < 100:
      label_pred[label_pred == i] = 0
      small_pred += 1


  print(f"{desc}: annotated: {refcount}, objects: {ncomps_pred}, small: {small_pred}, corrected cell count: {ncomps_pred - small_pred}")
# %%
