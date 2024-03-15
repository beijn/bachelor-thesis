
# %% [markdown]
# # Edge Detection and Segmnetation with Classical Methods
# First goal is to separate FG from BG using as little user info about cells as possible.
# As a second step I can use that as a prior for instances.
#
# https://scikit-image.org/docs/stable/user_guide/tutorial_segmentation.html
# - Canny
# - Sobel
# - Watershed
#
# ## General Notes
# - PC artifacts rise from the BG and fall almost instantly in the cell [prove]
# - not all cell boundaries have such artefacts
# %%

import onnxruntime as ort
import numpy as np
import pandas as pd
import scipy as sc
import skimage 
import matplotlib.pyplot as plt

import sys; sys.path += ['..']  # NOTE find shared modules
from util.preprocess import *
from util.plot import *



def norm(x):
  return (x - x.min()) / (x.max() - x.min())


dataset = 'third'

pos = preprocess_point(f'../data/{dataset}/points.json')

data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+')
data

# %%

model_path = "../data/phaseimaging-combo-v3.onnx"

def get_sess(model_path: str):
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "gpu_mem_limit": int(
                    8000 * 1024 * 1024
                ),  # parameter demands value in bytes
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    sess_opts: ort.SessionOptions = ort.SessionOptions()
    sess_opts.log_severity_level = 2
    sess_opts.log_verbosity_level = 2
    sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
    return sess

sess = get_sess(model_path=model_path)
output_names = [out.name for out in sess.get_outputs()]
input_name = sess.get_inputs()[0].name

# %% 

fig, axs = mk_fig(2,3, shape=skimage.io.imread(f'../data/{dataset}/1.jpg').shape[:2])
axs = axs.ravel()

fig, bxs = mk_fig(2,3, shape=skimage.io.imread(f'../data/{dataset}/1.jpg').shape[:2])
bxs = bxs.ravel()

for i, (_, row) in enumerate(data.query('objective == "40x"').iterrows()):
  I = skimage.io.imread(f'../data/{dataset}/{row["imgid"]}.jpg').astype(np.float32) / 255
  G = norm(I[...,1])

  input = np.expand_dims(np.transpose(I[..., [2, 1]], (2, 0, 1)), axis=0)
  pred = sess.run(output_names, {input_name: input})[0]

  thresh = 0.01
  mask = (pred[0, 0] > thresh).reshape(I.shape[:2])
  plot_image(
      skimage.color.label2rgb(1 - mask, G, bg_color=None, alpha=0.5),
      #title=f"Mask (Sten's fake phase reconstruction, threshold {thresh:.2f}): {row['imgid']}",
      ax = axs[i]
  )

  mark = np.zeros(I.shape[:2], dtype=int)
  mark[pos[:, 0], pos[:, 1]] = np.arange(len(pos)) + 1
  # mark[neg[:,0], neg[:,1]] = -1

  ws = skimage.segmentation.watershed(G, mask=mask, markers=mark)
  ws[ws == 1] = 0

  plot = skimage.color.label2rgb(ws, G, bg_color=None, alpha=0.4, colors=colors)
  # plot = skimage.segmentation.mark_boundaries(plot, ws, color=colors)
  plot_image(plot, ax=bxs[i],
  #           title="Watershed Segmentation"
             )
  bxs[i].scatter(pos[:, 1], pos[:, 0], c=list(mk_colors(pos)), marker="o", s=12)




# %% [markdown]
# ### Note this is bad because the points are from a different image

  