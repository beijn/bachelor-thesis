# %% 

from util.plot import *

import pickle

# %%


gt = pickle.load(open('gt_dict.pkl', 'rb'))

for k in gt:
  shape = list(gt[k].shape) if hasattr(gt[k], 'shape') else type(gt[k])
  print(f"{k:10} {shape}")

# %% 

# find the channel with the brughtest scoremap

max_channel = 0
max_score = 0
for i in range(80):
  score = gt['score_map'][0][i].sum()
  if score > max_score:
    max_score = score
    max_channel = i

scoremap = gt['score_map'][0][32-1].cpu().numpy()
plot_image(scoremap)
scoremap.shape
# %%

image = gt['images'].tensor[0].permute(1,2,0).cpu().numpy()
plot_image(image)
image.shape
# %%
128/512

gt['radii']
# %%


data = pickle.load(open('data.pkl', 'rb'))

data
# %%

len(data), data[0].keys()
# %%
