# %% [markdown]
# # [microSAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)
# see https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/use_as_library/instance_segmentation.py

# %%
import sys, os; sys.path += ['..']  # NOTE find shared modules

from micro_sam import instance_segmentation, util
import imageio.v3 as imageio
import pandas as pd
import itertools as it
import os
from datetime import datetime

from util.label_studio_converter__brush import mask2rle
from util.plot import *


# %%
dataset_id = 'third'

model_types = 'vit_b vit_l vit_h'.split(' ')
iou_thresh = 0.88

cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'thesis', 'micro-sam', 'TEMPORARY-timing', dataset_id)
[os.makedirs(os.path.join(cache_dir,d), exist_ok=True) for d in 'embed masks rles'.split(' ')]

# %%

out = pd.DataFrame()

for model_type in model_types:
  T0 = datetime.now()
  predictor = util.get_sam_model(model_type=model_type)
  T_load_predictor = (datetime.now() - T0).total_seconds()

  for imgid in [1]:

    pImage = f"../data/{dataset_id}/{imgid}.jpg"
    pEmbed = f"{cache_dir}/embed/{imgid}-{model_type}.zarr"
    pMasks = f"{cache_dir}/masks/{imgid}-{model_type}.npy"
    pRLEs  = f"{cache_dir}/rles/{imgid}-{model_type}.str"

    image = imageio.imread(pImage)

    print('LOADING'  if os.path.exists(pEmbed) else
          'WRITING', f'embedding cache at {pEmbed}')

    T0 = datetime.now()
    embeddings = util.precompute_image_embeddings(
      predictor, image, ndim = 2, save_path=pEmbed,
      tile_shape=(tile:=1024, tile), halo=(halo:=tile//4, halo)
    )
    T_embed = (datetime.now() - T0).total_seconds()

    T0 = datetime.now()
    amg = instance_segmentation.TiledAutomaticMaskGenerator(predictor)
    T_load_amg = (datetime.now() - T0).total_seconds()

    T0 = datetime.now()
    amg.initialize(image, embeddings, verbose=True)
    insts = amg.generate(pred_iou_thresh=iou_thresh)  # can try different

    masks = instance_segmentation.mask_data_to_segmentation(
      insts, shape=image.shape, with_background=True)        ## TODO: what does this function do with overlapping instances?
    T_segment = (datetime.now() - T0).total_seconds()

    np.save(pMasks, masks)

    rles = [mask2rle(I['segmentation'].astype(np.uint8)) for I in insts]
    with open(pRLEs, 'w') as f: f.write(str(rles))

    out = pd.concat([out, pd.DataFrame(dict(
      model = [model_type],
      imgid = [imgid],
      image = [pImage],
      embed = [pEmbed],
      masks = [pMasks],
      rles  = [pRLEs],
      T_embed = [T_embed],
      T_load_predictor = [T_load_predictor],
      T_load_amg = [T_load_amg],
      T_segment = [T_segment],
    ))], ignore_index=True)

# %%
out
# %%
colors = colorcet.m_glasbey.colors

fig, axs = plt.subplots(3, 1, figsize=(13/2, 10*3/2))
plt.tight_layout()

for ax, (_, it) in zip(axs.flat, out.iterrows()):
  image = imageio.imread(it['image'])
  masks = np.load(it['masks'])

  ax.set_title(f"µSAM ({it['model']}): third/{it['imgid']}.jpg. embed: {it['T_embed']:.2f}s, segment: {it['T_segment'] + it['T_load_amg']:.2f}s'")
  ax.axis('off')

  ax.imshow(skimage.color.label2rgb(
    masks, image, saturation=1, bg_color=None, alpha=0.5, colors=colors)
  )
