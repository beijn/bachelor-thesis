# %% [markdown]
# I need to use tiling of the images and masks because independent binary masks explode RAM
# TODO overlap and merges to final prediction

# %%
import itertools as it

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

from torchvision.io import read_image, write_png
# TODO make it torchvision independent so that I can put it in util
import colorcet 
import itertools as it

colors = colorcet.m_glasbey.colors


import sys, os; sys.path += ['..']  # NOTE find shared modules
from util.cache import *
from util.preprocess import *

def setup_cache(datadir, clear=False):
  """NOTE: run this function!!"""
  global cache_dir, dataset_id
  dataset_id = datadir
  cache_dir = mk_cache(f'data/{dataset_id}/tiles', ['images', 'masks', 'points'], clear=clear)
  return cache_dir

def make_tiles_dummyBox(imgid='1', points_file='annotations.json', R = 25, TILESIZE = 512, overlap = 0.5, every_nth=1):
  """ TODO: implement overlap and merge """
  global cache_dir, dataset_id

  image = read_image(f"../data/{dataset_id}/{imgid}.jpg")
  points = (preprocess_point(f"../data/{dataset_id}/{points_file}")+.5).astype(int)[::every_nth]

  tilesIxP = ([
    
    ( ix, iy
    , image[:, x:(X:=(x+TILESIZE)), y:(Y:=(y+TILESIZE))].clone()
    , [(u-x,v-y) for (u,v) in points if u-R > x and u+R < X 
                                    and v-R > y and v+R < Y] )
                                  # u+R > x  # to include partial boxes
    # NOTE for some reason masks that are no _fully_ in the image becmoe empty which explodes later code

    for ix, x in enumerate(range(0, image.shape[1], TILESIZE))
    for iy, y in enumerate(range(0, image.shape[2], TILESIZE))])

  tileids = []
  for ix, iy, img, pts in tilesIxP:
    masks = np.zeros((len(pts), *img.shape[1:]), dtype=np.int8)  # NOTE: beware the different channel order conventions

    for i, (x,y) in enumerate(pts):
      masks[i, x-R:x+R, y-R:y+R] = 1

    tileid = f'{imgid}-{ix}-{iy}'
    tileids.append(tileid)

    write_png(img, os.path.join(cache_dir, 'images', f'{tileid}.png'))
    np.save(os.path.join(cache_dir, 'masks', f'{tileid}.npy'), np.packbits(masks))
    np.save(os.path.join(cache_dir, 'points', f'{tileid}.npy'), pts)

    del masks  # free RAM

  return cache_dir, tileids


def get_tiles(imgid='1', nd=False):
  global cache_dir, dataset_id
  # split the string once at . from the right and take the first part
  tileids = sorted([x.rsplit('.', 1)[0] for x in os.listdir(os.path.join(cache_dir, 'images')) if x.startswith(imgid)])

  # the tiles have the format imgid-x-y. return an array with the x and y coordinates
  
  if nd:
    array = []
    X = None
    for id in tileids:
      x = id.split('-')[1]
      if X != x: array += [[]]
      array[-1].append(id)
      X = x 
    return array

  return tileids

def load_tile(tileid=0):
  global cache_dir, dataset_id

  if isinstance(tileid, int):
    tileid = sorted(os.listdir(os.path.join(cache_dir, 'images')))[tileid].split('.', 1)[0]
  elif isinstance(tileid, list):
    tileid = '-'.join(map(lambda i: f'{i}', tileid))
  
  img = read_image(os.path.join(cache_dir, 'images', f'{tileid}.png'))
  seg = np.unpackbits(np.load(os.path.join(cache_dir, 'masks', f'{tileid}.npy'))).reshape(-1, *img.shape[1:])
  pts = np.load(os.path.join(cache_dir, 'points', f'{tileid}.npy'))
 
  return img, seg, pts

def masks2mask(masks):
  """Merge multiple binary masks into one with different labels per object. Note that later objects may overwrite earlier ones."""
  # NOTE: because of high overlaps we save each mask in different layers
  mask = np.zeros(masks.shape[1:], dtype=np.int16)
  for i, m in enumerate(masks):
    mask[m==1] = i+1
  return mask

def plot_tiles(imgid, scale=6, image=True, gt_mask=False, points=True, posts=[lambda ax, i, s, p: None], suptitle=""):
  global cache_dir, dataset_id
  tileids = get_tiles(imgid, nd=True)
  X,Y = len(tileids), len(tileids[0])

  fig, axs = plt.subplots(ncols=Y, nrows=X, figsize=(scale*Y, scale*X))
  for tile, ax, post in zip(it.chain(*tileids), axs.flatten(), it.cycle(posts)):
    img, seg, pts = None, None, None
    try: 
      img, seg, pts = load_tile(tile)
      cs = list(it.islice(it.cycle(colors), len(pts)))
      
      if image: ax.imshow(img.permute(1,2,0))
      if gt_mask: ax.imshow(label2rgb(masks2mask(seg), np.array(img.permute(1,2,0)), bg_color=None, alpha=0.5, saturation=1, colors=cs))
    except FileNotFoundError: pass  # skip tile holes 
    
    if post: post(ax, img, seg, pts)

    if points and pts is not None: 
      ax.scatter(pts[:,1], pts[:,0], c=(cs if gt_mask else 'blue'), s=4*scale)
      # NOTE no legend because it would overwrite everythin from posts
    ax.axis('off')

  if suptitle: fig.suptitle(suptitle)
  plt.tight_layout()

  return fig, axs



if __name__ == '__main__':
  # make_tiles_dummyBox()
  cache_dir = setup_cache('third', clear=True)
  make_tiles_dummyBox()
  plot_tiles(
    imgid='1',
    scale=6,
    image=True,
    gt_mask=True,
    points=True,
    suptitle=f'Tiled dummy masks from Stuart\'s points. Note that overlapping masks are saved independently.',
  )



