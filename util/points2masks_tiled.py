# %% [markdown]
# I need to use tiling of the images and masks because independent binary masks explode RAM
# TODO overlap and merges to final prediction

# %%
import itertools as it

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.io import imread, imsave
from skimage.color import label2rgb


import colorcet 
import itertools as it

colors = colorcet.m_glasbey.colors


import sys, os; sys.path += ['..']  # NOTE find shared modules
from util.cache import *
from util.preprocess import *


def test_inside(inside_mode, R):
  if inside_mode == 'partial':
    inside = lambda u,v,x,y,X,Y: u+R >= x and u-R < X \
                             and v+R >= y and v-R < Y
  elif inside_mode == 'center':
    inside = lambda u,v,x,y,X,Y: u >= x and u < X \
                            #  and v >= y and v < Y
  elif inside_mode == 'complete':
    inside = lambda u,v,x,y,X,Y: u-R >= x and u+R < X \
                             and v-R >= y and v+R < Y
  return inside

def setup_cache(datadir, token, clear=False):
  """NOTE: run this function!!"""
  global cache_dir, dataset_id
  dataset_id = datadir
  cache_dir = mk_cache(f'data/{dataset_id}/{token}/tiles', ['images', 'masks', 'points'], clear=clear)
  return cache_dir

def make_tiles_dummyBox(imgid='1', points_file='annotations.json', bg_points_file='annot-bg.json', 
                        R = None, TILESIZE = 512, overlap = 128, inside_mode='partial'):
  """ TODO: implement overlap and merge """
  global cache_dir, dataset_id
  r = R if R else 0

  image = imread(f"../data/{dataset_id}/{imgid}.jpg")
  points = (preprocess_point(f"../data/{dataset_id}/{points_file}")+.5).astype(int)
  points_bg = (preprocess_point(f"../data/{dataset_id}/{bg_points_file}")+.5).astype(int)

  inside = test_inside(inside_mode, r)

  tilesIxP = ([
    ( ix, iy
    , image[x:(X:=(x+TILESIZE)), y:(Y:=(y+TILESIZE)), :].copy()
    , [(u-x,v-y) for (u,v) in points    if inside(u,v,x,y,X,Y)]
    , [(u-x,v-y) for (u,v) in points_bg if inside(u,v,x,y,X,Y)]
    )

    for ix, x in enumerate(range(0, image.shape[0], TILESIZE))
    for iy, y in enumerate(range(0, image.shape[1], TILESIZE))])

  tileids = []
  for ix, iy, img, pts, pts_bg in tilesIxP:
    tileid = f'{imgid}-{ix}-{iy}'
    tileids.append(tileid)

    # swap x and y in the points
    pts = [(y,x) for (x,y) in pts]
    pts_neg = [(y,x) for (x,y) in pts_bg]
  
    imsave(os.path.join(cache_dir, 'images', f'{tileid}.png'), img)
    np.save(os.path.join(cache_dir, 'points', f'{tileid}.fg.npy'), pts)
    np.save(os.path.join(cache_dir, 'points', f'{tileid}.bg.npy'), pts_bg)

    if R: 
      masks = np.zeros((len(pts), *img.shape[:2]), dtype=np.int8)  # NOTE: beware the different channel order conventions

      for i, (x,y) in enumerate(pts):
        masks[i, x-r:x+r, y-r:y+r] = 1

      np.save(os.path.join(cache_dir, 'masks', f'{tileid}.npy'), np.packbits(masks))

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
  
  img, seg, pts, pts_bg = None, None, None, None

  if os.path.exists(p:=os.path.join(cache_dir, 'images', f'{tileid}.png')):
    img = imread(p)
  if os.path.exists(p:=os.path.join(cache_dir, 'masks', f'{tileid}.npy')):
    seg = np.unpackbits(np.load(p)).reshape(-1, *img.shape[:2])
  if os.path.exists(p:=os.path.join(cache_dir, 'points', f'{tileid}.fg.npy')):
    pts = np.load(p)
  if os.path.exists(p:=os.path.join(cache_dir, 'points', f'{tileid}.bg.npy')):
    pts_bg = np.load(p)
 
  return img, seg, pts, pts_bg


def masks2mask(masks):
  """Merge multiple binary masks into one with different labels per object. Note that later objects may overwrite earlier ones."""
  # NOTE: because of high overlaps we save each mask in different layers
  mask = np.zeros(masks.shape[1:], dtype=np.int16)
  for i, m in enumerate(masks):
    mask[m==1] = i+1
  return mask

def plot_tiles(imgid, scale=6, image=True, gt_mask=False, points=True, points_bg=True, posts=[lambda ax, stuff: None], suptitle=""):
  global cache_dir, dataset_id
  tileids = get_tiles(imgid, nd=True)
  X,Y = len(tileids), len(tileids[0])

  fig, axs = plt.subplots(ncols=Y, nrows=X, figsize=(scale*Y, scale*X))
  for i, (tile, ax, post) in enumerate(zip(it.chain(*tileids), axs.flatten(), it.cycle(posts))):
    img, seg, pts, pts_neg = load_tile(tile)
    
    if image: ax.imshow(img)

    if points and pts is not None and len(pts) > 0: 
      cs = list(it.islice(it.cycle(colors), len(pts)))
      pts = [(x,y) for (x,y) in pts if x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]]
      ax.scatter(*zip(*pts), c=(cs if gt_mask else 'blue'), s=4*scale, marker='o')
    else: it.cycle(colors)

    if gt_mask: ax.imshow(label2rgb(masks2mask(seg), img, bg_color=None, alpha=0.5, saturation=1, colors=cs))

    if points_bg and pts_neg is not None and len(pts_neg) > 0:
      pts_neg = [(x,y) for (x,y) in pts_neg if x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]]
      ax.scatter(*zip(*pts_neg), c='red', s=4*scale, marker='D')
    
    if post: post(ax, dict(
      img = img,
      seg = seg,
      pts = pts,
      pts_bg = pts_neg,
      idx = i
    ))
      
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


# TODO: tiles with halo
# TODO: merge tiles with halo