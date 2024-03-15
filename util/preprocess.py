import numpy as np
import json


def preprocess_point(annot_file, dtype=np.float32, unique=True, device=False):
  annot = json.load(open(annot_file))
  out = np.zeros((len(annot[0]['annotations'][0]['result']), 2))

  for i, result in enumerate(annot[0]['annotations'][0]['result']):
    h,w = result['original_height'], result['original_width']

    x = result['value']['x']/100 * w
    y = result['value']['y']/100 * h

    # rot = result['image_rotation]
    #s = result['value']['width']
    #n = result['to_name']
    
    out[i,0], out[i,1] = x,y  # NOTE: THIS WAS SWAPPED AND OLDER CODE MAY FAIL NOW
  
  if not device:
    out.astype(dtype)

  if device:
    import torch
    out = torch.from_numpy(out).to(device).to(dtype)

  return out


def preprocess_bbox(file):
  raw = json.load(open(file))
  out = np.zeros((len(raw[0]['annotations'][0]['result']), 4))

  for i, result in enumerate(raw[0]['annotations'][0]['result']):
    H,W = result['original_height'], result['original_width']

    x = result['value']['x']/100 * W
    y = result['value']['y']/100 * H
    w = result['value']['width']/100 * W
    h = result['value']['height']/100 * H

    out[i] = [x,y,w,h]  # NOTE that x y order is normal again

  return out