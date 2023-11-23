import numpy as np
import json


def preprocess_point(annot_file = '../data/third/annotations.json', dims='yx'):
  annot = json.load(open(annot_file))
  out = np.zeros((len(annot[0]['annotations'][0]['result']), 2))

  for i, result in enumerate(annot[0]['annotations'][0]['result']):
    h,w = result['original_height'], result['original_width']

    x = result['value']['x']/100 * w
    y = result['value']['y']/100 * h

    # rot = result['image_rotation]
    #s = result['value']['width']
    #n = result['to_name']

    out[i,0] = y    # NOTE x y order
    out[i,1] = x
    # NOTE: its swapped but now my code depends on it... need to refactor this later

    if dims=='xy':
      out[i,:] = [x,y]

  return out.astype(np.int16)


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