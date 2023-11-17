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

    if dims=='xy':
      out[i,:] = [x,y]

  return out.astype(np.int16)
