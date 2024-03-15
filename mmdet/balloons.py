# %% [markdown]
# # MMDetection Framework - CenterNet
# %%

import sys; sys.path += ['..']  # NOTE find shared modules
from util.preprocess import *
from util.plot import *

import skimage

from mmdet.apis import init_detector, inference_detector
import torch 

from mmdet.registry import VISUALIZERS
import cv2
import mmcv


import torch

from os import system as sh


# %%

# transform labelstudio json export to coco
"""import json
import pandas as pd

dataset = 'third'
data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+')
points = json.load(open(f'../data/{dataset}/points.json'))


load_image = lambda i: skimage.io.imread(f"../data/{dataset}/{i}.jpg").astype(np.float32) / 255


__annot_id = 0
coco = dict(
  info = dict(
    year = 2023,
    version = 231201,
    description = 'EVOS with Stuarts constant settings.',
    contributor = 'Stuart Fawke',
  ),
  images = [(
    S := load_image(ID).shape[:2], 
    dict(
      id = ID,
      width = S[1],
      height = S[0],
      file_name = f'{dataset}/{ID}.jpg',
    ))[-1] for ID in data['img']
  ],
  annotations = [(
    W := A['original_width'],
    H := A['original_height'],
    w := A['value']['width']/100 * W,
    h := w,
    x := A['value']['x']/100 * W - w/2,
    y := A['value']['y']/100 * H - h/2,
    dict(
      id = (__annot_id := __annot_id+1),
      img_id = 1,
      category_id = 1,
      segmentation = [],
      area = w*h,
      bbox = [x,y,w,h],
      iscrowd = 0,
    ))[-1] for A in points[0]['annotations'][0]['result']
  ],
  categories = [dict(
    id = 1,
    name = 'cell',
    supercategory = 'cell',
  )]
)
# TODO keypoints? 

json.dump(coco, open(f'../data/{dataset}/points-coco.json', 'w'), indent=2)""";
import os.path as osp

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress


def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)
    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(track_iter_progress(list(data_infos.values()))):
        print(v)
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)

convert_balloon_to_coco(
  ann_file='../data/balloon/train/via_region_data.json',
  out_file='../data/balloon/train/annotation_coco.json',
  image_prefix='../data/balloon/train'
)
convert_balloon_to_coco(
  ann_file='../data/balloon/val/via_region_data.json',
  out_file='../data/balloon/val/annotation_coco.json',
  image_prefix='../data/balloon/val'
)

# %%
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')


# NOTE: kernel needs to be restarted after changing this
config = './mmdetection/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py'
checkpoint = './weights/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth'
model = init_detector(config, checkpoint, device=device)


visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

#


img = skimage.io.imread('../data/park.jpg')

result = inference_detector(model, img)



visualizer.add_datasample(
  name='result',
  image=img,
  data_sample=result,
  draw_gt=False,
  pred_score_thr=0.3,
  show=False
)

img = visualizer.get_image()

plot_image(img);
"""

# from os import system as sh
# sh('python ./mmdetection/tools/train.py ./confs/maskRCNN-balloon.py')
# sh('python ./mmdetection/tools/test.py  ./confs/maskRCNN-balloon.py work_dirs/maskRCNN-balloon/epoch_12.pth')

# %%

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from mmdet.apis import init_detector, inference_detector

config = './confs/centerNet.py'
weight = None

model = init_detector(config, weight, device=device)


