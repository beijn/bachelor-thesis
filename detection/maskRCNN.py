# %% [markdown]
# I need to use tiling of the images and masks because independent binary masks explode RAM

# %%
import sys, os; sys.path += ['..']  # NOTE important to find shared modules

import matplotlib.pyplot as plt
import numpy as np
import json
from skimage.color import label2rgb

import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import scipy.ndimage
from points2masks_tiled import *

R = 25
setup_cache('third', clear=False)
_,TILES = make_tiles_dummyBox(R=R, every_nth=1)

class ThirdDataset(torch.utils.data.Dataset):
  def __init__(self, transforms):
    self.tiles = TILES
    self.transforms = transforms

  def __len__(self): return len(self.tiles)
        
  def __getitem__(self, I):
    img, masks, pts = load_tile(self.tiles[I])
    masks = torch.from_numpy(masks)

    num_objs = len(masks)

    # get bounding box coordinates for each mask
    boxes = masks_to_boxes(masks)

    # there is only one class
    labels = torch.ones((num_objs,), dtype=torch.int64)

    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    # Wrap sample and targets into torchvision tv_tensors:
    img = tv_tensors.Image(img).to(dtype=torch.float) / img.max()
    print(img.shape, img.dtype, img.min(), img.max())

    target = dict(
      boxes=tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[-2:]),
      masks=tv_tensors.Mask(masks),
      labels=labels,
      image_id=I,
      area=area,
      iscrowd=iscrowd,
    )

    if self.transforms is not None:
      img, target = self.transforms(img, target)

    return img, target
  
plot_tiles(
    imgid='1',
    scale=6,
    image=True,
    gt_mask=True,
    points=True,
    suptitle=f'Tiled dummy masks from Stuart\'s points. Note that overlapping masks are saved independently.',
  );


# %%

######################################################################
# Object detection and instance segmentation model for PennFudan Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In our case, we want to finetune from a pre-trained model, given that
# our dataset is very small, so we will be following approach number 1.
#
# Here we want to also compute the instance segmentation masks, so we will
# be using Mask R-CNN:


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights 


WEIGHTS = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1  # MaskRCNN_ResNet50_FPN_Weights.DEFAULT  


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO  # TODO _v2(..)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=WEIGHTS, box_detections_per_img=300)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

False and os.system("""
mkdir -p references;
cd referencesf;
wget -nc https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py;
wget -nc https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py;
wget -nc https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py;
wget -nc https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py;
wget -nc https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py;
""");

from torchvision.transforms import v2 as T


def get_transform(train):
    #transforms = [WEIGHTS.transforms()]
    # append to tensor transform
    transforms = [T.ToTensor()]
    if train: transforms += [
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        # TODO add others
    ]
    return T.Compose(transforms)



from references.engine import train_one_epoch, evaluate
import references.utils as utils

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = ThirdDataset(get_transform(train=True))
dataset_test = ThirdDataset(get_transform(train=False))

# split the dataset in train and test set
# NOTE TODO: reintroduce splits
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# %%
# let's train it for 5 epochs
num_epochs = 1  # NOTE!!

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()
    
    # NOTE this raises "ValueError: Buffer dtype mismatch, expected 'uint8_t' but got 'float'" somewhere deep down
    #evaluate(model, data_loader_test, device=device)

print("That's it!")

# %%

import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import colorcet 
import itertools as it

colors_ = colorcet.m_glasbey.colors
colors = [tuple(map(lambda x: int(x*255), c)) for c in colors_]

S = 6

eval_transform = get_transform(train=False)
model.eval()

count_GT, count_pred = 0,0
saves = []

def doit(ax, image, seg, pts):
    global count_GT, count_pred, saves
    image = image.astype(np.float32) / 255.0
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"cell: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    
    cs = list(it.islice(it.cycle(colors), len(pred_boxes)))

    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors=cs, width=S)

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors=cs)
    ax.imshow(output_image.permute(1, 2, 0))

    pred_centers = [[(x+X)/2, (y+Y)/2] for x,y,X,Y in pred_boxes]
    ax.scatter(*zip(*pred_centers), s=S*4, c=[[float(x)/255 for x in c] for c in cs], marker='x')

    ax.legend([f'GT ({len(pts)})', f'pred ({len(pred_boxes)})'], loc='upper left')

    count_pred += len(pred_boxes)
    count_GT += len(pts)

    saves += [dict(
        image=image,
        pred=pred,
        gt_pts=pts
    )]

fig, axs = plot_tiles(
    imgid = '1',
    image = False,
    gt_mask = False,
    points = True,
    posts = [doit],
    scale = S,
);

suptitle = f"maskRCNN ({num_epochs} epochs): detected {count_pred()/count_GT()*100:.2f}% cells."
fig.suptitle(suptitle)


# %%

I = 0
def doit(ax, img, seg, pts):
    global saves, I
    save = saves[I]
    I += 1

    image = save['image']
    pred = save['pred']
    pred_labels = [f"{score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    
    cs = list(it.islice(it.cycle(colors), len(pred_boxes)))

    pred_centers = [[(x+X)/2, (y+Y)/2] for x,y,X,Y in pred_boxes]
    ax.scatter(*zip(*pred_centers), s=S*8, c=[[float(x)/255 for x in c] for c in cs], marker='x')

    ax.legend([ f'pred ({len(pred_boxes)}/{len(pts)})'], loc='upper left')

plot_tiles(
    imgid = '1',
    image = True,
    gt_mask = False,
    points = True,
    posts = [doit],
    scale = S,
    suptitle = f"maskRCNN ({num_epochs} epochs): detected {count_pred/count_GT*100:.2f}% cells."
);


# %%

# TODO: detections at borders and merge tiles