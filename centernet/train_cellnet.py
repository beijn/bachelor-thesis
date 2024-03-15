# %% [markdown]
# # CenterNet based point heatmap
# training code copied from centernet/CenterNet-better/tools/plain_train_net.py
# Bur removed: distributed training, checkpointing, writing / events, evaluation
# %%

import sys

sys.path += [".."]  # NOTE find shared modules

from util.preprocess import *
from util.plot import *

import torch


def norm(x):
  return (x - x.min()) / (x.max() - x.min())


CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")

print(f"Device: {device}")


# %%

dataset = "third"
imgid = "1"

SLICE = False if CUDA else 512
RGB = norm(skimage.io.imread(f"../data/{dataset}/{imgid}.jpg").astype(np.float32))
if SLICE:
  RGB = RGB[:SLICE, :SLICE]

# green channel turned out to be the sharpest
G = RGB[..., 1]

pos = preprocess_point("../data/third/points.json")
if SLICE:
  pos = np.array([[x, y] for x, y in pos if 0 <= x < SLICE and 0 <= y < SLICE])

# remove double points in pos
pos = np.unique(pos, axis=0)

ax = plot_image(G, title=f"Input image (green channel): {dataset}/{imgid}")
plot_points(pos, ax, c="r")

# %%

from config.cellnet import build_model
from config.centernet_resnet18_coco512 import config

model = build_model(config)

model
# %%

import torch


## implement a data loader
class CellDataset(torch.utils.data.Dataset):
  def __init__(self, img, pos):
    self.img = torch.from_numpy(img).permute(2, 0, 1)[...,:512,:512] # NOTE TODO: GT scoremap genaration assumes a fixed size, given by input size and overall model downsampling
    self.pos = torch.from_numpy(pos)
    # filter out all pos not in the sliced image
    self.pos = self.pos[   (0 <= self.pos[:, 0]) 
                         & (self.pos[:, 0] < 512) 
                         & (0 <= self.pos[:, 1]) 
                         & (self.pos[:, 1] < 512)]

  def __len__(self):
    return config.SOLVER.MAX_ITER

  def __getitem__(self, idx):
    return {
      'image': self.img,
      'center': self.pos,
      'class': 0 * torch.ones(len(self.pos), dtype=torch.long),  # TODO optimize - smaller is nicer, but highest prio is to prevent auto casting
      'radius': 2 * torch.ones(len(self.pos)),
    }



data_loader = torch.utils.data.DataLoader(
    CellDataset(RGB, pos),
    batch_size=1,
    shuffle=True,
    num_workers=config.DATALOADER.NUM_WORKERS,
    collate_fn=lambda x: x,
)


data = next(iter(data_loader))
gt = model.get_ground_truth(data)

plot_image(gt['scoremap'][0, 0].detach().cpu().numpy(), title="GT scoremap")
# %% 

from dl_lib.solver import build_lr_scheduler, build_optimizer
cfg = config

model = build_model(cfg)
optimizer = build_optimizer(cfg.SOLVER.OPTIMIZER, model)
scheduler = build_lr_scheduler(cfg.SOLVER.LR_SCHEDULER, optimizer)
criterion = torch.nn.MSELoss()

# %%
model.train()
start_iter = 0
max_iter = 1000


snaps = []

for data, iteration in zip(data_loader, range(start_iter, max_iter)):
  iteration = iteration + 1

  pred = model(data)
  gt = model.get_ground_truth(data)

  #loss = model.losses(pred, gt)['scoremap']
  loss = criterion(pred['scoremap'], gt['scoremap'])
  #print('Loss', loss.item())
  assert torch.isfinite(loss).all(), ""

  snaps.append(dict(
    pred = pred['scoremap'][0],
    loss = loss.item(),
    lr = optimizer.param_groups[0]['lr'],
    epoch = iteration
  ))

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  scheduler.step()

losses = np.array([x['loss'] for x in snaps])
plt.plot(losses)
# %%

data = next(iter(data_loader))
pred = model(data)
gt = model.get_ground_truth(data)

pred.keys(), gt.keys(), pred['scoremap'].shape

plot_image(gt['scoremap'][0, 0].detach().cpu().numpy(), title="Target scoremap")
plot_image(pred['scoremap'][0, 0].detach().cpu().numpy(), title="Predicted scoremap")
# %%


from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np

fig, ax = mk_fig(1,1)

FRAMES = len(snaps)

pred = snaps[0]['pred']

H = pred.shape[-2]

export=False
logify_scalars=False

losses, lrs = [ (
  y := np.array([x[dim] for x in snaps]),
  y := np.log(y) if logify_scalars else y,
  m := y.min() if logify_scalars else 0,
  H - (y - m) / (y.max() - m) * H
  )[-1]
  
  for dim in ['loss', 'lr']
]

def animate(i):
  epoch, loss, lr, pred = [snaps[i][k] for k in 'epoch loss lr pred'.split()]
  pred = pred[0].cpu()  # only one channel in this case
  print(pred.shape)

  H,W = pred.shape

  ax.clear()
  xs = np.linspace(0, W*(i-1)/FRAMES, i)
  ax.plot(xs, losses[:i], label=f"{'log' if logify_scalars else ''} loss")
  ax.plot(xs, lrs[:i], label=f"{'log' if logify_scalars else ''} lr")

  title = F"Predicted Heatmap Overlayed on Image"
  heat = norm(np.stack([pred, Z:=np.zeros_like(pred),Z,pred], axis=-1))
  #plot_image(inp, ax=ax, title=title)
  #plot_image(heat, ax=ax, title=title)
  plot_image(pred, ax=ax, title=f"Predicted Heatmap")
  ax.legend();

anim = FuncAnimation(fig, animate, frames=FRAMES, interval=100)

if export:  
  import os
  os.system("module load ffmpeg || echo could not: module load ffmpeg")
  writervideo = FFMpegWriter(fps=30) 
  anim.save('../runs/centernet/test/training-convergence.mp4', writer=writervideo) 
else:
  from IPython import display
  video = anim.to_html5_video()
  html = display.HTML(video)
  display.display(html)
  plt.close()

# %%
