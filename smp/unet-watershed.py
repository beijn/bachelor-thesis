# %% [markdown]
# # Train UNet on Watershed Boundaries
# %%

import skimage, scipy

import matplotlib.pyplot as plt

import sys; sys.path += [".."]  # NOTE find shared modules

from util.preprocess import *
from util.plot import *
from util.onnx import *

import segmentation_models_pytorch as smp
import torch

def norm(x): return (x - x.min()) / (x.max() - x.min())

CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu')

print(f"Device: {device}")

# %%

dataset = 'third'
imgid = '1'

SLICE = False if CUDA else 512

RGB = norm(skimage.io.imread(f"../data/{dataset}/{imgid}.jpg").astype(np.float32))
if SLICE: RGB = RGB[:SLICE, :SLICE]

# green channel turned out to be the sharpest
G = RGB[..., 1]


RGB2 = norm(skimage.io.imread(f"../data/{dataset}/2.jpg").astype(np.float32))
if SLICE: RGB2 = RGB2[:SLICE, :SLICE]
G2 = RGB2[..., 1]

pos = preprocess_point("../data/third/points-plus.json").astype(int)
if SLICE: pos = np.array([[x,y] for x,y in pos if 0 <= x < SLICE and 0 <= y < SLICE])

# remove double points in pos
pos = np.unique(pos, axis=0)

ax = plot_image(G, title=f"Input image (green channel): {dataset}/{imgid}")
plot_points(pos, ax, c='r')

# %%

bg_mask_predictor = get_onnx("../data/phaseimaging-combo-v3.onnx")

def bg_mask(thresh):

  input = np.expand_dims(np.transpose(RGB[..., [2, 1]], (2, 0, 1)), axis=0)
  pred = bg_mask_predictor(input)

  mask = (pred[0, 0] > thresh).reshape(G.shape)

  return pred[0,0], mask

thresh = 0.01
fake_phase, bg_mask = bg_mask(thresh=thresh)
plot_image(
  fake_phase, title=f"Predicted fake phase reconstruction from B and G channel"
)
plot_image(
  skimage.color.label2rgb(1 - bg_mask, RGB, bg_color=None, alpha=0.3),
  title=f"Mask (fake phase reconstruction, threshold {thresh:.2f})",
)


# %%

mark = np.zeros(RGB.shape[:2], dtype=int)
mark[pos[:, 0], pos[:, 1]] = np.arange(len(pos)) + 1
# mark[neg[:,0], neg[:,1]] = -1

ws = skimage.segmentation.watershed(bg_mask, mark, mask=bg_mask)
#ws = skimage.morphology.remove_small_holes  (ws, 20)
ws = skimage.morphology.remove_small_objects(ws, 20)

bounds = skimage.segmentation.find_boundaries(ws, mode="thick")

bounds = scipy.ndimage.maximum_filter(bounds, size=6)


plot = skimage.color.label2rgb(ws, G, bg_color=None, alpha=0.4, colors=mk_colors(pos))
plot = skimage.color.label2rgb(bounds, plot, bg_color=None, alpha=1, colors=['black'], saturation=1)
ax = plot_image(plot, title="Watershed Segmentation (postprocessed)")
plot_points(pos, ax, c='k');
del plot

# %%
# turn the ws segmentation into a mask with 1 for background, 2 for any cell instance and 3 for the boundaries
target = np.zeros((*ws.shape, 1))
target[..., 0] = bounds


plot_image(target[...,0], title=f"Target Bounds")


# target = G; print("NOTE: using G channel as target") # NOTE change this back TODO
target = torch.from_numpy(np.transpose(target, (2, 0, 1))).unsqueeze(0).float().to(device)  # NOTE that for multiple channels you have to replace the first unsqueeze(0) with permute(2,0,1)
input = torch.from_numpy(np.transpose(RGB, (2, 0, 1))).unsqueeze(0).float().to(device)
input2 = torch.from_numpy(np.transpose(RGB2, (2, 0, 1))).unsqueeze(0).float().to(device)

# %%

def sample_tile(X, Y, size):
  """Sample a tile of size `size` from the image X and the mask Y"""
  c,h,w = X.shape[-3:]
  x = np.random.randint(0, w - size)
  y = np.random.randint(0, h - size)
  return X[..., y:y+size, x:x+size], Y[..., y:y+size, x:x+size]

def batch_tiles(X, Y, size, n):
  """Sample n tiles of size `size` from the image X and the mask Y"""
  Xs, Ys = zip(*[sample_tile(X, Y, size) for _ in range(n)])
  return torch.stack(Xs), torch.stack(Ys)


# plot a few sampled tiles
fig, axs = mk_fig(2,2, scale=1, shape=(128,128))
for ax in axs:
  X,Y = sample_tile(input, target, 128)

  X,Y = X[0].cpu().numpy(), Y[0].cpu().numpy()
  #y = np.argmax(Y, axis=0)

  heat = norm(np.stack([Y[0],Z:=np.zeros_like(Y[0]),Z,Y[0]], axis=-1))
  
  plot_image(np.transpose(X, (1,2,0)), ax=ax)
  plot_image(heat, ax=ax)

# %% 
model = smp.Unet(
  encoder_name="resnet34",
  encoder_weights=None,
  in_channels=3,
  classes=1,
  activation='sigmoid',
)
model.to(device)

model

# %%

# train the model in a simple pytorch loop
LOG = []

n_epochs = 501 if CUDA else 51
batch_size = 32 if CUDA else 8

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = torch.nn.MSELoss()
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

model.train()

for epoch in range(n_epochs):
  model.train()
  optimizer.zero_grad()

  X,Y = batch_tiles(input[0], target[0], 128, batch_size)

  pred = model(X)
  loss = criterion(pred, Y)

  loss.backward()
  optimizer.step()

  if epoch % 1 == 0:
    with torch.no_grad():
      model.eval()
      pred = model(input)
      pred2 = model(input2)

      LOG.append(dict(
        epoch = epoch,
        loss = loss.item(),
        lr = optimizer.param_groups[0]['lr'],
        pred = pred[0].detach().cpu().numpy(),
        pred2 = pred2[0].detach().cpu().numpy(),
      ))

  
  scheduler.step()#(loss.item())

losses = np.array([x['loss'] for x in LOG])
plt.plot(losses)

# %%

model.eval()
with torch.no_grad():
  pred = model(input).detach().cpu().numpy()[0][0]
  pred2 = model(input2).detach().cpu().numpy()[0][0]

fig, axs = mk_fig(1,5, scale=1, shape=target.shape[-2:])

targ = target[0,0].detach().cpu().numpy()
inp = input[0,1].detach().cpu().numpy()
inp2 = input2[0,1].detach().cpu().numpy()

plot_points = lambda ax: ax.scatter(pos[:,1], pos[:,0], c='magenta', s=8**2, marker='+', label="Target points")

plot_image(targ, ax=axs[0], title=f"Target Bounds")
plot_image(pred, ax=axs[1], title=f"Predicted Bounds")

title = f"Difference between Target and Predicted Bounds"
plot_image(pred-targ, ax=axs[2], title=title, cmap='coolwarm')

title = F"Predicted Bounds Overlayed on Image"
heat = norm(np.stack([pred, Z:=np.zeros_like(pred),Z,pred], axis=-1))
plot_image(inp, ax=axs[3], title=title)
plot_image(heat, ax=axs[3], title=title)

title = F"Predicted Bounds Overlayed on Verification Image"
heat = norm(np.stack([pred2, Z:=np.zeros_like(pred2),Z,pred2], axis=-1))
plot_image(inp2, ax=axs[4], title=title)
plot_image(heat, ax=axs[4], title=title)

[plot_points(ax) for ax in axs[1:4]];

# %%
targ = target[0,0].detach().cpu().numpy().T
inp = input[0,1].detach().cpu().numpy().T
inp2 = input2[0,1].detach().cpu().numpy().T

try: 
  import numpy as np
  from matplotlib.animation import FuncAnimation, FFMpegWriter
  from IPython import display
  import matplotlib.pyplot as plt
  import os 
  try: 
    os.system("module load ffmpeg")
  except: 
    print("could not: module load ffmpeg")
    pass

  scale = 4 if CUDA else 1
  fig, (ax, ax2) = mk_fig(2,1, scale=scale, shape=target.shape[-2:]) 

  FRAMES = len(LOG)

  logify = False

  H = target.shape[-2]

  losses, lrs = [ (
    y := np.array([x[dim] for x in LOG]),
    y := np.log(y) if logify else y,
    m := y.min() if logify else 0,
    H - (y - m) / (y.max() - m) * H
    )[-1]
    
    for dim in ['loss', 'lr']
  ]
 
  def animate(i):
    epoch, loss, lr, pred, pred2 = [LOG[i][k] for k in 'epoch loss lr pred pred2'.split()]
    pred = pred[0].T  # only one channel in this case
    pred2 = pred2[0].T

    H,W = pred.shape

    ax.clear()
    xs = np.linspace(0, W*(i-1)/FRAMES, i)
    ax.plot(xs, losses[:i], label=f"{'log' if logify else ''} loss")
    ax.plot(xs, lrs[:i], label=f"{'log' if logify else ''} lr")

    #plot_image(pred, ax=ax0, title=f"Predicted Bounds")
    title = F"Predicted  Overlayed on Image"
    heat = norm(np.stack([pred, Z:=np.zeros_like(pred),Z,pred], axis=-1))
    plot_image(inp, ax=ax, title=title)
    plot_image(heat, ax=ax, title=title)
    #ax.scatter(pos[:,1], pos[:,0], c='magenta', s=8*scale**2, marker='+', label="Target points")
    ax.legend();
  
    ax2.clear()
    title = F"Predicted Bounds Overlayed on Verification Image"
    heat = norm(np.stack([pred2, Z:=np.zeros_like(pred2),Z,pred2], axis=-1))
    plot_image(inp2, ax=ax2, title=title)
    plot_image(heat, ax=ax2, title=title)


  anim = FuncAnimation(fig, animate, frames=FRAMES, interval=100)
 
  if CUDA:  
    writervideo = FFMpegWriter(fps=30) 
    anim.save('../runs/smp/unet-watershed/training-convergence.mp4', writer=writervideo) 
  else:
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()

except: pass
# %%
