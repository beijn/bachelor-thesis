# %% [markdown]
# # CenterNet for keypoint detection
# using only vanilla pytorch, no mmdetection...
# %%

import sys; sys.path += ['..']  # NOTE find shared modules
from util.preprocess import *
from util.plot import *

import torch 
def norm(x): return (x - x.min()) / (x.max() - x.min())


# %%

dataset = 'third'
imgid = '1'

SLICE = 512

RGB = norm(skimage.io.imread(f"../data/{dataset}/{imgid}.jpg").astype(np.float32))#[:SLICE, :SLICE]

# green channel turned out to be the sharpest
G = RGB[..., 1]e


data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+')
points = preprocess_point(f'../data/{dataset}/points.json')


pos = preprocess_point("../data/third/points.json")
#pos = np.array([[x,y] for x,y in pos if 0 <= x < SLICE and 0 <= y < SLICE])

ax = plot_image(G, title=f"Input image (green channel): {dataset}/{imgid}")
plot_points(pos, ax, c='r')

data.head()


# %%

from mmdet.apis import init_detector, inference_detector

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = init_detector('./configs/centerNet.py', checkpoint=None, device=device)

model

# %% 
# test the inference on one image

load_image = lambda i: skimage.io.imread(f"../data/{dataset}/{i}.jpg").astype(np.float32) / 255

img = load_image(data['img'][0])

img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

model.eval()
with torch.no_grad():
    out = model(img)

out.shape

# %%
