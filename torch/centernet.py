# %% [markdown]
# # CenterNet for keypoint detection
# using only vanilla pytorch, no mmdetection...
# %%

import sys; sys.path += ['..']  # NOTE find shared modules
from util.preprocess import *
from util.plot import *

import torch 


# %%

dataset = 'third'
data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+')
points = preprocess_point(f'../data/{dataset}/points.json')

load_image = lambda i: skimage.io.imread(f"../data/{dataset}/{i}.jpg").astype(np.float32) / 255

data.head()

# %%

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load('../data/CenterNet2_DLA-BiFPN-P5_640_24x_ST.pth', map_location=device)

model
# %%


# test the inference on one image

img = load_image(data['img'][0])
img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

model.eval()
with torch.no_grad():
    out = model(img)

out.shape
# %%
