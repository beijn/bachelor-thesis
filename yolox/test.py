# %% [markdown]
# # somthing YOLOX

# %%
from PIL import Image
import torch, torchvision
from torchvision import transforms

import sys, os; sys.path += ['..']  # NOTE find shared modules
from util.cache import *

cache_dir = mk_cache('yolox/test', clear=True)

model_type = 'yolox_nano'

#torch.hub.list('Megvii-BaseDetection/YOLOX')
model = torch.hub.load('Megvii-BaseDetection/YOLOX', model_type, pretrained=True)
model.eval()

# %%

pImage = '../data/third/1.jpg'
image = Image.open(pImage)

# Define the transformations to be applied to the image
transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

# Apply the transformations to the image
# Add a batch dimension to the tensor
input = transform(image).unsqueeze(0)

# Move the tensor to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input = input.to(device)


# %%

output = model(input).cpu()
# %%

torchvision.models.detection.__dict__
# %%
