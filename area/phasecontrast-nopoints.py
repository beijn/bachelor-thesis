# %% [markdown]
# # Area and density based count estimation
# %%

import skimage

import matplotlib.pyplot as plt

import sys

sys.path += [".."]  # NOTE find shared modules
from util.preprocess import *
from util.plot import *
from util.onnx import *

import pandas as pd


def norm(x):
  return (x - x.min()) / (x.max() - x.min())


# %%

dataset = 'third'

data = pd.read_csv(f'../data/{dataset}/data.csv', sep='\s+')
load_image = lambda i: skimage.io.imread(f"../data/{dataset}/{i}.jpg").astype(np.float32) / 255

data.head()

# %%
stens = get_onnx("../data/phaseimaging-combo-v3.onnx")


# %%

shape = skimage.io.imread(f"../data/{dataset}/{data.iloc[0]['img']}.jpg").shape[:2]
fig, axs = mk_fig(2,34, shape=shape, flat=False, scale=0.25)

for I,(i,row) in enumerate(data.iterrows()):
  X = load_image(row['img'])

  input = np.expand_dims(np.transpose(X[..., [2, 1]], (2, 0, 1)), axis=0)  # B and G chans

  pred = stens(input)
  mask = (pred[0, 0] > (thresh := 0.01)).reshape(X.shape[:2])

  area = mask.sum() 
  count = row['count']
  dens = count / area

  data.loc[i, 'area'] = area
  data.loc[i, 'dens'] = dens

  # mark the boundaries

  plot_image(
    skimage.segmentation.mark_boundaries(
      skimage.color.label2rgb(1 - mask, X, bg_color=None, alpha=0.05),
      mask, color=(1, 0, 0), mode='thick'),
    title=f"Mask: {row['img']} ({row['flask']}, {row['mag']}x)",
    ax=axs[I, 0],
  )

  plot_image(
    pred[0, 0], title=f"B-G Phase: {row['img']} ({row['flask']}, {row['mag']}x)", ax=axs[I, 1]
  )


data.head()

# %%

def flask_distribution(flasks=data['flask'].unique()):
  flask2color = dict(zip(flasks, colors))
  cs = list(data['flask'].map(flask2color).values)

  fig, ax = plt.subplots()
  ax.set_title('Cell Density per Image')
  ax.set_xlabel('Cell Count')
  ax.set_ylabel('Cell Area')

  _data = data.query(f'flask in {list(flasks)}')

  _allready_labeled__flasks = set()
  for i in range(len(_data)):
    count_norm = _data.iloc[i]['count'] / _data.iloc[i]['vol']
    ax.plot(count_norm, _data.iloc[i]['area'], 'o', c=cs[i])
    if _data.iloc[i]['flask'] not in _allready_labeled__flasks:
      ax.plot([], [], 'o', c=cs[i], label=f"Flask {_data.iloc[i]['flask']}")
      _allready_labeled__flasks.add(_data.iloc[i]['flask'])

  ax.legend()

  rel_std =  _data['dens'].std() / _data['dens'].mean()

  print(f"Relative standard deviation of the density is {rel_std:.2f}")

flask_distribution()

# %%

relstds = []
for flask in data['flask'].unique():
  dens = data.query(f'flask == "{flask}"')['dens'].mean()
  stdev = data.query(f'flask == "{flask}"')['dens'].std()
  relstd = stdev / dens 
  relstds.append(relstd)
  print(f"Flask {flask} has a mean density of {dens:.2f}±{stdev:.2f} (±{relstd*100:.2f}%)")

print(f"Mean relative standard deviation of the fluctuation is {np.mean(relstds):.2f} (±{np.std(relstds):.2f})")


# %% 

fig, ax = plt.subplots()
data.boxplot(column='dens', by='flask', ax=ax)
ax.set_title('Cell Density Distribution per Flask')
ax.set_xlabel('Flask')
ax.set_ylabel('Cell Density')
fig.suptitle("")

# %%

_data = data.query('flask == "D"')

# now plot only the images from flask D, but catergorical by the magnification
fig, ax = plt.subplots(1,1)

_data.boxplot(column='dens', by='mag', ax=ax)
ax.set_title(f"Cell Density Distribution per Magnification (Flask D)")
ax.set_xlabel('Objective Magnification')
ax.set_ylabel('Cell Density')
fig.suptitle("")

_data.groupby('mag')['dens'].apply(lambda x: (
  relstd := x.std() / x.mean(),
  print(f"Density of flask D, objective {x.name}x: {x.mean():.2f}±{x.std():.2f} (±{relstd:.2f}%)")
));

# %% [markdown]
# ## Observations
# - error for all magnifications is roughly the same
# - lower magnification detects more unproportionally more background 
#   - the reason seems to be that the phase invariances don't hold 