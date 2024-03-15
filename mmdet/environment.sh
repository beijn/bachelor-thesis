
conda=micromamba

$conda create -y -c pytorch -c conda-forge -n mmdet \
  python pip \
  pytorch torchvision libgcc \
  jupytext ipykernel nbconvert \
  scikit-image colorcet
$conda activate mmdet

pip install -U openmim
mim install mmengine mmcv

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .  # -e = editable mode == changes in code reflect in imports
cd ..