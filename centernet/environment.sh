
conda=micromamba

$conda create -y -c pytorch -c conda-forge -c nvidia -n centernet \
  python pip cxx-compiler cuda pytorch pytorch-cuda torchvision \
  'Pillow==9.5.0' colorama opencv \
  jupytext ipykernel nbconvert \
  scikit-image colorcet pandas  
$conda activate centernet

git clone git@github.com:FateScript/CenterNet-better.git
cd CenterNet-better
pip install -v -e .  # -e = editable mode == changes in code reflect in imports
cd ..

echo "Centernet environment created. The lib can be importeed via dl_lib"