DATADIR=~/".cache/thesis/data/coco2017"
mkdir -p "$DATADIR"
cd "$DATADIR"

ln -s "$DATADIR" ~/thesis/centernet/CenterNet-better/datasets/coco


for file in train2017.zip val2017.zip test2017.zip; do
  wget -nc http://images.cocodataset.org/zips/$file
  unzip $file
done


for file in annotations_trainval2017.zip; do
  wget -nc http://images.cocodataset.org/annotations/$file
  unzip $file
done