Hey! Here is the last version of the code I used for the baseline model (here are the models for semantic and instance segmentation)
You will have to adjust a few stuff for your data, mainly in the configs/base.py, configs/__init__.py,
You can create the dataset in datasets/datasets. I use this registry singleton format; therefore, dont forget to register your new dataset (you can look up how it is done in the datasets/datasets/rectangles for example) :leichtes_lächeln:
Additionally for semantic segmentation you will want to use a different loss (eg. bce, dice). You will have to add to the models/seg/loss.py  and register it too. Then you will be able to define it in the cfg
Train: sbatch scripts/train.sh
