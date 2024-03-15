# Bachelor Thesis Code

## Localizing Cells in Phase-Contrast Microscopy Images using Sparse and Noisy Center-Point Annotations

This is the code of my Bachelor's thesis I am currently writing in the Biomedical Computer Vision Lab at _University of Tartu_ and the Computational Cell Analytics group at _University of Göttingen_ for a human genome study at _Wellcome Sanger Institute_ in Cambridge.
The goal of this work is to localize and count the living cells in low-quality phase-contrast images using only sparse and noisy center-point annotations.

Notebooks with results of the experiments together with their code are under `runs`. **See the most promising approach at [./runs/smp/unet-heatmap/231211-124400-005627.ipynb](https://github.com/beijn/bachelor-thesis/blob/main/runs/smp/unet-heatmap/231211-124400-005627.ipynb).**

A current draft version of the thesis is in [Thesis Draft.pdf](https://github.com/beijn/bachelor-thesis/blob/main/Thesis%20Draft.pdf).

## Introduction

In the field of computer vision the localization of dense homogenous objects is an important task with
many applications. Classical image analysis approaches often suffer from susceptibility to small fluctuations in the data. Machine learning and in particular deep learning algorithms can learn robust
labeling functions, but often require large amounts of high-quality annotations. Producing annotations
of sufficient quality takes a lot of work and may therefore be a major bottleneck.
In this work we focus on predicting cell counts from microscopy images. Estimating the number of
living cells is the central step in many biological workflows. The currently used method, Countess 3
FL by ThermoFisher, we wish to improve upon, requires a trained human to expend several minutes
per sample, preparing it with costly consumables, and applying it to a dedicated expensive device.
The method yields unacceptable inaccuracies. The preparation of the samples makes them unusable
for further experiments. Especially in high-throughput situations where this analysis process has to
be repeated over very many times, these stack up to significant costs, time investments, and material
expenses.
Improving upon this via a passive and automated image analysis can drastically reduce the costs, by
saving a lot of human hours and reducing the required amount of costly consumables. Furthermore,
the less dedicated hardware is needed, the easier a method can be continuously improved with new
data and new software.
Labeling only cell centres minimizes the annotation effort, while providing just enough information
to localize individual cells. The images are obtained via phase-conrast microscopy with relatively low
resolution. It was impossible even for a trained expert to discern cells in the parts of the images with
high confluency. Therefore, while being cheap to obtain, the point labels are sparse and noisy.
We apply several pre-existing zero-shot methods like Cellpose, Stardist and MicroSAM and show that
they fail to provide the desired generalization on our dataset. We fine-tune and retrain well-known
general computer vision models like MaskRCNN and YOLOX on a suitable task automatically derived
from our dataset, and show that these lead to little promising results as well.
We address these issues by developing our own method based on predicting Gaussian heatmaps derived from the point annotations. We show that this method provides easily interpretable results which
are on par with human annotations. However, it falls short to extrapolate into the systematically annotation free regions of high confluence. In our discussion and outlook sections, we consider further approaches to deal with this and other shortcomings. As a baseline, we include a cell count estimate based
on foreground-background segmentation and an assumed constant cell density in the foreground. We
show that this method has severe shortcomings.
Briefly we report on other failed experiments and explorations, such as clustering based approaches,
classical computer-vision filters and Fourier space related transformations.

---

