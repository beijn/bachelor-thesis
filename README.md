# Bachelor's Thesis in Computer Science
## Localizing Cells in Phase-Contrast Microscopy Images using Sparse and Noisy Center-Point Annotations
Submitted by Benjamin Eckhardt on April 1, 2024. Reviewed by Dr. Dmytro Fishman and Dr. Constantin Pape.

This is the code of my [Bachelor's Thesis](<./Bachelor's Thesis.pdf>) conducted with the Biomedical Computer Vision Lab at _University of Tartu_ and the Computational Cell Analytics group at _University of Göttingen_ for a human genome study at _Wellcome Sanger Institute_ in Cambridge.
The goal of this work is to localize and count the living cells in low-quality phase-contrast images using only sparse and noisy center-point annotations.

Notebooks with results of the experiments together with their code are under [./runs](./runs). 

The data is in the [./data](./data) folder.


## Introduction

![Visual representation of the work performed in this thesis. The phase-contrast microscopy image were acquired by our research partners at Sanger (left panel). We implemented, applied and evaluated several deep learning approaches for detecting cells present on those images (middle panel), such as zero-Shot instance segmentation (middle top), object detection (middle center), and density map regression (middle  bottom). Those approaches were trained using sparse and at times noisy point annotations (right panel).](./graphical-abstract.png)

Computer vision and in particular object detection using deep learning has been the key for automated high-throughput data analysis in many applications. This thesis applies object detection in, at a first glance, very unusual area - genome research. Specifically, our collaborators at the Wellcome Sanger Institute in Cambridge aim to map vital genes in humans. To this end, they continuously introduce targeted mutations in the genome of a human cell culture and count the surviving cells' proportion. The great number of counts that has to be performed demands for an automated high-throughput workflow with low recurring costs. Our collaborators previously relied on using the Countess 3 FL cell counting device by ThermoFisher in an expensive and time-consuming semi-manual process with unacceptable inaccuracies.
By developing multiple approaches to count cells using only microscopy images, we minimize material and time expenses and contribute to a completely automated and resource efficient high-throughput workflow for the mapping of vital genes.

Classical image analysis approaches often suffer from susceptibility to small fluctuations in data. Machine learning and in particular deep learning algorithms can learn robust labeling functions, but often require large amounts of high-quality annotations. Producing annotations of sufficient quality takes a lot of work and may therefore be a major bottleneck. Therefore, in this thesis we focus on predicting cell counts in microscopy images using only point annotations at cell centers, which are very cheap to obtain with a single click per instance. 

The images were obtained via the physical phase-contrast microscope EVOS (ThermoFisher) with digital controls enabling precisely reproducible imaging settings. However, the images are of such low resolution, that even a trained expert could not discern cells in some parts of the images with high confluency. Therefore, while being cheap to obtain, the point labels are sparse and noisy. We partly address the noisiness, but leave the sparsity for future work.

We apply several pre-existing zero-shot methods like Cellpose, Stardist and MicroSAM and show that they fail to provide the desired generalization on our dataset. We fine tune the well-known object detection model Mask R-CNN on a synthetic task derived from the point labels, and show that these lead to little promising results as well.  

All of these approaches having failed, we develop our own method based on predicting density maps derived as a superposition of Gaussian distributions around the point annotations. We show that this method provides easily interpretable results, which are on par with human annotations. However, it falls short to extrapolate into the systematically annotation free regions of high confluence.

Briefly we report on other failed experiments and explorations, such as clustering based approaches, classical computer-vision filters and Fourier space related transformations with the aim to pronounce cell structures in the images. As a baseline, we develop a cell count estimate based on foreground-background segmentation and an assumed constant cell density in the foreground. We show that this method has severe shortcomings.