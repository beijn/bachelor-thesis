#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: wangfeng19950315@163.com

import numpy as np
import torch


# NOTE the purpose of that 'mask' is to regard only meaningful values in the tensors, because they always have a fixed size 'tensor_dim' regardless of the number of instances


class CellNetGT(object):
    """"Imagine you got only points and classes (here only a single) and no other info."""

    @staticmethod
    def generate(config, batched_input, device):
        upsampling_factor = 1 / config.MODEL.CENTERNET.DOWN_SCALE  # NOTE this is important: the down sampling factor is used in computation of the offset of the center poistion in model output space and the higher resulution input space
        num_classes = config.MODEL.CENTERNET.NUM_CLASSES
        output_size = config.INPUT.OUTPUT_SIZE
        min_overlap = config.MODEL.CENTERNET.MIN_OVERLAP  # This is the minimum overlap between the predicted bounding box and the ground truth bounding box. It is used to determine the radius of the gaussian kernel in such a way 
        tensor_dim = config.MODEL.CENTERNET.TENSOR_DIM  # TODO NOTE why does this have to be fixed, cant we just use the number of instances in the batched_input? 


        scoremaps = []; offsets = []; masks = []; indices = [];
        for data in batched_input:
            centers = data['center']
            classes = data['class']
            radii = data['radius']

            scoremap = torch.zeros(num_classes, *output_size)
            offset = torch.zeros(tensor_dim, 2)
            index = torch.zeros(tensor_dim, dtype=torch.int64) # note this needs to be as big as the number of pixels in a image, also the fmap.gather(...index=index) down the line expects this dtype
            mask = torch.zeros(tensor_dim)

            num_objects = centers.shape[0]  

            centers *= upsampling_factor
            centers_int = centers.to(torch.int32)

            index[:num_objects] = centers_int[..., 1] * output_size[1] + centers_int[..., 0] 
            # the gt_index is used to index the gt_scoremap, so it is a 1D tensor

            offset[:num_objects] = centers - centers_int
            mask[:num_objects] = 1

            # TODO to understand whether I can simplify the gaussian radius like this to a constant I need to visualize the score map of the original model
            CellNetGT.generate_score_map(scoremap, classes, centers_int, radii)

            scoremaps.append(scoremap)
            offsets.append(offset)
            masks.append(mask)
            indices.append(index)

        gt = {
            k: torch.stack(v, dim=0).to(device)
            for k,v in dict(
                scoremap = scoremaps,
                offset = offsets,
                mask = masks,
                index = indices,
            ).items()
        }
        return gt

    @staticmethod
    def generate_score_map(fmap, gt_class, centers_int, gaussian_radii):  # TODO grid search gaussian radius
        radius = torch.clamp_min(gaussian_radii, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CellNetGT.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CellNetGT.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        # return fmap
