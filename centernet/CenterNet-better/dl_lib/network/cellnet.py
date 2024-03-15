import math

import numpy as np
import torch
import torch.nn as nn

from dl_lib.layers import ShapeSpec
from dl_lib.structures import Boxes, ImageList, Instances

from .generator import CenterNetDecoder
from .generator.cellnet_gt import CellNetGT
from .loss import modified_focal_loss, reg_l1_loss


class CellNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.upsample = cfg.build_upsample_layers(cfg)
        self.head = cfg.build_head(cfg) 

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(self.mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        # TODO: mostly this is wrapping in objects and stuffs that I dont need. Buut there also seems to be some scaling happening... TODO
        #if not self.training:
        #    return self.inference(images)

        features = self.backbone(images.tensor)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        return pred_dict

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255) for img in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    # NOTE: changed: this needs to be called from an outside train loop like this: 
    def losses(self, pred, gt):
        r"""
        TODO update
        
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "score_map": gt scoremap,
                "offset": gt regression of box center point,
                "mask": mask of regression = what part of the tensor actually contains objects,
                "index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
            "score_map": predicted score map,
            "offset": predcited regression
            }
        """

        loss_scoremap = modified_focal_loss(pred['scoremap'], gt['scoremap'])

        loss_offset = reg_l1_loss(pred['offset'], gt['mask'], gt['index'], gt['offset'])

        loss_scoremap *= self.cfg.MODEL.LOSS.WEIGHT_SCOREMAP
        loss_offset *= self.cfg.MODEL.LOSS.WEIGHT_OFFSET

        loss = {
            "scoremap": loss_scoremap,
            "offset": loss_offset,
        }
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        return CellNetGT.generate(self.cfg, batched_inputs, self.device)

    @torch.no_grad()
    def inference(self, images):
        """
        image(tensor): ImageList in dl_lib.structures
        """
        n, c, h, w = images.tensor.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.cfg.MODEL.CENTERNET.DOWN_SCALE
        img_info = dict(center=center_wh, size=size_wh,
                        height=new_h // down_scale,
                        width=new_w // down_scale)

        pad_value = [-x / y for x, y in zip(self.mean, self.std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(images.tensor.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h:h + pad_h, pad_w:w + pad_w] = images.tensor

        features = self.backbone(aligned_img)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)
        results = self.decode_prediction(pred_dict, img_info)

        ori_w, ori_h = img_info['center'] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **results)

        return [{"instances": det_instance}]

    def decode_prediction(self, pred_dict, img_info):
        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        """
        fmap = pred_dict["cls"]
        reg = pred_dict["reg"]
        wh = pred_dict["wh"]

        boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)
        # boxes = Boxes(boxes.reshape(boxes.shape[-2:]))
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)

        # dets = CenterNetDecoder.decode(fmap, wh, reg)
        boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
        boxes = Boxes(boxes)
        return dict(pred_boxes=boxes, scores=scores, pred_classes=classes)



