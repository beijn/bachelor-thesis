#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .centernet_head import SingleHead


class CellNetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(CellNetHead, self).__init__()
        self.head4scoremap = SingleHead(64, cfg.MODEL.CENTERNET.NUM_CLASSES)
        self.head4offset = SingleHead(64, 2)  # maybe later parametrize the 64

    def forward(self, x):
        return dict(
            scoremap = torch.sigmoid(self.head4scoremap(x)),
            offset = self.head4offset(x),
        )


"""
class CellnetHead(nn.Module):
  def __init__(self, cfg):
    super(CellnetHead, self).__init__()
    self.objectness = SingleHead(64,1)  # predict object existence confidence
    self.confidence_position = SingleHead(64,1)  # predict accuracy in position
    self.center = SingleHead(64,2)  # predict the precise center position (relative to the downsampled feature map?)
    # TODO: experiment with more heads, eg for morphology features
"""