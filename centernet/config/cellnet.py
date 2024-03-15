from dl_lib.network.backbone import Backbone
from dl_lib.layers import ShapeSpec
from dl_lib.network import ResnetBackbone
from dl_lib.network import CenternetDeconv
from dl_lib.network.head import CellNetHead
from dl_lib.network.cellnet import CellNet


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = ResnetBackbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_upsample_layers(cfg, ):
    upsample = CenternetDeconv(cfg)
    return upsample


def build_head(cfg, ):
    head = CellNetHead(cfg)
    return head


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_upsample_layers = build_upsample_layers
    cfg.build_head = build_head
    model = CellNet(cfg)
    return model
