import torch.nn as nn

from .resnet import *


def build_backbone(backbone, output_stride, bn=nn.BatchNorm2d, multi_grid=False, freeze_bn=False, pretrained=True):
    if backbone[:6] == 'resnet':
        resnet = eval(backbone)
        return resnet(output_stride=output_stride, norm_layer=bn, multi_grid=multi_grid,
                      pretrained=pretrained, freeze_bn=freeze_bn)
    else:
        raise NotImplementedError
