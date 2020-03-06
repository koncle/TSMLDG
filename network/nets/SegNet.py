import torch
import torch.nn as nn

from network.backbone import build_backbone
from network.nn.seg_modules import FCNHead


class SegNet(nn.Module):
    def __init__(self, in_ch=3, nclass=21,
                 backbone='resnet101_mit', output_stride=8, pretrained=True,
                 bn='torch', multi_grid=False, aux=False):
        super(SegNet, self).__init__()

        if bn == 'torch':
            print('Using Torch.BatchNorm')
            self.norm = nn.BatchNorm2d
        else:
            raise NotImplementedError()

        self.backbone = build_backbone(backbone, output_stride, self.norm, multi_grid=multi_grid, pretrained=pretrained)

        self.aux = aux
        if self.aux:
            self.auxlayer = FCNHead(1024, nclass, self.norm)

    def base_forward(self, x):
        return self.backbone.base_forward(x)
