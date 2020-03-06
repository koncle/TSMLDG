import torch
import torch.nn.functional as F
from torch import nn as nn


class PyramidPooling(nn.Module):
    """ out_chanenls = 2 * in_channels
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=False)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=False)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=False)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=False)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


def _ASPPModule(in_channels, out_channels, kernel, padding, dilation, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride, bn=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        # if output_stride == 16:
        #     dilations = [1, 6, 12, 18]
        # elif output_stride == 8:
        #     dilations = [1, 12, 24, 36]
        # else:
        #     raise NotImplementedError

        self.aspp1 = _ASPPModule(in_channels, 256, 1, padding=0, dilation=dilations[0], norm_layer=bn)
        self.aspp2 = _ASPPModule(in_channels, 256, 3, padding=dilations[1], dilation=dilations[1], norm_layer=bn)
        self.aspp3 = _ASPPModule(in_channels, 256, 3, padding=dilations[2], dilation=dilations[2], norm_layer=bn)
        self.aspp4 = _ASPPModule(in_channels, 256, 3, padding=dilations[3], dilation=dilations[3], norm_layer=bn)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, 256, 1, bias=False),
                                             bn(256),
                                             nn.ReLU(True))
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = bn(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, dropout=0.1):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class ConvNormRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding, norm=nn.BatchNorm2d):
        super(ConvNormRelu, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            norm(in_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)
