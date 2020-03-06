from network.nn.seg_modules import FCNHead
from network.nets.SegNet import SegNet


def multi(lr):
    return 10*lr


class FCN(SegNet):
    """
     My Result
     ---------------------------------------------------------------
      Cityscapes : ResNet50_enc  + DA + AL + PSP + OS_8   69.06% (val set)
    """
    def __init__(self, in_ch=3, nclass=21,
                 backbone='resnet101_mit', output_stride=8, pretrained=True,
                 bn='sync_1', multi_grid=False, aux=False, freeze_bn=False):
        super(FCN, self).__init__(in_ch, nclass, backbone, output_stride, pretrained, bn, multi_grid, aux, freeze_bn)
        self.head = FCNHead(2048, nclass, self.norm)

    def get_parameters(self, lr):
        params_list = [{'params': self.backbone.parameters(), 'lr': lr},
                       {'params': self.head.parameters(), 'lr': lr , 'lr_func' : multi}]

        if self.aux:
            params_list.append({'params': self.auxlayer.parameters(), 'lr': lr, 'lr_func': multi})
        return params_list

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone.base_forward(x)
        x = self.head(c4)
        outputs = [x]
        if self.aux:
            aux_out = self.auxlayer(c3)
            outputs.append(aux_out)
        return outputs


class FCN_Vgg(SegNet):
    """
     My Result
     ---------------------------------------------------------------
      Cityscapes : ResNet50_enc  + DA + AL + PSP + OS_8   69.06% (val set)
    """
    def __init__(self, in_ch=3, nclass=21,
                 backbone='resnet101_mit', output_stride=8, pretrained=True,
                 bn='sync_1', multi_grid=False, aux=False, freeze_bn=False):
        super(FCN_Vgg, self).__init__(in_ch, nclass, backbone, output_stride, pretrained, bn, multi_grid, aux, freeze_bn)
        self.head = FCNHead(2048, nclass, self.norm)

    def get_parameters(self, lr):
        params_list = [{'params': self.backbone.parameters(), 'lr': lr},
                       {'params': self.head.parameters(), 'lr': lr , 'lr_func' : multi}]

        if self.aux:
            params_list.append({'params': self.auxlayer.parameters(), 'lr': lr, 'lr_func': multi})
        return params_list

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone.base_forward(x)
        x = self.head(c4)
        outputs = [x]
        if self.aux:
            aux_out = self.auxlayer(c3)
            outputs.append(aux_out)
        return outputs