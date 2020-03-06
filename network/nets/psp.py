from network.nets.SegNet import SegNet
from network.nn.seg_modules import PyramidPooling, FCNHead


def multi(lr):
    return 10*lr


class PSPNet(SegNet):
    '''

    Paper results:
    ---------------------------------------------------------------
      Voc_aug    : ResNet101 + DA + AL + PSP   82.6 (testing set)
      Cityscapes : ResNet101 + DA + AL + PSP   78.4 (testing set)
      ADE20k     : ResNet50  + DA + AL + PSP   41.68(val)

    My Results:
    ---------------------------------------------------------------
      Cityscapes : ResNet50  + DA + AL + PSP + OS_8  74.2 (val set)
    '''
    def __init__(self, in_ch=3, nclass=21,
                 backbone='resnet101_mit', output_stride=8, pretrained=True,
                 bn='sync_1', multi_grid=False, aux=False, freeze_bn=False):
        super(PSPNet, self).__init__(in_ch, nclass, backbone, output_stride, pretrained, bn, multi_grid, aux, freeze_bn)

        self.pyramid_pooling = PyramidPooling(2048, self.norm) # for cityscapes batch_size 8
        self.classifier = FCNHead(2048*2, nclass, self.norm)

    def get_parameters(self, lr):
        params_list = [{'params': self.backbone.parameters(), 'lr': lr},
                       {'params': self.pyramid_pooling.parameters(), 'lr': lr, 'lr_func': multi},
                       {'params': self.classifier.parameters(), 'lr': lr, 'lr_func': multi}]
        if self.aux:
            params_list.append({'params': self.auxlayer.parameters(), 'lr': lr, 'lr_func': multi})
        return params_list

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone.base_forward(x)
        out = self.pyramid_pooling(c4)
        outputs = [self.classifier(out)]
        if self.aux:
            aux_out = self.auxlayer(c3)
            outputs.append(aux_out)
        return outputs
