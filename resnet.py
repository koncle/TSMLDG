from torch import nn
from network.nets.SegNet import SegNet


class Net(SegNet):
    def __init__(self, in_ch=3, nclass=19, backbone='resnet50_enc', output_stride=8, pretrained=True, bn='torch'):
        super(Net, self).__init__(in_ch, nclass, backbone, output_stride, pretrained, bn, multi_grid=False, aux=False)
        self.x = nn.Sequential(
            nn.Conv2d(2048, 2048 // 4, 3, padding=1, bias=False),
            self.norm(2048 // 4),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
        )
        self.seg_classifier = nn.Conv2d(2048 // 4, nclass, 1)

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone.base_forward(x)
        feats = self.x(c4)
        seg_logits = self.seg_classifier(feats)
        return seg_logits, c1, c2, c3, c4, feats

    def not_track(self, module=None):
        if module is None:
            module = self
        if len(module._modules) != 0:
            for (k, v) in module._modules.items():
                self.not_track(v)
        else:
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

    def remove_dropout(self):
        self.x[-1].p = 1e-10

    def recover_dropout(self):
        self.x[-1].p = 0.1