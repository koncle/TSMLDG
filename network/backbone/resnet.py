"""Dilated ResNet"""
import math
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet18_mit', 'resnet50_mit', 'resnet101_mit',
           'resnet50_enc', 'resnet101_enc', 'resnet152_enc']


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, mid_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=mid_dilation, dilation=mid_dilation, bias=False)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, mid_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, block_nums, num_classes=1000,
                 deep_base=False, norm_layer=nn.BatchNorm2d,
                 output_stride=8, multi_grid=False, freeze_bn=False):
        super(ResNet, self).__init__()
        # strides and dilations for every layer.
        if output_stride == 32:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            # mark : not the same as the multi-grid in deeplabv3
            # multi grid in the block 4
            dilations = [1, 1, 2, 4] if not multi_grid else [1, 1, 2, [4, 8, 16]]
        else:
            raise NotImplementedError

        self.inplanes = 128 if deep_base else 64

        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_nums[0], stride=strides[0],
                                       dilation=dilations[0], norm_layer=norm_layer)

        self.layer2 = self._make_layer(block, 128, block_nums[1], stride=strides[1],
                                       dilation=dilations[1], norm_layer=norm_layer)

        self.layer3 = self._make_layer(block, 256, block_nums[2], stride=strides[2],
                                       dilation=dilations[2], norm_layer=norm_layer)

        self.layer4 = self._make_layer(block, 512, block_nums[3], stride=strides[3],
                                       dilation=dilations[3], norm_layer=norm_layer)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, block_nums, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if isinstance(dilation, (tuple, list)):
            for i in range(0, block_nums):
                layers.append(block(self.inplanes, planes, dilation=dilation[i], mid_dilation=dilation[i],
                                    downsample=downsample, norm_layer=norm_layer))
                downsample = None
                self.inplanes = planes * block.expansion
        else:
            # first block dilation is halve
            if dilation == 1 or dilation == 2:
                layers.append(block(self.inplanes, planes, stride, dilation=1,
                                    downsample=downsample, mid_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2,
                                    downsample=downsample, mid_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            # rest layers have same dilation
            self.inplanes = planes * block.expansion
            for i in range(1, block_nums):
                layers.append(block(self.inplanes, planes, dilation=dilation, mid_dilation=dilation,
                                    norm_layer=norm_layer))
        self.bn_freezed = False
        return nn.Sequential(*layers)

    def freeze_bn(self):
        self.bn_freezed = True
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d) and self.bn_freezed:
                module.train(False)
            else:
                module.train(mode)
        return self

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1, l2, l3, l4

    def forward(self, x):
        _, _, _, c4 = self.base_forward(x)
        x = self.avgpool(c4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    torch_home = os.path.expanduser('~/.torch')
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    torch_home = os.path.expanduser('~/.torch')
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    torch_home = os.path.expanduser('~/.torch')
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir))
        print('Loaded pretraied model')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    torch_home = os.path.expanduser('~/.torch')
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=model_dir))
        print('Loaded pretraied model')
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    torch_home = os.path.expanduser('~/.torch')
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir=model_dir))
    return model


# --------------------------------------------------------------
# ------------------- MIT pretrained resnet --------------------
# --------------------------------------------------------------


def load_url(url, model_dir=None, map_location=None):
    if model_dir is None:
        torch_home = os.path.expanduser('~/.torch')
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


mit_model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}

deep_base_sequential_dicts = [
    'conv1.0.weight', 'conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var',
    # 'conv1.1.num_batches_tracked',
    'conv1.3.weight', 'conv1.4.weight', 'conv1.4.bias', 'conv1.4.running_mean', 'conv1.4.running_var',
    # 'conv1.4.num_batches_tracked',
    'conv1.6.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var'] # 'bn1.num_batches_tracked']

mit_dicts = [
    'conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
    'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
    'conv3.weight', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var']


def resnet18_mit(pretrained=False, **kwargs):
    kwargs.update({'deep_base': True})
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_url(mit_model_urls['resnet18'])
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in mit_dicts:
                idx = mit_dicts.index(k)
                k = deep_base_sequential_dicts[idx]
            new_state_dict.update({k: v})
        model.load_state_dict(new_state_dict)
        print('Loaded pretraied model')
    return model


def resnet50_mit(pretrained=False, **kwargs):
    kwargs.update({'deep_base': True})
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_url(mit_model_urls['resnet50'])
        # convert to sequential format
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in mit_dicts:
                idx = mit_dicts.index(k)
                k = deep_base_sequential_dicts[idx]
            new_state_dict.update({k: v})
        model.load_state_dict(new_state_dict)
        print('Loaded pretraied model')
    return model


def resnet101_mit(pretrained=False, **kwargs):
    kwargs.update({'deep_base': True})
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = load_url(mit_model_urls['resnet101'])
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in mit_dicts:
                idx = mit_dicts.index(k)
                k = deep_base_sequential_dicts[idx]
            new_state_dict.update({k: v})
        model.load_state_dict(new_state_dict)
        print('Loaded pretraied model')
    return model


# -----------------------------------------------------------------
# ------------------- EncNet pretrained resnet --------------------
# -----------------------------------------------------------------

def load_zip(url, model_dir=None, map_location=None):
    if model_dir is None:
        torch_home = os.path.expanduser('~/.torch')
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = url.split('/')[-1].split('.zip')[0] + '.pth'
    cached_file = os.path.join(model_dir, filename)
    zip_file_path = os.path.join(model_dir, url.split('/')[-1])
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, zip_file_path)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(model_dir)
        os.remove(zip_file_path)
    assert Path(cached_file).exists()
    return torch.load(cached_file, map_location='cpu')


_model_sha1 = {name: checksum for checksum, name in [
    ('25c4b50959ef024fcc050213a06b614899f94b3d', 'resnet50'),
    ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    ('0d43d698c66aceaa2bc0309f55efdd7ff4b143af', 'resnet152'),
    ('da4785cfc837bf00ef95b52fb218feefe703011f', 'wideresnet38'),
    ('b41562160173ee2e979b795c551d3c7143b1e5b5', 'wideresnet50'),
    ('1225f149519c7a0113c43a056153c1bb15468ac0', 'deepten_resnet50_minc'),
    ('662e979de25a389f11c65e9f1df7e06c2c356381', 'fcn_resnet50_ade'),
    ('eeed8e582f0fdccdba8579e7490570adc6d85c7c', 'fcn_resnet50_pcontext'),
    ('54f70c772505064e30efd1ddd3a14e1759faa363', 'psp_resnet50_ade'),
    ('075195c5237b778c718fd73ceddfa1376c18dfd0', 'deeplab_resnet50_ade'),
    ('5ee47ee28b480cc781a195d13b5806d5bbc616bf', 'encnet_resnet101_coco'),
    ('4de91d5922d4d3264f678b663f874da72e82db00', 'encnet_resnet50_pcontext'),
    ('9f27ea13d514d7010e59988341bcbd4140fcc33d', 'encnet_resnet101_pcontext'),
    ('07ac287cd77e53ea583f37454e17d30ce1509a4a', 'encnet_resnet50_ade'),
    ('3f54fa3b67bac7619cd9b3673f5c8227cf8f4718', 'encnet_resnet101_ade'),
]}

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}.zip'
enc_model_urls = {
    'resnet50': _url_format.format('resnet50-'   + _model_sha1['resnet50' ][:8]),
    'resnet101': _url_format.format('resnet101-' + _model_sha1['resnet101'][:8]),
    'resnet152': _url_format.format('resnet152-' + _model_sha1['resnet152'][:8]),
}


def resnet50_enc(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs.update({'deep_base': True})
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_zip(enc_model_urls['resnet50']))
        print('Loaded pretraied model')
    return model


def resnet101_enc(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs.update({'deep_base': True})
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_zip(enc_model_urls['resnet101']))
        print('Loaded pretraied model')
    return model


def resnet152_enc(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs.update({'deep_base': True})
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_zip(enc_model_urls['resnet152']))
        print('Loaded pretraied model')
    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)
    net = resnet50(pretrained=False)
    net(x)
