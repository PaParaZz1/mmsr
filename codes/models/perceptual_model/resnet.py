import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from functools import partial
from models.grad_op_conv import BlurGradConv, DownUpGradConv, InterpGradConv
from .perceptual_model import IPerceptualModel

BN = None
CONV = None

__all__ = ['resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return CONV(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_skip=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.stride = stride
        self.use_skip = use_skip
        if use_skip:
            self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_skip:
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_skip=True, avg_down=False):
        super(Bottleneck, self).__init__()
        self.conv1 = CONV(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        if avg_down and stride > 1:
            self.conv2 = nn.Sequential(
                        nn.AvgPool2d(stride, stride),
                        CONV(planes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
                    )
        else:
            self.conv2 = CONV(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = CONV(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.use_skip = use_skip
        if use_skip:
            self.downsample = downsample

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

        if self.use_skip:
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module, IPerceptualModel):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 avg_down=False,
                 grad_op=None,  # blur down_up
                 blur_grad_kernel_size=3,
                 blur_range='only_conv_s2',  # all
                 gauss_sigma=0,
                 use_skip_s2=True,
                 downsample_kernel_size=1):

        global BN, CONV, bypass_bn_weight_list

        def CONVFunc(*args, **kwargs):
            if grad_op == 'down_up':
                return DownUpGradConv(nn.Conv2d(*args, **kwargs))
            elif grad_op == 'blur':
                if blur_range == 'all':
                    return BlurGradConv(blur_grad_kernel_size, nn.Conv2d(*args, **kwargs), gauss_sigma=gauss_sigma)
                elif blur_range == 'only_conv_s2':
                    if 'stride' in kwargs.keys():
                        stride = kwargs['stride']
                    else:
                        if len(args) >= 4:
                            stride = args[3]
                        else:
                            stride = 1

                    if stride == 1:
                        return nn.Conv2d(*args, **kwargs)
                    else:
                        return BlurGradConv(blur_grad_kernel_size, nn.Conv2d(*args, **kwargs), gauss_sigma=gauss_sigma)
                else:
                    raise ValueError
            elif grad_op == 'interp':
                if blur_range == 'all':
                    return InterpGradConv(*args, **kwargs)
                elif blur_range == 'only_conv_s2':
                    if 'stride' in kwargs.keys():
                        stride = kwargs['stride']
                    else:
                        if len(args) >= 4:
                            stride = args[3]
                        else:
                            stride = 1

                    if stride == 1:
                        return nn.Conv2d(*args, **kwargs)
                    else:
                        return InterpGradConv(*args, **kwargs)
                else:
                    raise ValueError
            else:
                raise ValueError

        BN = nn.BatchNorm2d
        if grad_op is None:
            CONV = nn.Conv2d
        else:
            CONV = CONVFunc

        bypass_bn_weight_list = []

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.down_k = downsample_kernel_size

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        CONV(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        CONV(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        CONV(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            if self.avg_down:
                self.conv1 = nn.Sequential(
                    CONV(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.AvgPool2d(2, 2),
                    CONV(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
                )
            else:
                self.conv1 = CONV(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool0 = nn.AvgPool2d(2, 2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_skip_s2=use_skip_s2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_skip_s2=use_skip_s2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_skip_s2=use_skip_s2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif (isinstance(m, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_skip_s2=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    CONV(self.inplanes, planes * block.expansion,
                              kernel_size=self.down_k, stride=1, padding=self.down_k // 2, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    CONV(self.inplanes, planes * block.expansion,
                              kernel_size=self.down_k, stride=stride, padding=self.down_k // 2, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_skip_s2, avg_down=self.avg_down))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.avg_down:
            x = self.avgpool0(x)
        else:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # override
    def forward_feature(self, x, feature_name=['layer4']):
        features = []
        handles = []

        def hook(module, act_in, act_out):
            features.append(act_out)

        for name in feature_name:
            handles.append(getattr(self, name).register_forward_hook(hook))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.avg_down:
            x = self.avgpool0(x)
        else:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        for t in handles:
            t.remove()

        return features


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet50c(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, deep_stem=True)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, deep_stem=True, avg_down=True)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
