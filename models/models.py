import torch
import math
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn import init
from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock, Bottleneck


class ResNet_FeatureClassifer(nn.Module):

    def __init__(self, num_classes):
        super(ResNet_FeatureClassifer, self).__init__()
        self.feature_classifier = nn.Sequential()
        self.feature_classifier.add_module('fc_n', nn.Linear(256, num_classes))

    def forward(self, feature):
        return self.feature_classifier(feature)


class ResNet_DomainClassifier(nn.Module):

    def __init__(self):
        super(ResNet_DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256, 1024))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(1024, 2))

    def forward(self, feature):
        return self.domain_classifier(feature)


class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], block=Bottleneck):

        super(ResNet_FeatureExtractor, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bottleneck = nn.Sequential()
        self.bottleneck.add_module('avgpool1', nn.AdaptiveAvgPool2d((1, 1)))
        self.linear_layer = nn.Linear(2048, 256)

        zero_init_residual = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, domain='target', lamda=0):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature = self.bottleneck(x)
        feature = feature.view(-1, 2048)
        feature = self.linear_layer(feature)

        return feature

class ResNet(nn.Module):

    def __init__(self, num_classes, device, args):

        super(ResNet, self).__init__()

        self.feature_extractor = ResNet_FeatureExtractor()
        self.feature_classifier = ResNet_FeatureClassifer(num_classes)
        self.domain_classifier = ResNet_DomainClassifier()
        self.method = args.method

    def forward(self, input, domain, lamda):
        feature = self.feature_extractor(input, domain, lamda)

        class_prediction = self.feature_classifier(feature)

        if self.method == "vaada":
            domain_prediction = self.domain_classifier(feature)
        else:
            reverse_feature = ReverseLayer.apply(feature, lamda)
            domain_prediction = self.domain_classifier(reverse_feature)

        return class_prediction, domain_prediction, feature


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None


def load_single_state_dict(net, state_dict):

    own_state = net.state_dict()
    count = 0
    fe_param_count = 0
    for name, param in own_state.items():
        if('feature_extractor' not in name):
            continue
        fe_param_count += 1
        # parsed = name.split('.')
        new_name = name.replace('feature_extractor.', '')
        if new_name in state_dict.keys():
            # print(new_name)
            param_ = state_dict[new_name]
            if isinstance(param_, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param_ = param_.data
            param_data = param_.data
            own_state[name].copy_(param_data)
            count += 1
        else:
            pass
            # print(name)
    print("Imagenet pre-trained weights loaded")
