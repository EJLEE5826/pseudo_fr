import torch
import pickle
from torch.autograd import Variable
import torch.nn.functional as F
#from __init__ import *

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, block, layers, num_classes=8631, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)


class Model(nn.Module):
    
    def __init__(self, args, pretrained=True, num_classes=7):
        super(Model, self).__init__()
        
        # self.mode = mode
        self.num_classes = num_classes
        
        # if self.mode == 'fix':
        #     self.norm_fn = FixedBatchNorm
        # else:
        #     self.norm_fn = nn.BatchNorm2d
        
        self.resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        resnet50_path = '/home/jihyun/code/Erasing-Attention-Consistency/model/resnet50_ft_weight.pkl'
        if pretrained:
            with open(resnet50_path, 'rb') as f:
                obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        
            # weights = torch.load(resnet50_path)
        
            weights.pop('fc.weight')
            weights.pop('fc.bias')
        
        self.resnet50.load_state_dict(weights, strict=False)

        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes
        
        self.stage1 = nn.Sequential(self.resnet50.conv1, 
                                    self.resnet50.bn1, 
                                    self.resnet50.relu, 
                                    self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, 7)
                    

    def forward(self, x):
        x = self.stage1(x) 
        x = self.stage2(x) 
        x = self.stage3(x) 
        x = self.stage4(x) 
        x = self.stage5(x)
        feature = self.avgpool(x)
        # x = self.avgpool(x)
        feature = feature.view(feature.size(0), -1) 
        output = self.fc(feature)
        
        params = list(self.fc.parameters())
        
        fc_weight = params[-2].data
        fc_weight = fc_weight.contiguous().view(1, 7, 2048, 1, 1) 
        fc_weight = Variable(fc_weight, requires_grad=False)
        
        feat = x.unsqueeze(1) 
        
        hm = feat * fc_weight 
        hm = hm.sum(2) 

        return output, hm