'''ResNet in PyTorch.
taken from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation, stride=1):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)),
                ('bn0', nn.BatchNorm2d(self.expansion*planes))
            ]))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, activation, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.activation = activation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)),
                ('bn0', nn.BatchNorm2d(self.expansion*planes))
            ]))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_dim=3, out_dim=10, width_factor=1, nonlinearity='relu'):
        super(ResNet, self).__init__()
        self.wf = width_factor
        self.in_planes = round(64 * self.wf)
        
        if nonlinearity == 'leaky_relu':
            self.activation = F.leaky_relu
        elif nonlinearity == 'relu':
            self.activation = F.relu

        self.conv1 = nn.Conv2d(in_dim, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, round(64 * self.wf), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, round(128 * self.wf), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 * self.wf), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 * self.wf), num_blocks[3], stride=2)
        self.linear = nn.Linear(round(512*block.expansion*self.wf), out_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.activation, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, penultimate=False):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if penultimate:
            return out
        out = self.linear(out)
        return out


def ResNet18(in_dim=3, out_dim=10, width_factor=1, nonlinearity='relu'):
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_dim=in_dim, out_dim=out_dim, width_factor=width_factor, nonlinearity=nonlinearity)
    return model

def ResNet50(out_dim=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim)