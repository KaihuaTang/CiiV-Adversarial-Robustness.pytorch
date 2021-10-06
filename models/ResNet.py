import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

from models.Base_Model import Base_Model

from utils.train_utils import *


class PreActBlock(Base_Model):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation='ReLU', softplus_beta=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True, affine=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        elif activation == 'GELU':
            self.relu = nn.GELU()
            print('GELU')
        elif activation == 'ELU':
            self.relu = nn.ELU(alpha=1.0, inplace=True)
            print('ELU')
        elif activation == 'LeakyReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            print('LeakyReLU')
        elif activation == 'SELU':
            self.relu = nn.SELU(inplace=True)
            print('SELU')
        elif activation == 'CELU':
            self.relu = nn.CELU(alpha=1.2, inplace=True)
            print('CELU')
        elif activation == 'Tanh':
            self.relu = nn.Tanh()
            print('Tanh')

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(Base_Model):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation='ReLU', softplus_beta=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True, affine=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, track_running_stats=True, affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out



class ClassifierCOS(Base_Model):
    def __init__(self, feat_dim, num_classes=1000, num_head=1, tau=16.0):
        super(ClassifierCOS, self).__init__()
        # classifier weights
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.reset_parameters(self.weight)

        # parameters
        self.scale = tau / num_head   # 16.0 / num_head
        self.num_head = num_head
        self.head_dim = feat_dim // num_head

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normed_x = self.multi_head_call(self.l2_norm, x)
        normed_w = self.multi_head_call(self.l2_norm, self.weight)
        y = torch.mm(normed_x * self.scale, normed_w.t())
        return y

    def multi_head_call(self, func, x):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x


class PreActResNet(Base_Model):
    def __init__(self, block, num_blocks, num_classes=10, cls_type='fc', activation='ReLU', softplus_beta=1):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.activation = activation
        self.softplus_beta = softplus_beta

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion, track_running_stats=True, affine=True)
        if cls_type == 'fc':
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif cls_type == 'cos':
            self.linear = ClassifierCOS(512*block.expansion, num_classes)
        else:
            raise ValueError('Wrong Classifier Type')


        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        elif activation == 'GELU':
            self.relu = nn.GELU()
            print('GELU')
        elif activation == 'ELU':
            self.relu = nn.ELU(alpha=1.0, inplace=True)
            print('ELU')
        elif activation == 'LeakyReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            print('LeakyReLU')
        elif activation == 'SELU':
            self.relu = nn.SELU(inplace=True)
            print('SELU')
        elif activation == 'CELU':
            self.relu = nn.CELU(alpha=1.2, inplace=True)
            print('CELU')
        elif activation == 'Tanh':
            self.relu = nn.Tanh()
            print('Tanh')
        print('Use activation of ' + activation)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                activation=self.activation, softplus_beta=self.softplus_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # config is passed through main
        x = rgb_norm(x, self.config)

        # forward modules
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn(out))
        size = out.shape[-1]
        out = F.avg_pool2d(out, size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(num_classes=10, cls_type='fc', activation='ReLU', softplus_beta=1):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, cls_type=cls_type,
                                                activation=activation, softplus_beta=softplus_beta)

def PreActResNet34(num_classes, cls_type):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes, cls_type)

def PreActResNet50(num_classes, cls_type):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes, cls_type)


def create_model(m_type='resnet18', num_classes=1000, cls_type='fc'):
    # create various resnet models
    if m_type == 'resnet18':
        model = PreActResNet18(num_classes=num_classes, cls_type=cls_type)
    elif m_type == 'resnet50':
        model = PreActResNet50(num_classes=num_classes, cls_type=cls_type)
    else:
        raise ValueError('Wrong Model Type')
    return model