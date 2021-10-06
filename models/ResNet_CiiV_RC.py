import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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


class PreActResNet(Base_Model):
    def __init__(self, block, num_blocks, num_classes=10, activation='ReLU', softplus_beta=1, 
                    num_sample=3, aug_weight=0.9, mask_center=[5, 16, 27]):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.activation = activation
        self.softplus_beta = softplus_beta

        self.num_sample = num_sample
        self.aug_weight = aug_weight
        self.mask_center = mask_center

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion, track_running_stats=True, affine=True)
        self.linear = nn.Linear(512*block.expansion, num_classes)


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

    def create_mask(self, w, h, center_x, center_y, alpha=10.0):
        widths = torch.arange(w).view(1, -1).repeat(h,1)
        heights = torch.arange(h).view(-1, 1).repeat(1,w)
        mask = ((widths - center_x)**2 + (heights - center_y)**2).float().sqrt()
        # non-linear
        mask = (mask.max() - mask + alpha) ** 0.3
        mask = mask / mask.max()
        # sampling
        mask = (mask + mask.clone().uniform_(0, 1)) > 0.9
        mask.float()
        return mask.unsqueeze(0)

    def ciiv_forward(self, x, loop):
        b, c, w, h = x.shape
        samples = []
        masks = []
        NUM_LOOP = loop
        NUM_INNER_SAMPLE = self.num_sample
        NUM_TOTAL_SAMPLE = NUM_LOOP * NUM_INNER_SAMPLE

        # generate all samples
        for i in range(NUM_TOTAL_SAMPLE):
            # differentiable sampling
            sample = self.relu(x + x.detach().clone().uniform_(-1,1) * self.aug_weight)
            sample = sample / (sample + 1e-5)
            #on_sample = torch.clamp(x + torch.randn_like(x) * 0.1, min=0, max=1)
            if i % NUM_INNER_SAMPLE == 0:
                center_x = random.randint(self.mask_center[0], self.mask_center[-1])
                center_y = random.randint(self.mask_center[0], self.mask_center[-1])
            # attention
            mask = self.create_mask(w, h, center_x, center_y, alpha=10.0).to(x.device)
            sample = sample * mask
            samples.append(sample)
            masks.append(mask)

        # run network
        outputs = []
        features = []
        z_scores = []
        for i in range(NUM_LOOP):
            # Normalized input
            inputs = sum(samples[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]) / NUM_INNER_SAMPLE
            z_score = (sum(masks[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]).float() / NUM_INNER_SAMPLE).mean()
            # forward modules
            out = self.conv1(inputs)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.relu(self.bn(out))
            size = out.shape[-1]
            out = F.avg_pool2d(out, size)
            feats = out.view(out.size(0), -1)
            preds = self.linear(feats)
            z_scores.append(z_score.view(1,1).repeat(b, 1))
            features.append(feats)
            outputs.append(preds)

        final_pred = sum([pred / (z + 1e-9) for pred, z in zip(outputs, z_scores)]) / NUM_LOOP

        ## Randomized Smoothing Inference
        #if self.training or self.is_attack():
        #    final_pred = sum([pred / (z + 1e-9) for pred, z in zip(outputs, z_scores)]) / NUM_LOOP
        #else:
        #    counts = []
        #    for item in outputs:
        #        pred = item.max(-1)[1]
        #        counts.append(F.one_hot(pred, self.num_classes))
        #    final_pred = sum(counts)
        return final_pred, z_scores, features, outputs


    def forward(self, x, loop=1):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # config is passed through main
        x = rgb_norm(x, self.config)
        
        if self.training:
            return self.ciiv_forward(x, loop=loop)
        else:
            return self.ciiv_forward(x, loop=loop)[0]


def PreActResNet18(num_classes=10, activation='ReLU', softplus_beta=1, **kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, activation=activation, softplus_beta=softplus_beta, **kwargs)

def PreActResNet34(num_classes, **kwargs):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes, **kwargs)

def PreActResNet50(num_classes, **kwargs):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes, **kwargs)


def create_model(m_type='resnet18', num_classes=1000, num_sample=3, aug_weight=0.9, mask_center=[5, 16, 27]):
    # create various resnet models
    if m_type == 'resnet18':
        model = PreActResNet18(num_classes=num_classes, num_sample=num_sample,
                                aug_weight=aug_weight, mask_center=mask_center)
    elif m_type == 'resnet50':
        model = PreActResNet50(num_classes=num_classes, num_sample=num_sample,
                                aug_weight=aug_weight, mask_center=mask_center)
    else:
        raise ValueError('Wrong Model Type')
    return model