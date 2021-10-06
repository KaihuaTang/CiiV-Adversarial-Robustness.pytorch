import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Base_Model import Base_Model
from utils.train_utils import *

class BasicBlock(Base_Model):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activation='ReLU', softplus_beta=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if activation == 'ReLU':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            print('R')
        elif activation == 'Softplus':
            self.relu1 = nn.Softplus(beta=softplus_beta, threshold=20)
            self.relu2 = nn.Softplus(beta=softplus_beta, threshold=20)
            print('S')
        elif activation == 'GELU':
            self.relu1 = nn.GELU()
            self.relu2 = nn.GELU()
            print('G')
        elif activation == 'ELU':
            self.relu1 = nn.ELU(alpha=1.0, inplace=True)
            self.relu2 = nn.ELU(alpha=1.0, inplace=True)
            print('E')

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(Base_Model):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activation='ReLU', softplus_beta=1):
        super(NetworkBlock, self).__init__()
        self.activation = activation
        self.softplus_beta = softplus_beta
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                self.activation, self.softplus_beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(Base_Model):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, activation='ReLU', softplus_beta=1, num_sample=3, aug_weight=0.9, mask_center=[5, 16, 27]):
        super(WideResNet, self).__init__()
        self.num_sample = num_sample
        self.aug_weight = aug_weight
        self.mask_center = mask_center

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        #self.scale = scale
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
        elif activation == 'GELU':
            self.relu = nn.GELU()
        elif activation == 'ELU':
            self.relu = nn.ELU(alpha=1.0, inplace=True)
        print('Use activation of ' + activation)

        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

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
                idx = int(i // NUM_INNER_SAMPLE)
                x_idx = int(idx // 3)
                y_idx = int(idx % 3)
                center_x = self.mask_center[x_idx]
                center_y = self.mask_center[y_idx]
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
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            size = out.shape[-1]
            out = F.avg_pool2d(out, size)
            feats = out.view(out.size(0), -1)
            preds = self.fc(feats)

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


def create_model(m_type='WRN34-10', num_classes=1000, num_sample=3, aug_weight=0.9, mask_center=[5, 16, 27]):
    if m_type == 'WRN34-10':
        model = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0, num_sample=num_sample,
                                aug_weight=aug_weight, mask_center=mask_center)
    else:
        raise ValueError('Wrong Model Type')
    return model