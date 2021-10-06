import torch
import torch.nn as nn
import torchvision.models as models

import random
from models.Base_Model import Base_Model

class VGG(Base_Model):
    def __init__(self, features, num_classes=1000, init_weights=True, num_sample=3, 
                    mask_aug=False, samp_aug=False, aug_weight=1.0, mask_center=[5, 16, 27]):
        super(VGG, self).__init__()

        self.num_sample = num_sample
        self.mask_aug = mask_aug
        self.samp_aug = samp_aug
        self.aug_weight = aug_weight
        self.mask_center = mask_center

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, loop=1, bpda=False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b, c, w, h = x.shape
        on_samples = []
        off_samples = []
        NUM_LOOP = loop
        NUM_INNER_SAMPLE = self.num_sample
        NUM_TOTAL_SAMPLE = NUM_LOOP * NUM_INNER_SAMPLE
        
        # generate all samples
        for i in range(NUM_TOTAL_SAMPLE):
            if self.mask_aug:
                if i % NUM_INNER_SAMPLE == 0:
                    center_x = self.mask_center[random.randint(0,2)]
                    center_y = self.mask_center[random.randint(0,2)]
                # attention
                mask = self.create_mask(w, h, center_x, center_y, alpha=10.0).to(x.device)

            # sample weight augmentation
            if self.samp_aug:
                weight = 0.75 + float(i) / NUM_TOTAL_SAMPLE / 2
            else:
                weight = self.aug_weight

            # ON-center bipolar cell
            on_sample = self.relu(x + x.detach().clone().uniform_(-1,1) * weight)
            on_sample = on_sample / (on_sample + 1e-5)
            #on_sample = torch.clamp(x + torch.randn_like(x) * 0.1, min=0, max=1)
            if self.mask_aug:
                on_sample = on_sample * mask
            on_samples.append(on_sample)

            # OFF-center bipolar cell
            off_sample = self.relu(1.0 - x + x.detach().clone().uniform_(-1,1) * weight)
            off_sample = off_sample / (off_sample + 1e-5)
            #off_sample = torch.clamp(1.0 - x + torch.randn_like(x) * 0.1, min=0, max=1)
            if self.mask_aug:
                off_sample = off_sample * mask
            off_samples.append(off_sample)

        # run network
        outputs = []
        features = []
        for i in range(NUM_LOOP):
            if bpda:
                assert self.is_attack()
                inputs = torch.cat([x, 1.0 - x], dim=1)
            else:
                on_inputs = sum(on_samples[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]) / NUM_INNER_SAMPLE
                off_inputs = sum(off_samples[NUM_INNER_SAMPLE * i : NUM_INNER_SAMPLE * (i+1)]) / NUM_INNER_SAMPLE
                inputs = torch.cat([on_inputs, off_inputs], dim=1)
            # Normalized input
            inputs = inputs - inputs.mean(-1, keepdim=True).mean(-2, keepdim=True)

            inputs = self.features(inputs)
            inputs = self.avgpool(inputs)
            feats = torch.flatten(inputs, 1)
            preds = self.classifier(feats)

            features.append(feats)
            outputs.append(preds)

        final_pred = sum(outputs) / NUM_LOOP
        return final_pred, features, outputs


    def create_mask(self, w, h, center_x, center_y, alpha=10.0):
        widths = torch.arange(w).view(1, -1).repeat(h,1)
        heights = torch.arange(h).view(-1, 1).repeat(1,w)
        mask = ((widths - center_x)**2 + (heights - center_y)**2).float().sqrt()
        mask = (mask.max() - mask + alpha) ** 0.3
        mask = mask / mask.max()
        mask = (mask + mask.clone().uniform_(0, 1)) > 0.9
        mask.float()
        return mask.unsqueeze(0)

    def set_half_sample(self, half_sample=False):
        self.half_sample = half_sample

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 6
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs):
    return _vgg('vgg11', 'A', False, **kwargs)

def vgg11_bn(**kwargs):
    return _vgg('vgg11_bn', 'A', True, **kwargs)

def vgg13(**kwargs):
    return _vgg('vgg13', 'B', False, **kwargs)

def vgg13_bn(**kwargs):
    return _vgg('vgg13_bn', 'B', True, **kwargs)

def vgg16(**kwargs):
    return _vgg('vgg16', 'D', False, **kwargs)

def vgg16_bn(**kwargs):
    return _vgg('vgg16_bn', 'D', True, **kwargs)

def vgg19(**kwargs):
    return _vgg('vgg19', 'E', False, **kwargs)

def vgg19_bn(**kwargs):
    return _vgg('vgg19_bn', 'E', True, **kwargs)



def create_model(m_type='vgg16', num_classes=1000, first_conv_size=7, num_sample=3, mask_aug=False, samp_aug=False, aug_weight=1.0, mask_center=[5, 16, 27]):
    # create various resnet models
    if m_type == 'vgg16':
        model = vgg16(num_classes=num_classes, num_sample=num_sample, mask_aug=mask_aug, 
                        samp_aug=samp_aug, aug_weight=aug_weight, mask_center=mask_center)

    return model