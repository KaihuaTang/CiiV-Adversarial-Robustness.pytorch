import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Union, List, Dict, Any, cast

import math

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

from models.Base_Model import Base_Model
from utils.train_utils import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(Base_Model):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        num_sample: int = 3, 
        aug_weight: float = 0.9, 
        mask_center: list = [5, 16, 27]
    ) -> None:
        super(VGG, self).__init__()
        self.num_sample = num_sample
        self.aug_weight = aug_weight
        self.mask_center = mask_center
        self.features = features
        self.mlp = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()


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
            out = self.features(inputs)
            size = out.shape[-1]
            out = F.avg_pool2d(out, size)
            out = out.view(out.size(0), -1)
            feats = self.mlp(out)
            preds = self.classifier(feats)

            z_scores.append(z_score.view(1,1).repeat(b, 1))
            features.append(feats)
            outputs.append(preds)

        final_pred = sum([pred / (z + 1e-9) for pred, z in zip(outputs, z_scores)]) / NUM_LOOP

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
            

    def _initialize_weights(self) -> None:
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


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    return _vgg('vgg11', 'A', True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    return _vgg('vgg13', 'B', True, **kwargs)


def vgg16(**kwargs: Any) -> VGG:
    return _vgg('vgg16', 'D', True, **kwargs)


def create_model(m_type='vgg11', num_classes=1000, num_sample=3, aug_weight=0.9, mask_center=[5, 16, 27]):
    # create various resnet models
    if m_type == 'vgg11':
        model = vgg11(num_classes=num_classes, num_sample=num_sample,
                        aug_weight=aug_weight, mask_center=mask_center)
    elif m_type == 'vgg13':
        model = vgg13(num_classes=num_classes, num_sample=num_sample,
                        aug_weight=aug_weight, mask_center=mask_center)
    elif m_type == 'vgg16':
        model = vgg16(num_classes=num_classes, num_sample=num_sample,
                        aug_weight=aug_weight, mask_center=mask_center)
    else:
        raise ValueError('Wrong Model Type')
    return model