import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.Base_Model import Base_Model

class VGG(Base_Model):
    def __init__(self, features, num_classes=1000, init_weights=True, num_trial=10, sigma=0.25):
        super(VGG, self).__init__()
        self.num_trial = num_trial
        self.sigma = sigma
        self.num_classes = num_classes
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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

    def forward(self, x, bpda=False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.training or self.is_attack():
            if not bpda:
                x = self.sample_noise(x)
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        # Randomized Smoothing Inference
        # original https://github.com/locuslab/smoothing/blob/master/code/core.py#L61 will return ABSTAIN if the output 
        # doesn't pass the binom_test. However, we will always return the prediction
        else:
            counts = []
            for _ in range(self.num_trial):
                if not bpda:
                    x_t = self.sample_noise(x)
                x_t = self.features(x_t)
                x_t = self.avgpool(x_t)
                x_t = torch.flatten(x_t, 1)
                x_t = self.classifier(x_t)
                pred = x_t.max(-1)[1]
                counts.append(F.one_hot(pred, self.num_classes))
            return sum(counts)            

    def sample_noise(self, x):
        noise_x = x + self.sigma * torch.randn_like(x).to(x.device)
        noise_x = torch.clamp(noise_x, min=0, max=1)
        return noise_x

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
    in_channels = 3
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


def create_model(m_type='vgg16', num_classes=1000, first_conv_size=7, num_trial=10, sigma=0.25):
    # create various resnet models
    if m_type == 'vgg16':
        model = vgg16(num_classes=num_classes, num_trial=num_trial, sigma=sigma)

    return model