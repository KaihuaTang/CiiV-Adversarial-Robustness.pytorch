import torch
import torch.nn as nn

import random
import attacker

from train_baseline import train_baseline
from train_mixup import train_mixup
from train_ciiv import train_ciiv
from train_ciiv_img import train_ciiv_img
from train_ciiv_mixup import train_ciiv_mixup
from train_bpfc import train_bpfc


def get_train_func(config):
    # choosing training strategy
    if config['strategy']['train_type'] == 'baseline':
        training_func = train_baseline
    elif config['strategy']['train_type'] == 'mixup':
        training_func = train_mixup
    elif config['strategy']['train_type'] == 'ciiv':
        training_func = train_ciiv
    elif config['strategy']['train_type'] == 'ciiv_img':
        training_func = train_ciiv_img
    elif config['strategy']['train_type'] == 'ciiv_mixup':
        training_func = train_ciiv_mixup
    elif config['strategy']['train_type'] == 'bpfc':
        training_func = train_bpfc
    else:
        raise ValueError('Wrong Training Strategy')
    return training_func

def rgb_norm(images, config):
    # set mean, std for different dataset
    if config['dataset']['name'] == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
    elif config['dataset']['name'] == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2673, 0.2564, 0.2762]
    elif config['dataset']['name'] == 'mini-imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif config['dataset']['name'] == 'TOYData':
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        raise ValueError('Wrong Dataset ({}) for RGB normalization.'.format(config['dataset']['name']))
    # apply normalization
    mean = torch.tensor(mean).view(1,3,1,1).to(images.device)
    std = torch.tensor(std).view(1,3,1,1).to(images.device)
    images = (images - mean) / std
    return images