import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from .dt_imagenet import ImageNetData
from .dt_mini_imagenet import miniImageNetData
from .dt_cifar import CIFAR10Data, CIFAR100Data
from .dt_mnist import MNISTData
from .dt_toy import TOYData

##################################
# return a dataloader
##################################
def get_loader(config, phase, logger, blackbox_save=False):
    dataset = config['dataset']['name']
    if dataset == 'imagenet':
        split = ImageNetData(phase=phase, 
                             data_path=config['dataset']['data_path'], 
                             output_path=config['output_dir'], 
                             id2label_path=config['dataset']['id2label_path'], 
                             val_info_path=config['dataset']['val_info_path'],
                             num_classes=config['dataset']['num_classes'],
                             blackbox_save=blackbox_save,
                             logger=logger)
    elif dataset == 'mini-imagenet':
        split = miniImageNetData(phase=phase, 
                             data_path=config['dataset']['data_path'], 
                             category_path=config['dataset']['category_path'],
                             train_path=config['dataset']['train_path'], 
                             test_path=config['dataset']['test_path'], 
                             val_path=config['dataset']['val_path'], 
                             output_path=config['output_dir'], 
                             num_classes=config['dataset']['num_classes'],
                             blackbox_save=blackbox_save,
                             logger=logger)
    elif dataset == 'cifar10':
        split = CIFAR10Data(phase=phase, 
                             data_path=config['dataset']['data_path'], 
                             output_path=config['output_dir'], 
                             blackbox_save=blackbox_save,
                             logger=logger)
    elif dataset == 'cifar100':
        split = CIFAR100Data(phase=phase, 
                             data_path=config['dataset']['data_path'], 
                             output_path=config['output_dir'], 
                             blackbox_save=blackbox_save,
                             logger=logger)
    elif dataset == 'mnist':
        split = MNISTData(phase=phase, 
                          data_path=config['dataset']['data_path'], 
                          output_path=config['output_dir'], 
                          blackbox_save=blackbox_save,
                          logger=logger)
    elif dataset == 'TOYData':
        split = TOYData(phase=phase, 
                        output_path=config['output_dir'], 
                        num_classes=config['dataset']['num_classes'],
                        logger=logger)
    else:
        logger.info('********** ERROR: unidentified dataset **********')
    
    loader = data.DataLoader(
        split, 
        batch_size=config['training_opt']['batch_size'],
        shuffle=True if phase == 'train' else False,
        pin_memory=True,
        num_workers=config['training_opt']['data_workers'],
    )

    return loader


