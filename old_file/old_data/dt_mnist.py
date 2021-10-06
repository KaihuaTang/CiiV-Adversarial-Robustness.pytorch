import os
import json

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from randaugment import RandAugment

class MNISTData(torchvision.datasets.MNIST):
    def __init__(self, phase, data_path, output_path, logger):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        train = True if phase == 'train' else False
        super(MNISTData, self).__init__(data_path, train, transform=None, target_transform=None, download=True)
        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase

        self.dataset_info = {}
        self.phase = phase
        self.transform = self.get_data_transform(phase)

        # get dataset info
        self.get_dataset_info()
        # save dataset info
        logger.info('=====> Save dataset info')
        self.save_dataset_info(output_path)


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = img.unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index
 

    #######################################
    #  get dataset info
    #######################################
    def get_dataset_info(self):
        self.id2class = {i:_class for i, _class in enumerate(self.classes)}
        self.dataset_info['id2class'] = self.id2class
        

    #######################################
    #  transform
    #######################################
    def get_data_transform(self, phase):
        transform_info = {}

        if phase == 'train':
            trans = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                            transforms.ToTensor(),
                        ])
            transform_info['operations'] = ['ToPILImage()', 'transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),', 'ToTensor()']
        else:
            trans = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(32),
                            transforms.ToTensor(),
                        ])
            transform_info['operations'] = ['ToPILImage()', 'Resize(32)', 'ToTensor()']
        
        # save dataset info
        self.dataset_info['transform_info'] = transform_info

        return trans

    #######################################
    #  Save dataset info
    #######################################
    def save_dataset_info(self, output_path):

        with open(os.path.join(output_path, 'dataset_info_{}.json'.format(self.phase)), 'w') as f:
            json.dump(self.dataset_info, f)

        del self.dataset_info

