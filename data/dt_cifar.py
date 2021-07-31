import os
import json

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class CIFAR10Data(torchvision.datasets.CIFAR10):
    def __init__(self, phase, data_path, output_path, blackbox_save, logger):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        train = True if phase == 'train' else False
        super(CIFAR10Data, self).__init__(data_path, train, transform=None, target_transform=None, download=True)
        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase
        self.blackbox_save = blackbox_save

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

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

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
    #  Save dataset info
    #######################################
    def save_dataset_info(self, output_path):

        with open(os.path.join(output_path, 'dataset_info_{}.json'.format(self.phase)), 'w') as f:
            json.dump(self.dataset_info, f)

        del self.dataset_info


    #######################################
    #  transform
    #######################################
    def get_data_transform(self, phase):
        transform_info = {}

        if phase == 'train' and (not self.blackbox_save):
            base_trans = [transforms.RandomCrop(size=32,padding=4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),]
            transform_info['operations'] = ['transforms.RandomCrop(size=32,padding=4)', 'RandomHorizontalFlip()', 'ToTensor()']

            trans = transforms.Compose(base_trans)

        else:
            base_trans = [transforms.ToTensor(),]
            transform_info['operations'] = ['ToTensor()']

            trans = transforms.Compose(base_trans)
            
        # save dataset info
        self.dataset_info['transform_info'] = transform_info

        return trans



class CIFAR100Data(CIFAR10Data):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }