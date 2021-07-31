import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image


class miniImageNetData(data.Dataset):
    def __init__(self, phase, data_path, category_path, train_path, test_path, val_path, output_path, num_classes, blackbox_save, logger):
        super(miniImageNetData, self).__init__()
        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase
        self.blackbox_save = blackbox_save

        self.num_classes = num_classes
        self.dataset_info = {}
        self.phase = phase
        self.transform = self.get_data_transform(phase)
        self.data_path = data_path
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        # load dataset category info
        logger.info('=====> Load dataset category info')
        self.id2label, self.label2id = self.load_data_info(category_path)

        # load all image info
        if phase == 'train':
            logger.info('=====> Load train image info')
            self.img_paths, self.labels = self.load_img_info(self.train_path)
        elif phase == 'val':
            logger.info('=====> Load val image info')
            self.img_paths, self.labels = self.load_img_info(self.val_path)
        else:
            logger.info('=====> Load test image info')
            self.img_paths, self.labels = self.load_img_info(self.test_path)

        # save dataset info
        logger.info('=====> Save dataset info')
        self.save_dataset_info(output_path)


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


    #######################################
    #  Load image info
    #######################################
    def load_img_info(self, path):
        data_file = pd.read_csv(path).values

        labels = []
        img_paths = []

        for item in data_file:
            name = item[1]
            label = self.label2id[item[2]]
            img_path = os.path.join(self.data_path, name)
            labels.append(label)
            img_paths.append(img_path)

        # save dataset info
        self.dataset_info['labels'] = labels
        self.dataset_info['img_paths'] = img_paths

        return img_paths, labels


    #######################################
    #  Load dataset category info
    #######################################
    def load_data_info(self, path):
        id2label = {} # id to label, e.g., n02119789
        label2id = {}  

        category_list = pd.read_csv(path).values

        for item in category_list:
            id = int(item[0])
            label = str(item[1])
            id2label[id] = label
            label2id[label] = id

        # save dataset info
        self.dataset_info['id2label'] = id2label
        self.dataset_info['label2id'] = label2id
        assert len(id2label) == self.num_classes
        assert len(label2id) == self.num_classes
        
        return id2label, label2id
 

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
            trans = transforms.Compose([
                            transforms.RandomResizedCrop(84),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ])
            transform_info['operations'] = ['RandomResizedCrop(84)', 'RandomHorizontalFlip()', 'ToTensor()',]
        else:
            trans = transforms.Compose([
                            transforms.Resize(84),
                            transforms.CenterCrop(84),
                            transforms.ToTensor(),
                        ])
            transform_info['operations'] = ['Resize(84)', 'CenterCrop(84)', 'ToTensor()',]
        
        # save dataset info
        self.dataset_info['transform_info'] = transform_info

        return trans