import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class ImageNetData(data.Dataset):
    def __init__(self, phase, data_path, output_path, id2label_path, val_info_path, num_classes, blackbox_save, logger):
        super(ImageNetData, self).__init__()
        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase
        self.blackbox_save = blackbox_save

        self.num_classes = num_classes
        self.dataset_info = {}
        self.phase = phase
        self.transform = self.get_data_transform(phase)
        self.data_path = os.path.join(data_path, phase)
        self.val_info_path = val_info_path

        # load dataset category info
        logger.info('=====> Load dataset category info')
        self.id2class, self.id2label, self.label2id = self.load_data_info(id2label_path)

        # load all image info
        if phase == 'train':
            logger.info('=====> Load train image info')
            self.img_paths, self.labels = self.load_train_img_info()
        elif phase == 'val':
            logger.info('=====> Load val image info')
            self.img_paths, self.labels = self.load_val_img_info()
        else:
            logger.info('=====> Load test image info')
            self.img_paths, self.labels = self.load_test_img_info()

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
    #  Load test image paths
    #######################################
    def load_test_img_info(self):
        labels = []
        img_paths = []

        file_list = os.listdir(self.data_path)
        assert len(file_list) > 0
        test_img_num = len(file_list)

        for item in file_list:
            labels.append(-1)
            img_paths.append(os.path.join(self.data_path, item))

        # save dataset info
        self.dataset_info['img_paths'] = img_paths
        self.dataset_info['test_img_num'] = test_img_num

        return img_paths, labels

    #######################################
    #  Load val image paths
    #######################################
    def load_val_img_info(self):
        labels = []
        img_paths = []

        file_list = os.listdir(self.data_path)
        assert len(file_list) > 0

        # load val annotation
        val_anno = json.load(open(self.val_info_path))

        for item in file_list:
            anno = val_anno[item.split('.')[0]]
            if anno in self.label2id:
                label = int(self.label2id[anno]) - 1           # label_id starts from 1, so minus 1 here. 
                labels.append(label)
                img_paths.append(os.path.join(self.data_path, item))

        # save dataset info
        self.dataset_info['img_paths'] = img_paths
        self.dataset_info['val_labels'] = labels
        self.dataset_info['test_img_num'] = len(labels)

        return img_paths, labels

    #######################################
    #  Load train image info
    #######################################
    def load_train_img_info(self):
        id2num = {}  # id to number of samples
        labels = []
        img_paths = []

        for label_id, label_name in self.id2label.items():
            label_path = os.path.join(self.data_path, label_name)
            file_list = os.listdir(label_path)
            assert len(file_list) > 0
            id2num[label_id] = len(file_list)

            for item in file_list:
                labels.append(int(label_id) - 1)                  # label_id starts from 1, so minus 1 here. 
                img_paths.append(os.path.join(label_path, item))

        # save dataset info
        self.dataset_info['id2num'] = id2num
        self.dataset_info['labels'] = labels
        self.dataset_info['img_paths'] = img_paths

        return img_paths, labels


    #######################################
    #  Load dataset category info
    #######################################
    def load_data_info(self, path):
        id2class = {} # id to category name, e.g., kit_fox
        id2label = {} # id to label, e.g., n02119789
        label2id = {}  

        # load dataset info
        with open(path) as f:
            for line in f:
                id = int(line.split()[1])
                label = str(line.split()[0])
                category = str(line.split()[2])
                if id <= self.num_classes:
                    id2class[id] = category
                    id2label[id] = label
                    label2id[label] = id

        # save dataset info
        self.dataset_info['id2class'] = id2class
        self.dataset_info['id2label'] = id2label
        self.dataset_info['label2id'] = label2id
        assert len(id2class) == self.num_classes
        assert len(id2label) == self.num_classes
        assert len(label2id) == self.num_classes
        
        return id2class, id2label, label2id
 

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
                            transforms.RandomResizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ])
            transform_info['operations'] = ['RandomResizedCrop(64)', 'RandomHorizontalFlip()', 'ToTensor()',]
        else:
            trans = transforms.Compose([
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                        ])
            transform_info['operations'] = ['Resize(64)', 'CenterCrop(64)', 'ToTensor()',]
        
        # save dataset info
        self.dataset_info['transform_info'] = transform_info

        return trans