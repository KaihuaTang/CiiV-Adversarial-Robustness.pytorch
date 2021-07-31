import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image


class TOYData(data.Dataset):
    def __init__(self, phase, output_path, num_classes, logger):
        super(TOYData, self).__init__()
        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase

        if phase == 'train':
            self.num_data = 10000
        elif phase == 'val':
            self.num_data = 1000
        elif phase == 'test':
            self.num_data = 1000

        self.num_classes = num_classes
        self.dataset_info = {}
        self.phase = phase
        self.generate_data()

        # save dataset info
        logger.info('=====> Save dataset info')
        self.save_dataset_info(output_path)


    def __len__(self):
        return len(self.labels)

    def generate_data(self):
        assert self.num_classes == 3
        self.images = []
        self.labels = torch.randint(0, self.num_classes, (self.num_data,)).tolist()

        for i in range(self.num_data):
            tp = str(self.labels[i])
            self.images.append(self.generate_patterns(tp=tp))

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return image, label, index

    def generate_patterns(self, tp='1', noise=True):
        if noise:
            tps = {'0':{'r':0.8, 'g':0.1, 'b':0.1}, '1':{'r':0.1, 'g':0.8, 'b':0.1}, '2':{'r':0.1, 'g':0.1, 'b':0.8}}
            r = torch.zeros(1, 16, 1, 16, 1).uniform_(0,tps[tp]['r']).repeat(1,1,4,1,4).contiguous().view(1,64,64)
            g = torch.zeros(1, 16, 1, 16, 1).uniform_(0,tps[tp]['g']).repeat(1,1,4,1,4).contiguous().view(1,64,64)
            b = torch.zeros(1, 16, 1, 16, 1).uniform_(0,tps[tp]['b']).repeat(1,1,4,1,4).contiguous().view(1,64,64)
            img = torch.cat([r,g,b], dim=0)
            img = img.contiguous().view(3, 64*64).contiguous().permute(1,0).max(-1)[1]
            img = torch.nn.functional.one_hot(img, 3).permute(1,0).contiguous().view(3,64,64).float()
            img = img * 0.2 + 0.8
        else:
            img = torch.zeros(3, 64, 64)
    
        radius = torch.zeros(1).float().uniform_(15, 25).item()
        if tp == '0':
            w = torch.arange(64).view(1,64).repeat(64,1).float()
            h = torch.arange(64).view(64,1).repeat(1,64).float()
            mask =  ((w - 31.5)**2 + (h - 31.5)**2) ** 0.5
            mask = (mask < radius).float().view(1,64,64)
        elif tp == '1':
            w = torch.arange(64).view(1,64).repeat(64,1).float()
            h = torch.arange(64).view(64,1).repeat(1,64).float()
            mask =  ((w - 31.5).abs() < radius).float() + ((h - 31.5).abs() < radius).float()
            mask = (mask >= 2.0).float().view(1,64,64)
        elif tp == '2':
            w = torch.arange(64).view(1,64).repeat(64,1).float()
            h = torch.arange(64).view(64,1).repeat(1,64).float()
            r1 = (h < (25+radius)).float()
            r2 = (w+h > (64-radius/2)).float()
            r3 = ((h-w) > -radius/2).float()
            mask =  ((r1+r2+r3) >= 3.0).float().view(1,64,64)
    
        img = torch.clamp(img - mask, 0, 1)
        return img

 

    #######################################
    #  Save dataset info
    #######################################
    def save_dataset_info(self, output_path):

        with open(os.path.join(output_path, 'dataset_info_{}.json'.format(self.phase)), 'w') as f:
            json.dump(self.dataset_info, f)

        del self.dataset_info