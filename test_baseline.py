import torch
import torch.nn as nn
import torch.nn.functional as F 

from data.dataloader import get_loader

import os
import time
import random
import attacker

from utils.attack_utils import *

class test_baseline():
    def __init__(self, args, config, logger, model, val=False):
        self.config = config
        self.logger = logger
        self.model = model

        # initialize attacker
        self.adv_test = self.config['attacker_opt']['adv_val']
        if self.adv_test:
            self.attacker = create_adversarial_attacker(config, model, logger)

        # save test
        self.test_save = True if config['test_opt']['save_data'] else False

        # get dataloader
        if val:
            self.phase = 'val'
            self.loader = get_loader(config, 'val', logger)
        else:
            self.phase = 'test'
            self.loader = get_loader(config, 'test', logger)

    def run_val(self, epoch):
        self.logger.info('------------- Start Validation at Epoch {} -----------'.format(epoch))
        total_acc = []
        
        # set model to evaluation
        self.model.eval()

        # save test
        if self.test_save:
            org_list = []
            adv_list = []
            gt_list = []
            pred_list = []

        # run batch
        for i, (inputs, labels, indexes) in enumerate(self.loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            batch_size = inputs.shape[0]
            # print test time
            # trigger adversarial attack or not
            if self.adv_test:
                if self.config['targeted_attack']:
                    adv_targets = get_adv_target(self.config['targeted_type'], self.model, inputs, labels)
                    adv_inputs = self.attacker.get_adv_images(inputs, adv_targets, random_start=rand_adv_init(self.config), targeted=True)
                else:
                    adv_inputs = self.attacker.get_adv_images(inputs, labels, random_start=rand_adv_init(self.config))
                self.model.eval()
                final_inputs = adv_inputs
            else:
                final_inputs = inputs

            # run model
            with torch.no_grad():
                predictions = self.model(final_inputs)

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            total_acc.append((predictions.max(1)[1] == labels).view(-1, 1))

            # save adversarial images
            if self.test_save and i < self.config['test_opt']['save_length']:
                org_list.append(inputs.cpu())
                gt_list.append(labels.cpu())
                pred_list.append(predictions.max(1)[1].cpu())
                if self.adv_test:
                    adv_list.append(adv_inputs.cpu())

        all_acc = torch.cat(total_acc, dim=0).float()
        avg_acc = all_acc.mean().item()
        self.logger.info('Epoch {:5d} Evaluation Complete ==> Total Accuracy : {:9.4f}, Number Samples : {:9d}'.format(epoch, avg_acc, all_acc.shape[0]))

        # set back to training mode again
        self.model.train()

        # save adversarial images
        if self.test_save:
            file_name = os.path.join(self.config['output_dir'], self.config['test_opt']['file_name'])
            adv_output = {
                    'org_images' : torch.cat(org_list, 0),
                    'gt_labels'  : torch.cat(gt_list, 0),
                    'adv_images' : torch.cat(adv_list, 0) if self.adv_test else 0,
                    'pred_labels': torch.cat(pred_list, 0),
                    }
            torch.save(adv_output, file_name)
            self.logger.info('=====> Complete! Adversarial images have been saved to {}'.format(file_name))

        return avg_acc