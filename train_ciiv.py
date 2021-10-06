import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F 

import utils.general_utils as utils
from data.dataloader import get_loader
from utils.checkpoint_utils import Checkpoint

import time
import math
import random
import attacker
import numpy as np

from utils.attack_utils import *
from utils.train_utils import *
from utils.test_utils import *

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10):
        super(LabelSmoothingLoss, self).__init__()
        self.cls = classes

    def forward(self, pred, target, confidence, dim=-1):
        smoothing = 1.0 - confidence
        pred = pred.log_softmax(dim=dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=dim))
        
class train_ciiv():
    def __init__(self, args, config, logger, model, eval=False):
        self.config = config
        self.logger = logger
        self.model = model
        self.training_opt = config['training_opt']
        self.checkpoint = Checkpoint(config)
        self.logger.info('============= Training Strategy: CiiV Training ============')

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', logger)

        # init inst setting
        self.init_inst_sample()

        # create optimizer
        self.create_optimizer()

        # create scheduler
        self.create_scheduler()

        # create loss
        self.creat_loss()
        
        # adversarial train
        self.adv_train = self.config['attacker_opt']['adv_train']
        if self.adv_train:
            self.attacker = create_adversarial_attacker(config, model, logger)

        # set eval
        if eval:
            # choosing test strategy
            test_func = get_test_func(config)
            # start testing
            self.testing = test_func(args, config, logger, model, val=True)

    def init_inst_sample(self):
        self.logger.info('=====> Init Instrumental Sampling')
        self.w_ce = self.config['inst_sample']['w_ce']
        self.w_reg = self.config['inst_sample']['w_reg']
        self.mul_ru = self.config['inst_sample']['mul_ru']
        self.num_loop = self.config['inst_sample']['num_loop']

    def creat_loss(self):
        if self.config['inst_sample']['ce_smooth']:
            self.criterion = LabelSmoothingLoss(classes=self.config['networks']['params']['num_classes'])
        else:
            self.criterion = nn.CrossEntropyLoss()

    def create_optimizer(self):
        self.logger.info('=====> Create optimizer')
        optim_params = self.training_opt['optim_params']
        optim_params_dict = {'params': self.model.parameters(), 
                            'lr': optim_params['lr'],
                            'momentum': optim_params['momentum'],
                            'weight_decay': optim_params['weight_decay']
                            }
    
        if self.training_opt['optimizer'] == 'Adam':
            self.optimizer = optim.Adam([optim_params_dict, ])
        elif self.training_opt['optimizer'] == 'SGD':
            self.optimizer = optim.SGD([optim_params_dict, ])
        else:
            self.logger.info('********** ERROR: unidentified optimizer **********')


    def create_scheduler(self):
        self.logger.info('=====> Create Scheduler')
        scheduler_params = self.training_opt['scheduler_params']
        if self.training_opt['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
        elif self.training_opt['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=scheduler_params['gamma'], milestones=scheduler_params['milestones'])
        else:
            self.logger.info('********** ERROR: unidentified optimizer **********')

    def l2_loss(self, x, y):
        diff = x - y
        diff = diff*diff
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff
    
    def smooth_l1_loss(self, x, y):
        diff = F.smooth_l1_loss(x, y, reduction='none')
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff      

    def get_mean_wo_i(self, inputs, i):
        return (sum(inputs) - inputs[i]) / float(len(inputs) - 1)

    def run(self):
        # Start Training
        self.logger.info('=====> Start Naive Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            self.model.train()

            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                # naive training
                inputs, labels = inputs.cuda(), labels.cuda()
                if self.adv_train:
                    final_inputs = self.attacker.get_adv_images(inputs, labels)
                else:
                    final_inputs = inputs

                # instrumental sampling training by running all samples parallelly
                iter_info_print = {}
                all_ces = []
                all_regs = []
                preds, z_scores, features, logits = self.model(final_inputs, loop=self.num_loop)
                for i, logit in enumerate(logits):
                    if self.config['inst_sample']['ce_smooth']:
                        ce_loss = self.criterion(logit, labels, confidence=float(z_scores[i]))
                    else:
                        ce_loss = self.criterion(logit, labels)
                    iter_info_print['ce_loss_{}'.format(i)] = ce_loss.sum().item()
                    all_ces.append(ce_loss)

                for i in range(len(features)):
                    if self.config['inst_sample']['reg_loss'] == 'L2':
                        reg_loss = self.l2_loss(features[i] * self.get_mean_wo_i(z_scores, i), self.get_mean_wo_i(features, i) * z_scores[i])
                        iter_info_print['ciiv_l2loss_{}'.format(i)] = reg_loss.sum().item()
                    elif self.config['inst_sample']['reg_loss'] == 'L1':
                        reg_loss = self.smooth_l1_loss(features[i] * self.get_mean_wo_i(z_scores, i), self.get_mean_wo_i(features, i) * z_scores[i])
                        iter_info_print['ciiv_l1loss_{}'.format(i)] = reg_loss.sum().item()
                    else:
                        raise ValueError('Wrong Reg Loss Type')
                    all_regs.append(reg_loss)

                loss = self.w_ce * sum(all_ces) / len(all_ces) + self.w_reg * sum(all_regs) / len(all_regs)
                iter_info_print['w_ce'] = self.w_ce
                iter_info_print['w_reg'] = self.w_reg
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)

            # update regression loss weight for BPFC or Instrumental Sampling
            if (epoch in self.config['inst_sample']['milestones']):
                self.logger.info('update regression weight from {} to {}'.format(self.w_reg, self.w_reg * self.mul_ru))
                self.w_reg = self.w_reg * self.mul_ru

            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # update scheduler
            self.scheduler.step()
        # save best model path
        self.checkpoint.save_best_model(self.logger)

    