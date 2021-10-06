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


class train_bpfc():
    def __init__(self, args, config, logger, model, eval=False):
        self.config = config
        self.logger = logger
        self.model = model
        self.training_opt = config['training_opt']
        self.checkpoint = Checkpoint(config)
        self.logger.info('============= Training Strategy: BPFC Training ============')

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', logger)

        # create optimizer
        self.create_optimizer()

        # create scheduler
        self.create_scheduler()

        # BPFC training
        self.init_BPFC()
        
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

    def init_BPFC(self):
        self.logger.info('=====> Init BPFC')
        self.p_val = self.config['bpfc_opt']['p_val']
        self.p_pow = math.pow(2, self.p_val)
        self.qnoise_scale = math.pow(2, min(self.p_val - 2, 3))
        self.w_ce = self.config['bpfc_opt']['w_ce']
        self.w_reg = self.config['bpfc_opt']['w_reg']
        self.mul_ru = self.config['bpfc_opt']['mul_ru']

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

                # BPFS pre-processing
                q_inputs = torch.round(final_inputs * 255)
                q_noises = torch.Tensor(final_inputs.size()).uniform_(-1,1).to(final_inputs.device) * self.qnoise_scale
                q_inputs = q_inputs + q_noises
                q_inputs = q_inputs - (q_inputs % self.p_pow) + (self.p_pow / 2)
                q_inputs = torch.clamp(q_inputs, 0, 255)
                q_inputs = q_inputs / 255
                # BPFC training
                preds = self.model(final_inputs)
                q_preds = self.model(q_inputs)
                # calculate loss
                ce_loss = F.cross_entropy(preds, labels)
                reg_loss = self.l2_loss(preds, q_preds)
                loss = self.w_ce * ce_loss + self.w_reg * reg_loss
                iter_info_print = {'ce_loss' : ce_loss.sum().item(),
                                       'reg_loss': reg_loss.sum().item(),
                                       'w_ce' : self.w_ce,
                                       'w_reg' : self.w_reg,}

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
            if (epoch in self.config['bpfc_opt']['milestones']):
                self.logger.info('update regression weight from {} to {}'.format(self.w_reg, self.w_reg * self.mul_ru))
                self.w_reg = self.w_reg * self.mul_ru

            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # update scheduler
            self.scheduler.step()
        # save best model path
        self.checkpoint.save_best_model(self.logger)

    