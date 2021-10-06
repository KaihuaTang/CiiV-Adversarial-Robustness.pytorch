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


class train_mixup():
    def __init__(self, args, config, logger, model, eval=False):
        self.config = config
        self.logger = logger
        self.model = model
        self.training_opt = config['training_opt']
        self.checkpoint = Checkpoint(config)
        self.logger.info('============= Training Strategy: MixUp Training ============')

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', logger)

        self.loss_fc = nn.CrossEntropyLoss()
        
        # create optimizer
        self.create_optimizer()

        # create scheduler
        self.create_scheduler()
        
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


    def mixup_data(self, x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.shape[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
        
    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_fc(pred, y_a) + (1 - lam) * self.loss_fc(pred, y_b)

    def mixup_accuracy(self, pred, y_a, y_b, lam):
        correct = lam * (pred.max(1)[1] == y_a) + (1 - lam) * (pred.max(1)[1] == y_b)
        accuracy = correct.sum().float() / pred.shape[0]
        return accuracy


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

                # mixup
                final_inputs, labels_a, labels_b, lam = self.mixup_data(final_inputs, labels)

                preds = self.model(final_inputs)
                loss = self.mixup_criterion(preds, labels_a, labels_b, lam)
                iter_info_print = {'mixup_loss' : loss.sum().item(),}  
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # calculate accuracy
                accuracy = self.mixup_accuracy(preds, labels_a, labels_b, lam)

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)
            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # update scheduler
            self.scheduler.step()
        # save best model path
        self.checkpoint.save_best_model(self.logger)

    
