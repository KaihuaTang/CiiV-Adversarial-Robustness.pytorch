import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F 

from test import test_net
import utils.general_utils as utils
from data.dataloader import get_loader
from utils.checkpoint_utils import Checkpoint

import time
import math
import random
import attacker
import numpy as np

class train_net():
    def __init__(self, args, config, logger, model, eval=False):
        self.config = config
        self.logger = logger
        self.model = model
        self.training_opt = config['training_opt']

        self.checkpoint = Checkpoint(config)

        # get dataloader
        self.logger.info('=====> Get train dataloader')
        self.train_loader = get_loader(config, 'train', logger)

        # create action trigger
        self.create_triggers()
    
        # create optimizer
        self.create_optimizer()

        # create scheduler
        self.create_scheduler()

        # adversarial train
        self.create_adversarial_attacker()

        # BPFC training
        self.init_BPFC()

        # mixup training
        self.mixup_on = ('mixup_on' in self.config) and self.config['mixup_on']

        # Instrumental Sample
        self.init_inst_sample()

        # set eval
        if eval:
            self.testing = test_net(args, config, logger, model, val=True)

    def init_BPFC(self):
        self.bpfc_on = ('bpfc_opt' in self.config) and self.config['bpfc_opt']['bpfc_on']
        if self.bpfc_on:
            self.logger.info('=====> Init BPFC')
            self.p_val = self.config['bpfc_opt']['p_val']
            self.p_pow = math.pow(2, self.p_val)
            self.qnoise_scale = math.pow(2, min(self.p_val - 2, 3))
            self.w_ce = self.config['bpfc_opt']['w_ce']
            self.w_reg = self.config['bpfc_opt']['w_reg']
            self.mul_ru = self.config['bpfc_opt']['mul_ru']

    
    def init_inst_sample(self):
        self.inst_on = ('inst_sample' in self.config) and self.config['inst_sample']['inst_on']
        if self.inst_on:
            self.logger.info('=====> Init Instrumental Sampling')
            self.inst_half = self.config['inst_sample']['inst_half']
            self.w_ce = self.config['inst_sample']['w_ce']
            self.w_reg = self.config['inst_sample']['w_reg']
            self.mul_ru = self.config['inst_sample']['mul_ru']
            self.num_loop = self.config['inst_sample']['num_loop']
            if self.inst_half:
                self.model.module.set_half_sample(True)
            else:
                self.model.module.set_half_sample(False)
            if ('accumulate_grad' in self.config['inst_sample'] and self.config['inst_sample']['accumulate_grad']):
                self.accumulate_grad = True
            else:
                self.accumulate_grad = False


    def create_triggers(self):
        self.logger.info('=====> Create Triggers')
        self.epoch_start_trigger = utils.TriggerAction('Trigger at the start of each epoch')
        self.epoch_end_trigger = utils.TriggerAction('Trigger at the end of each epoch')


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
        # set action trigger
        self.epoch_end_trigger.add_action('run scheduler.step()', self.scheduler.step)


    def create_adversarial_attacker(self):
        # training attacker
        self.adv_train = self.config['attacker_opt']['adv_train']
        if self.adv_train:
            if self.config['attacker_opt']['attack_type'] == 'PGD':
                self.attacker = attacker.PGD(self.model, self.logger, self.config, 
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        alpha=self.config['attacker_opt']['attack_alpha'], 
                                        steps=self.config['attacker_opt']['attack_step'],
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'PGDL2':
                self.attacker = attacker.PGDL2(self.model, self.logger, self.config,
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        alpha=self.config['attacker_opt']['attack_alpha'], 
                                        steps=self.config['attacker_opt']['attack_step'],
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'FGSM':
                self.attacker = attacker.FGSM(self.model, self.logger, self.config,
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'FFGSM':
                self.attacker = attacker.FFGSM(self.model, self.logger, self.config,
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        alpha=self.config['attacker_opt']['attack_alpha'], 
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'GN':
                self.attacker = attacker.GN(self.model, self.logger, self.config,
                                        sigma=self.config['attacker_opt']['gn_sigma'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'UN':
                self.attacker = attacker.UN(self.model, self.logger, self.config,
                                        sigma=self.config['attacker_opt']['un_sigma'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'BPDAPGD':
                self.attacker = attacker.BPDAPGD(self.model, self.logger, self.config, 
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        alpha=self.config['attacker_opt']['attack_alpha'], 
                                        steps=self.config['attacker_opt']['attack_step'],
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'EOT':
                self.attacker = attacker.EOT(self.model, self.logger, self.config,
                                        eps=self.config['attacker_opt']['attack_eps'], 
                                        learning_rate=self.config['attacker_opt']['attack_lr'], 
                                        steps=self.config['attacker_opt']['attack_step'], 
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'CW':
                self.attacker = attacker.CW(self.model, self.logger, self.config,
                                            c=self.config['attacker_opt']['c'],
                                            kappa=self.config['attacker_opt']['kappa'],
                                            steps=self.config['attacker_opt']['steps'],
                                            lr=self.config['attacker_opt']['lr'],
                                        )
            else:
                self.logger.raise_error('Wrong Attacker Type')

    
    def rand_adv_init(self):
        if random.uniform(0, 1) < self.config['attacker_opt']['attack_rand_ini']:
            return True  
        else:
            return False

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


    def run(self):
        # start training based on different training strategies
        if self.adv_train:
            self.run_adv()
        elif self.mixup_on:
            self.run_mixup()
        elif self.bpfc_on:
            self.run_bpfc()
        elif self.inst_on:
            if self.inst_half:
                self.run_inst_half()
            if self.accumulate_grad:
                self.run_inst_accu()
            else:
                self.run_inst()
        else:
            self.run_clean()


    def run_clean(self):
        # Start Training
        self.logger.info('=====> Start Naive Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            self.model.train()

            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # naive training
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                loss = F.cross_entropy(preds, labels)
                iter_info_print = {'ce_loss' : loss.sum().item(),}  
                # backward
                loss.backward()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)
            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)
        # save best model path
        self.checkpoint.save_best_model(self.logger)

    def run_mixup(self):
        # Start Training
        self.logger.info('=====> Start Mixup Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            self.model.train()

            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # naive training
                inputs, labels = inputs.cuda(), labels.cuda()
                #labels = F.one_hot(labels, self.config['networks']['params']['num_classes'])
                # mix-up 
                batch_size = inputs.shape[0]
                lam = np.random.beta(1.0, 1.0)

                index = torch.randperm(batch_size).cuda()
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                labels_a = labels
                labels_b = labels[index]
                
                preds = self.model(mixed_inputs)
                loss_a = F.cross_entropy(preds, labels_a)
                loss_b = F.cross_entropy(preds, labels_b)
                loss = lam * loss_a + (1 - lam) * loss_b

                iter_info_print = {'ce_loss' : loss.sum().item(),}  
                # backward
                loss.backward()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)
            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)
        # save best model path
        self.checkpoint.save_best_model(self.logger)

    def run_adv(self):
        # Start Training
        self.logger.info('=====> Start Adversarial Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            
            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # adv train
                inputs, labels = inputs.cuda(), labels.cuda()
                adv_inputs = self.attacker.get_adv_images(inputs, labels, random_start=self.rand_adv_init())
                self.model.train()
                preds = self.model(adv_inputs)
                loss = F.cross_entropy(preds, labels)
                iter_info_print = {'adv_loss' : loss.sum().item(),}  

                # backward
                loss.backward()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)
            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)

        # save best model path
        self.checkpoint.save_best_model(self.logger)


    def run_bpfc(self):
        # Start Training
        self.logger.info('=====> Start BPFC Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            
            self.model.train()
            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs, labels = inputs.cuda(), labels.cuda()

                # BPFS pre-processing
                q_inputs = torch.round(inputs * 255)
                q_noises = torch.Tensor(inputs.size()).uniform_(-1,1).to(inputs.device) * self.qnoise_scale
                q_inputs = q_inputs + q_noises
                q_inputs = q_inputs - (q_inputs % self.p_pow) + (self.p_pow / 2)
                q_inputs = torch.clamp(q_inputs, 0, 255)
                q_inputs = q_inputs / 255
                # BPFC training
                preds = self.model(inputs)
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
                loss.backward()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)

            # update regression loss weight for BPFC or Instrumental Sampling
            if (epoch in self.config['bpfc_opt']['milestones']):
                self.logger.info('update regression weight from {} to {}'.format(self.w_reg, self.w_reg * self.mul_ru))
                self.w_reg = self.w_reg * self.mul_ru

            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)

        # save best model path
        self.checkpoint.save_best_model(self.logger)


    def run_inst_half(self):
        # Start Training
        self.logger.info('=====> Start Instrumental Sampler for Half of the Network Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            
            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs, labels = inputs.cuda(), labels.cuda()

                # instrumental sampling training by merging at the middle of the network
                iter_info_print = {}
                all_regs = []
                preds, other_feats = self.model(inputs, loop=self.num_loop)
                ce_loss = F.cross_entropy(preds, labels)
                iter_info_print['ce_loss'] = ce_loss.sum().item()

                for i in range(len(other_feats) - 1):
                    if self.config['inst_sample']['reg_loss'] == 'L2':
                        reg_loss = self.l2_loss(other_feats[i], other_feats[i+1])
                        iter_info_print['reg_l2loss_{}'.format(i)] = reg_loss.sum().item()
                    elif self.config['inst_sample']['reg_loss'] == 'L1':
                        reg_loss = self.smooth_l1_loss(other_feats[i], other_feats[i+1])
                        iter_info_print['reg_l1loss_{}'.format(i)] = reg_loss.sum().item()
                    else:
                        raise ValueError('Wrong Reg Loss Type')
                    all_regs.append(reg_loss)
                loss = self.w_ce * ce_loss + self.w_reg * sum(all_regs) / len(all_regs)
                iter_info_print['w_ce'] = self.w_ce
                iter_info_print['w_reg'] = self.w_reg

                # backward
                loss.backward()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)

            # update regression loss weight for BPFC or Instrumental Sampling
            if (epoch in self.config['inst_sample']['milestones']):
                self.logger.info('update regression weight from {} to {}'.format(self.w_reg, self.w_reg * self.mul_ru))
                self.w_reg = self.w_reg * self.mul_ru

            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)

        # save best model path
        self.checkpoint.save_best_model(self.logger)


    def run_inst(self):
        # Start Instrumental Sampler Training
        self.logger.info('=====> Start Instrumental Sampler Training')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            # print training time
            start_time = time.time()
            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs, labels = inputs.cuda(), labels.cuda()

                # instrumental sampling training by running all samples parallelly
                iter_info_print = {}
                all_ces = []
                all_regs = []
                preds, other_feats, other_logits = self.model(inputs, loop=self.num_loop)
                for i, logits in enumerate(other_logits):
                    ce_loss = F.cross_entropy(logits, labels)
                    iter_info_print['ce_loss_{}'.format(i)] = ce_loss.sum().item()
                    all_ces.append(ce_loss)
                
                if self.config['inst_sample']['reg_type'] == 'neighbour':
                    for i in range(len(other_feats) - 1):
                        if self.config['inst_sample']['reg_loss'] == 'L2':
                            reg_loss = self.l2_loss(other_feats[i], other_feats[i+1])
                            iter_info_print['reg_l2loss_{}'.format(i)] = reg_loss.sum().item()
                        elif self.config['inst_sample']['reg_loss'] == 'L1':
                            reg_loss = self.smooth_l1_loss(other_feats[i], other_feats[i+1])
                            iter_info_print['reg_l1loss_{}'.format(i)] = reg_loss.sum().item()
                        else:
                            raise ValueError('Wrong Reg Loss Type')
                        all_regs.append(reg_loss)
                elif self.config['inst_sample']['reg_type'] == 'regmean':
                    mean_feat = sum(other_feats) / len(other_feats)
                    for i in range(len(other_feats)):
                        if self.config['inst_sample']['reg_loss'] == 'L2':
                            reg_loss = self.l2_loss(other_feats[i], mean_feat)
                            iter_info_print['reg_l2loss_{}'.format(i)] = reg_loss.sum().item()
                        elif self.config['inst_sample']['reg_loss'] == 'L1':
                            reg_loss = self.smooth_l1_loss(other_feats[i], mean_feat)
                            iter_info_print['reg_l1loss_{}'.format(i)] = reg_loss.sum().item()
                        else:
                            raise ValueError('Wrong Reg Loss Type')
                        all_regs.append(reg_loss)
                elif self.config['inst_sample']['reg_type'] == 'none':
                    all_regs.append(0)
                else:
                    raise ValueError('Wrong Reg Feature Type')
                
                loss = self.w_ce * sum(all_ces) / len(all_ces) + self.w_reg * sum(all_regs) / len(all_regs)
                iter_info_print['w_ce'] = self.w_ce
                iter_info_print['w_reg'] = self.w_reg

                # backward
                loss.backward()
                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)

            # update regression loss weight for BPFC or Instrumental Sampling
            if (epoch in self.config['inst_sample']['milestones']):
                self.logger.info('update regression weight from {} to {}'.format(self.w_reg, self.w_reg * self.mul_ru))
                self.w_reg = self.w_reg * self.mul_ru
            
            # print training time
            print("--- {} seconds for the {} epoch ---".format(time.time() - start_time, epoch))

            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)

        # save best model path
        self.checkpoint.save_best_model(self.logger)


    def run_inst_accu(self):
        # Start Instrumental Sampler Training
        self.logger.info('=====> Start Instrumental Sampler Training using Accumulated Grad')

        # run epoch
        for epoch in range(self.training_opt['num_epochs']):
            self.logger.info('------------ Start Epoch {} -----------'.format(epoch))
            # trigger epoch start actions
            self.epoch_start_trigger.run_all(self.logger)
            
            # run batch
            total_batch = len(self.train_loader)
            for step, (inputs, labels, indexes) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs, labels = inputs.cuda(), labels.cuda()

                # instrumental sampling training by running all samples parallelly
                iter_info_print = {}
                pred_hist = []
                feat_hist = []
                for i in range(self.num_loop):
                    preds, other_feats, other_logits = self.model(inputs, loop=1)
                    crt_feats = other_feats[0]
                    crt_preds = other_logits[0]
                    ce_loss = F.cross_entropy(preds, labels)
                    iter_info_print['ce_loss_{}'.format(i)] = ce_loss.sum().item()
                    # feature similar
                    if len(feat_hist) > 0:
                        target_feat = sum(feat_hist) / len(feat_hist)
                        if self.config['inst_sample']['reg_loss'] == 'L2':
                            reg_loss = self.l2_loss(crt_feats, target_feat)
                            iter_info_print['reg_l2loss_{}'.format(i)] = reg_loss.sum().item()
                        elif self.config['inst_sample']['reg_loss'] == 'L1':
                            reg_loss = self.smooth_l1_loss(crt_feats, target_feat)
                            iter_info_print['reg_l1loss_{}'.format(i)] = reg_loss.sum().item()
                        else:
                            raise ValueError('Wrong Reg Loss Type')
                        loss = self.w_ce * ce_loss + self.w_reg * reg_loss
                    else:
                        loss = ce_loss
                    # update hist
                    feat_hist.append(crt_feats.detach().clone())
                    pred_hist.append(crt_preds.detach().clone())
                    # backward
                    loss.backward()

                iter_info_print['w_ce'] = self.w_ce
                iter_info_print['w_reg'] = self.w_reg
                preds = sum(pred_hist) / len(pred_hist)

                # calculate accuracy
                accuracy = (preds.max(1)[1] == labels).sum().float() / preds.shape[0]

                # log information 
                iter_info_print.update( {'Accuracy' : accuracy.item(), 'Loss' : loss.sum().item(), 'Poke LR' : float(self.optimizer.param_groups[0]['lr'])} )
                self.logger.info_iter(epoch, step, total_batch, iter_info_print, self.config['logger_opt']['print_iter'])
                if self.config['logger_opt']['print_grad'] and step % 1000 == 0:
                    utils.print_grad(self.model.named_parameters())

                self.optimizer.step()

            # evaluation on validation set
            self.optimizer.zero_grad()
            val_acc = self.testing.run_val(epoch)

            # update regression loss weight for BPFC or Instrumental Sampling
            if (epoch in self.config['inst_sample']['milestones']):
                self.logger.info('update regression weight from {} to {}'.format(self.w_reg, self.w_reg * self.mul_ru))
                self.w_reg = self.w_reg * self.mul_ru

            # checkpoint
            self.checkpoint.save(self.model, epoch, self.logger, acc=val_acc)
            # trigger epoch end actions
            self.epoch_end_trigger.run_all(self.logger)

        # save best model path
        self.checkpoint.save_best_model(self.logger)