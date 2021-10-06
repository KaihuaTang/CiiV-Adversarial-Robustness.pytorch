import torch
import torch.nn as nn
import torch.nn.functional as F 

from data.dataloader import get_loader

import os
import time
import random
import attacker


class test_net():
    def __init__(self, args, config, logger, model, val=False):
        self.config = config
        self.logger = logger
        self.model = model

        # initialize attacker
        self.create_adversarial_attacker()

        # init inst setting
        self.init_inst_sample()

        # save test
        self.test_save = True if config['save_test']['save_data'] else False

        # get dataloader
        if val:
            self.phase = 'val'
            self.loader = get_loader(config, 'val', logger)
        else:
            self.phase = 'test'
            self.loader = get_loader(config, 'test', logger)

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

    def create_adversarial_attacker(self):
        # attacker
        self.adv_test = self.config['attacker_opt']['adv_val']
        if self.adv_test:
            if self.config['attacker_opt']['attack_type'] == 'PGD':
                self.attacker = attacker.PGD(self.model, self.logger, self.config, 
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        alpha=self.config['attacker_opt']['attack_alpha'], 
                                        steps=self.config['attacker_opt']['attack_step'],
                                        eot_iter=self.config['attacker_opt']['eot_iter'],
                                        )
            elif self.config['attacker_opt']['attack_type'] == 'ADAPTIVE':
                self.attacker = attacker.ADAPTIVE(self.model, self.logger, self.config, 
                                        eps=self.config['attacker_opt']['attack_eps'],
                                        alpha=self.config['attacker_opt']['attack_alpha'], 
                                        steps=self.config['attacker_opt']['attack_step'],
                                        weight=self.config['attacker_opt']['weight'],
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
            elif self.config['attacker_opt']['attack_type'] == 'BFS':
                self.attacker = attacker.BFS(self.model, self.logger, self.config,
                                        sigma=self.config['attacker_opt']['sigma'],
                                        eps=self.config['attacker_opt']['eps'],
                                        steps=self.config['attacker_opt']['steps'],
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
                                        sigma=self.config['attacker_opt']['sigma'],
                                        eps=self.config['attacker_opt']['eps'],
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
            elif self.config['attacker_opt']['attack_type'] == 'SPSA':
                self.attacker = attacker.SPSA(self.model, self.logger, self.config,
                                            delta=self.config['attacker_opt']['delta'],
                                            steps=self.config['attacker_opt']['steps'],
                                            batch_size=self.config['attacker_opt']['batch_size'],
                                            lr=self.config['attacker_opt']['lr'],
                                        )
            else:
                self.logger.raise_error('Wrong Attacker Type')

    
    def rand_adv_init(self):
        if random.uniform(0, 1) < self.config['attacker_opt']['attack_rand_ini']:
            return True  
        else:
            return False

    def get_blackbox_name(self):
        model_name = self.config['networks']['params']['m_type']
        attack_type = self.config['attacker_opt']['attack_type']
        return self.config['blackbox_name'].format(model_name, attack_type)

    def save_blackbox_adv(self, epoch):
        self.logger.info('------------- Save Blackbox adversarial samples at Epoch {} -----------'.format(epoch))
        
        # set model to evaluation
        self.model.eval()

        num_img = len(self.loader.dataset)
        c, w, h = self.loader.dataset[0][0].shape
        blackbox_dataset = torch.zeros(num_img, c, w, h)

        # run batch
        for i, (inputs, labels, indexes) in enumerate(self.loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # get adversarial example
            adv_inputs = self.attacker.get_adv_images(inputs, labels, random_start=self.rand_adv_init())
            self.model.eval()
            adv_inputs = adv_inputs.cpu()

            batch_size = adv_inputs.shape[0]
            for i in range(batch_size):
                idx = int(indexes[i])
                blackbox_dataset[idx] = adv_inputs[i]

        file_name = os.path.join(self.config['output_dir'], self.get_blackbox_name())
        self.logger.info('====== Save Blackbox Atttack Examples to {} ========'.format(file_name))    
        torch.save(blackbox_dataset, file_name)
        return 


    def run_targeted_val(self, epoch):
        assert ('targeted_on' in self.config['attacker_opt']) and self.config['attacker_opt']['targeted_on']
        assert (not self.accumulate_grad)
        self.logger.info('------------- Start Targeted Validation at Epoch {} -----------'.format(epoch))
        total_num = []
        total_rate = []
        
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

            # original prediction
            org_preds = self.model(inputs, loop=self.num_loop)
            if isinstance(org_preds, tuple):
                org_preds = org_preds[0]
            batch_size, num_class = org_preds.shape

            # trigger adversarial attack or not
            if self.adv_test and self.config['attacker_opt']['targeted_type'] == 'type1':
                # type1: untargeted attack
                adv_inputs = self.attacker.get_adv_images(inputs, labels, random_start=self.rand_adv_init())
                self.model.eval()
                final_inputs = adv_inputs

            elif self.adv_test and self.config['attacker_opt']['targeted_type'] == 'type2':
                # type2: random targeted
                dummy = torch.randn(batch_size, num_class).cuda()
                dummy[torch.arange(batch_size).cuda(), labels] = -100
                random_label = dummy.max(-1)[1]
                adv_inputs = self.attacker.get_adv_images(inputs, random_label, random_start=self.rand_adv_init(), targeted=True)
                self.model.eval()
                final_inputs = adv_inputs

            elif self.adv_test and self.config['attacker_opt']['targeted_type'] == 'type3':
                # type3: least likely targeted
                dummy = org_preds.detach().clone()
                dummy[torch.arange(batch_size).cuda(), labels] = 100
                least_label = dummy.min(-1)[1]
                adv_inputs = self.attacker.get_adv_images(inputs, least_label, random_start=self.rand_adv_init(), targeted=True)
                self.model.eval()
                final_inputs = adv_inputs

            elif self.adv_test and self.config['attacker_opt']['targeted_type'] == 'type4':
                # type4: most likely targeted
                dummy = org_preds.detach().clone()
                dummy[torch.arange(batch_size).cuda(), labels] = -100
                most_label = dummy.max(-1)[1]
                adv_inputs = self.attacker.get_adv_images(inputs, most_label, random_start=self.rand_adv_init(), targeted=True)
                self.model.eval()
                final_inputs = adv_inputs

            else:
                final_inputs = inputs

            # run model
            with torch.no_grad():
                if self.inst_on:
                    predictions = self.model(final_inputs, loop=self.num_loop)
                else:
                    predictions = self.model(final_inputs)

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            total_num.append(predictions.shape[0])
            
            # check success rate
            total_rate.append((predictions.max(1)[1] != labels).sum().item())
            #if self.config['attacker_opt']['targeted_type'] == 'type1':

            # save adversarial images
            if self.test_save and i < self.config['save_test']['save_length']:
                org_list.append(inputs.cpu())
                gt_list.append(labels.cpu())
                pred_list.append(predictions.max(1)[1].cpu())
                if self.adv_test:
                    adv_list.append(adv_inputs.cpu())

        succ_rate = sum(total_rate) / float(sum(total_num))
        self.logger.info('Epoch {:5d} Evaluation Complete ==> Total Success Rate : {:9.2f}, Number Samples : {:9d}'.format(epoch, succ_rate * 100, sum(total_num)))

        # set back to training mode again
        self.model.train()

        # save adversarial images
        if self.test_save:
            file_name = os.path.join(self.config['output_dir'], self.config['save_test']['file_name'])
            adv_output = {
                    'org_images' : torch.cat(org_list, 0),
                    'gt_labels'  : torch.cat(gt_list, 0),
                    'adv_images' : torch.cat(adv_list, 0) if self.adv_test else 0,
                    'pred_labels': torch.cat(pred_list, 0),
                    }
            torch.save(adv_output, file_name)
            self.logger.info('=====> Complete! Adversarial images have been saved to {}'.format(file_name))

        return succ_rate 


    def run_val(self, epoch):
        self.logger.info('------------- Start Validation at Epoch {} -----------'.format(epoch))
        total_num = []
        total_acc = []
        
        # set model to evaluation
        self.model.eval()

        # save test
        if self.test_save:
            org_list = []
            adv_list = []
            gt_list = []
            pred_list = []

        if self.config['blackbox_test']:
            file_name = os.path.join(self.config['output_dir'], self.config['blackbox_name'])
            self.logger.info('====== Load Blackbox Atttack Examples to {} ========'.format(file_name))    
            blackbox_data = torch.load(file_name)

        # run batch
        for i, (inputs, labels, indexes) in enumerate(self.loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            batch_size = inputs.shape[0]
            # print test time
            #start_time = time.time()
            # trigger adversarial attack or not
            if self.config['blackbox_test']:
                black_inputs = [blackbox_data[int(indexes[i])].unsqueeze(0) for i in range(batch_size)]
                final_inputs = torch.cat(black_inputs, dim=0).cuda()
            elif self.adv_test:
                adv_inputs = self.attacker.get_adv_images(inputs, labels, random_start=self.rand_adv_init())
                self.model.eval()
                final_inputs = adv_inputs
            else:
                final_inputs = inputs

            # run model
            with torch.no_grad():
                if self.inst_on:
                    if self.accumulate_grad:
                        pred_hist = []
                        for _ in range(self.num_loop):
                            crt_pred = self.model(final_inputs, loop=1)
                            crt_pred = crt_pred[0] if isinstance(crt_pred, tuple) else crt_pred
                            pred_hist.append(crt_pred)
                        predictions = sum(pred_hist) / len(pred_hist)
                    else:
                        predictions = self.model(final_inputs, loop=self.num_loop)
                else:
                    predictions = self.model(final_inputs)

            # print test time
            #print("--- {} seconds for the {} batch ---".format(time.time() - start_time, i))

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            total_num.append(predictions.shape[0])
            total_acc.append((predictions.max(1)[1] == labels).sum().item())

            # save adversarial images
            if self.test_save and i < self.config['save_test']['save_length']:
                org_list.append(inputs.cpu())
                gt_list.append(labels.cpu())
                pred_list.append(predictions.max(1)[1].cpu())
                if self.adv_test:
                    adv_list.append(adv_inputs.cpu())

        acc = sum(total_acc) / float(sum(total_num))
        self.logger.info('Epoch {:5d} Evaluation Complete ==> Total Accuracy : {:9.2f}, Number Samples : {:9d}'.format(epoch, acc * 100, sum(total_num)))

        # set back to training mode again
        self.model.train()

        # save adversarial images
        if self.test_save:
            file_name = os.path.join(self.config['output_dir'], self.config['save_test']['file_name'])
            adv_output = {
                    'org_images' : torch.cat(org_list, 0),
                    'gt_labels'  : torch.cat(gt_list, 0),
                    'adv_images' : torch.cat(adv_list, 0) if self.adv_test else 0,
                    'pred_labels': torch.cat(pred_list, 0),
                    }
            torch.save(adv_output, file_name)
            self.logger.info('=====> Complete! Adversarial images have been saved to {}'.format(file_name))

        return acc