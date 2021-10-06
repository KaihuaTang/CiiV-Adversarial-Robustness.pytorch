##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attacker import Attacker
import utils.general_utils as utils

class ADAPTIVE(Attacker):
    def __init__(self, model, logger, config, eps=16.0, alpha=1.0, steps=30, weight=10.0, eot_iter=1):
        super(ADAPTIVE, self).__init__("ADAPTIVE", model, logger, config)
        self.eps = eps / 255.0
        self.alpha = alpha / 255.0
        self.weight = weight
        self.steps = steps
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker ADAPTIVE with eps: {}, alpha: {}, steps: {}, eot: {}, weight: {}'.format(eps, alpha, steps, eot_iter, weight))

    def forward(self, images, labels, random_start=False, targeted=False):
        """
        Overridden.
        """
        if targeted:
            self.sign = -1
        else:
            self.sign = 1
            
        images, labels = images.cuda(), labels.cuda()

        org_images = images.detach()
        adv_images = images.clone().detach()
        
        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for _ in range(self.steps):
            adv_images = adv_images.detach()
            adv_images.requires_grad = True

            # apply EOT to the attacker
            eot_grads = []
            # EOT is applied when eot_iter > 1
            for _ in range(self.eot_iter):
                if adv_images.grad:
                    adv_images.grad.zero_()
                all_ces = []
                all_regs = []
                preds, other_feats, other_logits = self.model(adv_images, loop=5)
                for i, logits in enumerate(other_logits):
                    ce_loss = F.cross_entropy(logits, labels)
                    all_ces.append(ce_loss)
                mean_feat = sum(other_feats) / len(other_feats)
                for i in range(len(other_feats)):
                    reg_loss = self.smooth_l1_loss(other_feats[i], mean_feat)
                    all_regs.append(reg_loss)
                loss = sum(all_ces) + self.weight * sum(all_regs)
                cost = self.sign * loss
                grad = torch.autograd.grad(cost, adv_images, create_graph=True)[0]
                eot_grads.append(grad.detach().clone())
            grad = sum(eot_grads) / self.eot_iter

            # adv image update, image is NOT normalized
            adv_images = self.adv_image_update(adv_images, org_images, grad)

        return adv_images

    def smooth_l1_loss(self, x, y):
        diff = F.smooth_l1_loss(x, y, reduction='none')
        diff = diff.sum(1)
        diff = diff.mean(0)
        return diff

    def adv_image_update(self, adv_images, org_images, grad):
        # image is NOT normalized
        adv_images = adv_images.detach() + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - org_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(org_images + delta, min=0, max=1)
        return adv_images

