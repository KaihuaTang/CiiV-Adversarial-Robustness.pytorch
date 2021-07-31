##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class EOT(Attacker):
    """
    EOT(Linf) attack in the paper 'Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples'
    [https://arxiv.org/abs/1802.00420]
        
    """
    def __init__(self, model, logger, config, eps=16.0, learning_rate=0.1, steps=100, eot_iter=30):
        super(EOT, self).__init__("EOT", model, logger, config)
        self.eps = eps / 255.0
        self.learning_rate = learning_rate
        self.steps = steps
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker EOT with eps: {}, learning_rate: {}, steps: {}, eot: {}'.format(eps, learning_rate, steps, eot_iter))

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

        for _ in range(self.steps):
            adv_images = adv_images.detach()
            adv_images.requires_grad = True

            # apply EOT to the attacker
            eot_grads = []
            # EOT is applied when eot_iter > 1
            for _ in range(self.eot_iter):
                outputs = self.model(adv_images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                cost = self.sign * self.loss(outputs, labels)
                grad = torch.autograd.grad(cost, adv_images, allow_unused=True,
                                       retain_graph=False, create_graph=False)[0]
                eot_grads.append(grad.detach().clone())
            grad = sum(eot_grads)

            # adv image update, image is NOT normalized
            adv_images = self.adv_image_update(adv_images, org_images, grad)

        return adv_images

    def adv_image_update(self, adv_images, org_images, grad):
        # for the original EOT implemented in https://github.com/anishathalye/obfuscated-gradients/blob/72e24bc4f4669c01a4f2987a2c4a01d2ba363687/randomization/robustml_attack.py
        # instead of grad.sign(), the grad is directly used
        adv_images = adv_images.detach() + self.learning_rate * grad
        delta = torch.clamp(adv_images - org_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(org_images + delta, min=0, max=1)
        return adv_images

