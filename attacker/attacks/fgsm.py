##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class FGSM(Attacker):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.007)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, logger, config, eps=16.0, eot_iter=1):
        super(FGSM, self).__init__("FGSM", model, logger, config)
        self.eps = eps / 255.0
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker FGSM with eps: {}, eot: {}'.format(eps, eot_iter))

    def forward(self, images, labels, random_start=False, targeted=False):
        """
        Overridden.
        Note: FGSM doesn't contain random_start
        """
        if targeted:
            self.sign = -1
        else:
            self.sign = 1

        images, labels = images.cuda(), labels.cuda()

        org_images = images.detach()

        if random_start:
            adv_images = images.clone().detach() + torch.randn_like(images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            adv_images = images.clone().detach()
            
        adv_images.requires_grad = True

        # apply EOT to the attacker
        eot_grads = []
        # EOT is applied when eot_iter > 1
        for _ in range(self.eot_iter):
            if adv_images.grad:
                adv_images.grad.zero_()
            outputs = self.model(adv_images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            cost = self.sign * self.loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, create_graph=True)[0]
            eot_grads.append(grad.detach().clone())
        grad = sum(eot_grads) / self.eot_iter

        # adv image update, image is NOT normalized
        adv_images = org_images.detach() + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images.detach()
