##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class BPDAPGD(Attacker):
    """
    PGD(Linf) attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    BPDA attack in the paper 'Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples'
    [https://arxiv.org/abs/1802.00420]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 16(max 255)) 
        alpha (float): step size. (DEFALUT : 1(max 255))
        steps (int): number of steps. (DEFALUT : 30)
        random_start (bool): using random initialization of delta. (DEFAULT : False)
        targeted (bool): using targeted attack with input labels as targeted labels. (DEFAULT : False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.BPDAPGD(model, eps = 4, alpha = 1, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, logger, config, eps=16.0, alpha=1.0, steps=30, eot_iter=1):
        super(BPDAPGD, self).__init__("BPDAPGD", model, logger, config)
        self.eps = eps / 255.0
        self.alpha = alpha / 255.0
        self.steps = steps
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker BPDAPGD with eps: {}, alpha: {}, steps: {}, eot: {}'.format(eps, alpha, steps, eot_iter))

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
                outputs = self.model(adv_images, bpda=True)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                cost = self.sign * self.loss(outputs, labels)
                grad = torch.autograd.grad(cost, adv_images, allow_unused=True,
                                       retain_graph=False, create_graph=False)[0]
                eot_grads.append(grad.detach().clone())
            grad = sum(eot_grads) / self.eot_iter

            # adv image update, image is NOT normalized
            adv_images = self.adv_image_update(adv_images, org_images, grad)

        return adv_images

    def adv_image_update(self, adv_images, org_images, grad):
        # image is NOT normalized
        adv_images = adv_images.detach() + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - org_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(org_images + delta, min=0, max=1)
        return adv_images

