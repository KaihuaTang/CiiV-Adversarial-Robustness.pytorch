##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class FFGSM(Attacker):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]
    
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 8/255)
        alpha (float): step size. (DEFALUT: 10/255)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, logger, config, eps=8.0, alpha=10.0, eot_iter=1):
        super(FFGSM, self).__init__("FFGSM", model, logger, config)
        self.eps = eps / 255.0
        self.alpha = alpha / 255.0
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker FFGSM with eps: {}, alpha: {}, eot: {}'.format(eps, alpha, eot_iter))

    def forward(self, images, labels, random_start=False, targeted=False):
        """
        Overridden.
        Note: FFGSM always apply random_start
        """
        if targeted:
            self.sign = -1
        else:
            self.sign = 1

        images, labels = images.cuda(), labels.cuda()

        org_images = images.detach()
        adv_images = images.clone().detach() + torch.randn_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
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
        adv_images = adv_images.detach() + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - org_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(org_images + delta, min=0, max=1).detach()
        return adv_images.detach()
