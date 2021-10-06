##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class PGD(Attacker):
    """
    PGD(Linf) attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

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
        >>> attack = torchattacks.PGD(model, eps = 4, alpha = 1, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, logger, config, eps=16.0, alpha=1.0, steps=30, eot_iter=1):
        super(PGD, self).__init__("PGD", model, logger, config)
        self.eps = eps / 255.0
        self.alpha = alpha / 255.0
        self.steps = steps
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker PGD with eps: {}, alpha: {}, steps: {}, eot: {}'.format(eps, alpha, steps, eot_iter))


    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

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
        delta = torch.zeros_like(org_images).cuda()

        if random_start:
            # Starting at a uniformly random point
            delta.uniform_(-self.eps, self.eps)
            delta = self.clamp(delta, 0.0-org_images, 1.0-org_images)

        for _ in range(self.steps):
            delta = delta.detach()
            delta.requires_grad = True

            # apply EOT to the attacker
            eot_grads = []
            # EOT is applied when eot_iter > 1
            for _ in range(self.eot_iter):
                if delta.grad:
                    delta.grad.zero_()
                outputs = self.model(org_images + delta)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                cost = self.sign * self.loss(outputs, labels)
                grad = torch.autograd.grad(cost, delta, create_graph=True)[0]
                #cost.backward()
                #grad = delta.grad.detach()
                eot_grads.append(grad.detach().clone())
            grad = sum(eot_grads) / self.eot_iter

            # adv image update, image is NOT normalized
            delta = self.adv_image_update(delta, org_images, grad)

        return (org_images + delta).detach()


    def adv_image_update(self, delta, org_images, grad):
        # image is NOT normalized
        delta = torch.clamp(delta + self.alpha * grad.sign(), min=-self.eps, max=self.eps)
        delta = self.clamp(delta, 0.0-org_images, 1.0-org_images)
        return delta

