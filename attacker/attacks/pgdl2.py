##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class PGDL2(Attacker):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 1.0)
        alpha (float): step size. (DEFALUT: 0.2)
        steps (int): number of steps. (DEFALUT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, logger, config, eps=16.0, alpha=1.0, steps=30, eot_iter=1, eps_for_division=1e-10):
        super(PGDL2, self).__init__("PGDL2", model, logger, config)
        self.eps = eps / 255.0
        self.alpha = alpha / 255.0
        self.steps = steps
        self.eot_iter = eot_iter
        self.loss = nn.CrossEntropyLoss()
        self.eps_for_division = eps_for_division

        logger.info('Create Attacker PGD-L2 with eps: {}, alpha: {}, steps: {}, eot: {}'.format(eps, alpha, steps, eot_iter))

    def forward(self, images, labels, random_start=False, targeted=False):
        """
        Overridden.
        """
        if targeted:
            self.sign = -1
        else:
            self.sign = 1
            
        images, labels = images.cuda(), labels.cuda()
        batch_size = images.shape[0]

        org_images = images.detach()
        adv_images = images.clone().detach()
        
        if random_start:
            # Starting at a uniformly random point
            delta = torch.zeros_like(adv_images).to(adv_images.device)
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta = delta / n * r * self.eps
            adv_images = adv_images + delta
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
                outputs = self.model(adv_images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                cost = self.sign * self.loss(outputs, labels)
                grad = torch.autograd.grad(cost, adv_images, create_graph=True)[0]
                eot_grads.append(grad.detach().clone())
            grad = sum(eot_grads) / self.eot_iter

            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)

            # adv image update, image is NOT normalized
            adv_images = self.adv_image_update(adv_images, org_images, grad, batch_size)

        return adv_images

    def adv_image_update(self, adv_images, org_images, grad, batch_size):
        delta = adv_images - org_images
        g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
        scaled_g = grad/(g_norm + 1e-10)
        delta = (delta + scaled_g*self.alpha).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=self.eps).view_as(delta)
        adv_images = torch.clamp(org_images + delta, min=0, max=1).detach()
        return adv_images.detach()

