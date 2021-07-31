##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class GN(Attacker):
    r"""
    Add Gaussian Noise.
    Arguments:
        model (nn.Module): model to attack.
        sigma (nn.Module): sigma (DEFAULT: 0.1).
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.GN(model)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, logger, config, sigma=0.1, eps=16):
        super(GN, self).__init__("GN", model, logger, config)
        self.sigma = sigma
        self.eps = eps / 255.0
        self.loss = nn.CrossEntropyLoss()
        
        logger.info('Create Attacker Gaussian Noise with sigma: {}, eps: {}'.format(sigma, eps))

    def forward(self, images, labels, random_start=False, targeted=False):
        """
        Overridden.
        """
        images = images.cuda()
        adv_images = images + torch.clamp(self.sigma * torch.randn_like(images), min=-self.eps, max=self.eps).detach()    
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
