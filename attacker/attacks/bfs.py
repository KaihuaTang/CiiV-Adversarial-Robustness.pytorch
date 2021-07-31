##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

class BFS(Attacker):
    '''
    Brute-Fore Search
    '''
    def __init__(self, model, logger, config, sigma=0.1, eps=16, steps=100, eot_iter=100):
        super(BFS, self).__init__("BFS", model, logger, config)
        self.sigma = sigma
        self.eps = eps / 255.0
        self.steps = steps
        self.loss = nn.CrossEntropyLoss()

        logger.info('Create Attacker BFS with sigma: {}, eps: {}, steps: {}'.format(sigma, eps, steps))

    def forward(self, images, labels, random_start=False, targeted=False):
        """
        Overridden.
        Note: BFS doesn't contain random_start
        """

        images, labels = images.cuda(), labels.cuda()

        final_images = images.clone()

        # EOT is applied when eot_iter > 1
        for _ in range(self.steps):
            noise = torch.clamp(self.sigma * torch.randn_like(images), min=-self.eps, max=self.eps).detach()  
            adv_images = torch.clamp(images + noise, min=0, max=1).detach()

            outputs = self.model(adv_images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            success = (labels != outputs.max(-1)[-1]).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # update successed attack
            final_images = success * adv_images + (1.0 - success) * final_images

        return final_images
