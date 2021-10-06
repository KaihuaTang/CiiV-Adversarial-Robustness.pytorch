##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

from ..attacker import Attacker
import utils.general_utils as utils

from autoattack import AutoAttack

class AA(Attacker):
    """
    Auto-Attack
    installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack
    """
    def __init__(self, model, logger, config, eps=16.0, norm='Linf'):
        super(AA, self).__init__("AA", model, logger, config)
        self.eps = eps / 255.0
        self.attacker = AutoAttack(model, norm=norm, eps=self.eps, version='standard')

        logger.info('Create Attacker Auto-Attack with eps: {}'.format(eps))

    def forward(self, images, labels, random_start=False, targeted=False):
        
        images, labels = images.cuda(), labels.cuda()
        batch_size = images.shape[0]

        adv_images = self.attacker.run_standard_evaluation(images, labels, bs=batch_size)

        return adv_images