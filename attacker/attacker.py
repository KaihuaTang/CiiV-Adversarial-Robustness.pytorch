##############################################################################
#   Modified from https://github.com/Harry24k/adversarial-attacks-pytorch
##############################################################################

import torch
import torch.nn as nn

class Attacker():
    """
    Base class for all attackers.
    note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's `training mode` to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model, logger, config):
        self.attacker = name
        self.model = model
        self.logger = logger
        self.config = config
        

    def forward(self, *input):
        """
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    
    def get_adv_images(self, images, labels, random_start=False, targeted=False):
        adv_images = self.attack(images, labels, random_start, targeted)
        return adv_images.detach()


    def __str__(self):
        # Whole structure of the model will be NOT displayed for print pretty.
        return '(Attacker Model : {})'.format(self.attacker)

    def attack(self, *input, **kwargs):
        # eval model before attack
        self.model.eval()
        # set model to attack mode
        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_attack()
        else:
            self.model.set_attack()
        
        # generate attack images
        images = self.forward(*input, **kwargs)

        # set back to the training phase
        self.model.train()
        # set model to unattack mode
        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_unattack()
        else:
            self.model.set_unattack()

        return images
