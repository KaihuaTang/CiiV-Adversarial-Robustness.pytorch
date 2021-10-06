import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_Model(nn.Module):
    """
    base model used for adversarial attack and defense
    """
    def __init__(self):
        super(Base_Model, self).__init__()
        # attacking mode, i.e., generating attacking images
        self.attacking = False
    
    def set_attack(self):
        self.attacking = True
        # recursive set all modules to attack
        for m in self.modules():
            if isinstance(m, Base_Model) and (not m.is_attack()):
                m.set_attack()

    def set_unattack(self):
        self.attacking = False
        # recursive set all modules to unattack
        for m in self.modules():
            if isinstance(m, Base_Model) and m.is_attack():
                m.set_unattack()

    def is_attack(self):
        return self.attacking

    

