##############################################################################
#   Modified from https://github.com/BorealisAI/advertorch/
##############################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..attacker import Attacker
import utils.general_utils as utils

class SPSA(Attacker):
    r"""
    SPSA in the paper 'Adversarial Risk and the Dangers of Evaluating Against Weak Attacks'
    [https://arxiv.org/abs/1802.05666]
    """
    def __init__(self, model, logger, config, delta=0.1, eps=16, batch_size=128, steps=5, lr=0.01):
        super(SPSA, self).__init__("SPSA", model, logger, config)
        self.batch_size = batch_size
        self.delta = delta
        self.eps = eps / 255.0
        self.steps = steps
        self.lr = lr

        logger.info('Create Attacker SPSA with delta: {}, eps: {}, steps: {}, lr: {}, batch_size: {}'.format(delta, eps, steps, lr, batch_size))

    def forward(self, images, labels, random_start=False, targeted=False):
            
        images, labels = images.cuda(), labels.cuda()

        dx = torch.zeros_like(images)
        dx.grad = torch.zeros_like(dx)
        optimizer = optim.Adam([dx], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(images + dx, labels)
            optimizer.step()
            adv_images = torch.clamp(images + dx , min=0, max=1)
            dx = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = images + dx

        return adv_images

    def f(self, x, y):
        pred = self.model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        return F.cross_entropy(pred, y)

    def spsa_grad(self, x, y):
        b, c, w, h = x.shape
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = x.repeat(self.batch_size, 1, 1, 1, 1).contiguous().view(self.batch_size*b,c,w,h)
        y = y.repeat(self.batch_size, 1).contiguous().view(-1)

        v = torch.zeros_like(x).bernoulli_().mul_(2.0).sub_(1.0)
        df = self.f(x + self.delta * v, y) - self.f(x - self.delta * v, y)
        grad = df / (2. * self.delta * v)
        grad = grad.view(self.batch_size,b,c,w,h).mean(dim=0)
        return grad