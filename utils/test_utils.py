import torch
import torch.nn as nn

import random
import attacker

from test_baseline import test_baseline
from test_ciiv import test_ciiv

def get_test_func(config):
    # choosing test strategy
    if config['strategy']['test_type'] == 'baseline':
        test_func = test_baseline
    elif config['strategy']['test_type'] == 'ciiv':
        test_func = test_ciiv
    else:
        raise ValueError('Wrong Test Strategy')
    return test_func



def get_adv_target(target_type, model, inputs, gt_label):
    with torch.no_grad():
        preds = model(inputs).softmax(-1)
    num_batch, num_class = preds.shape
    
    if target_type == 'random':
        adv_targets = torch.randint(0, num_class, (num_batch,)).to(gt_label.device)
        # validation check
        adv_targets = adv_target_update(gt_label, adv_targets, num_batch, num_class)
    elif target_type == 'most':
        idxs = torch.arange(num_batch).to(inputs.device)
        preds[idxs, gt_label] = -1
        adv_targets = preds.max(-1)[1]
    elif target_type == 'least':
        idxs = torch.arange(num_batch).to(inputs.device)
        preds[idxs, gt_label] = 100.0
        adv_targets = preds.min(-1)[1]
    else:
        raise ValueError('Wrong Targeted Attack Type')

    assert (adv_targets == gt_label).long().sum().item() == 0
    return adv_targets

def adv_target_update(gt_label, adv_target, num_batch, num_class):
    for i in range(num_batch):
        if int(gt_label[i]) == int(adv_target[i]):
            adv_target[i] = (int(adv_target[i]) + random.randint(0, num_class-1)) % num_class
    return adv_target