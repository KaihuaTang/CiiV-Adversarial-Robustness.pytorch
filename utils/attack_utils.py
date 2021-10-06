import torch
import importlib
import random
import attacker

def rand_adv_init(config):
    if random.uniform(0, 1) < config['attacker_opt']['attack_rand_ini']:
        return True  
    else:
        return False


def create_adversarial_attacker(config, model, logger):
    if config['attacker_opt']['attack_type'] == 'PGD':
        return attacker.PGD(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                alpha=config['attacker_opt']['attack_alpha'], 
                                                steps=config['attacker_opt']['attack_step'],
                                                eot_iter=config['attacker_opt']['eot_iter'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'PGDL2':
        return attacker.PGDL2(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                alpha=config['attacker_opt']['attack_alpha'], 
                                                steps=config['attacker_opt']['attack_step'],
                                                eot_iter=config['attacker_opt']['eot_iter'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'AutoAttack':
        return attacker.AA(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                norm=config['attacker_opt']['attack_norm'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'FGSM':
        return attacker.FGSM(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                eot_iter=config['attacker_opt']['eot_iter'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'FFGSM':
        return attacker.FFGSM(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                alpha=config['attacker_opt']['attack_alpha'], 
                                                eot_iter=config['attacker_opt']['eot_iter'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'GN':
        return attacker.GN(model, logger, config, sigma=config['attacker_opt']['gn_sigma'],
                                                eps=config['attacker_opt']['attack_eps'], 
                                            )
    elif config['attacker_opt']['attack_type'] == 'UN':
        return attacker.UN(model, logger, config, sigma=config['attacker_opt']['un_sigma'],
                                                eps=config['attacker_opt']['attack_eps'], 
                                            )
    elif config['attacker_opt']['attack_type'] == 'BPDAPGD':
        return attacker.BPDAPGD(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                alpha=config['attacker_opt']['attack_alpha'], 
                                                steps=config['attacker_opt']['attack_step'],
                                                eot_iter=config['attacker_opt']['eot_iter'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'EOT':
        return attacker.EOT(model, logger, config, eps=config['attacker_opt']['attack_eps'], 
                                                learning_rate=config['attacker_opt']['attack_lr'], 
                                                steps=config['attacker_opt']['attack_step'], 
                                                eot_iter=config['attacker_opt']['eot_iter'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'CW':
        return attacker.CW(model, logger, config, c=config['attacker_opt']['c'],
                                                kappa=config['attacker_opt']['kappa'],
                                                steps=config['attacker_opt']['steps'],
                                                lr=config['attacker_opt']['lr'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'BFS':
        return attacker.BFS(model, logger, config, sigma=config['attacker_opt']['sigma'],
                                                eps=config['attacker_opt']['attack_eps'],
                                                steps=config['attacker_opt']['steps'],
                                            )
    elif config['attacker_opt']['attack_type'] == 'SPSA':
        return attacker.SPSA(model, logger, config, eps=config['attacker_opt']['attack_eps'],
                                                delta=config['attacker_opt']['delta'],
                                                batch_size=config['attacker_opt']['batch_size'],
                                                steps=config['attacker_opt']['steps'],
                                                lr=config['attacker_opt']['lr'],
                                            )                          
    else:
        logger.raise_error('Wrong Attacker Type')



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
            adv_target[i] = (int(adv_target[i]) + random.randint(1, num_class-1)) % num_class
    return adv_target