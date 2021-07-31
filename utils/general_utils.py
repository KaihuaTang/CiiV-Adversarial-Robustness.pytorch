import torch
import importlib

def get_value(old_value, new_value):
    if new_value is not None:
        return new_value
    else:
        return old_value

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def int_to_others(val):
    if val == 'true' or val == 'True':
        return True
    elif val == 'false' or val == 'False':
        return False
    elif RepresentsInt(val):
        return int(val)
    elif RepresentsFloat(val):
        return float(val)
    else:
        return val

def update_config_key(update_dict, key, new_val):
    names = key.split('.')
    while(len(names) > 1):
        item_key = names.pop(0)
        update_dict = update_dict[item_key]
    old_val = update_dict[names[-1]]
    update_dict[names[-1]] = int_to_others(new_val)
    return old_val

def update(config, args, logger):
    config['output_dir'] = get_value(config['output_dir'], args.output_dir)
    logger.info('output_dir is set to : ' + str(config['output_dir']))
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)
    logger.info('batch_size is set to : ' + str(config['training_opt']['batch_size']))
    # update config from command
    if len(args.opts) > 0 and (len(args.opts) % 2) == 0:
        for i in range(len(args.opts)//2):
            key = args.opts[2*i]
            val = args.opts[2*i+1]
            old_val = update_config_key(config, key, val)
            logger.info('=====> {}: {} (old key) => {} (new key)'.format(key, old_val, val))
    return config

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_grad(named_parameters):
    """ show grads """
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()
    total_norm = total_norm ** (1. / 2)
    print('---Total norm {:.3f} -----------------'.format(total_norm))
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
    print('-------------------------------', flush=True)
    return total_norm

def print_config(config, logger, head=''):
    for key, val in config.items():
        if isinstance(val, dict):
            logger.info(head + str(key))
            print_config(val, logger, head=head + '   ')
        else:
            logger.info(head + '{} : {}'.format(str(key), str(val)))

class TriggerAction():
    def __init__(self, name):
        self.name = name
        self.action = {}

    def add_action(self, name, func):
        assert str(name) not in self.action
        self.action[str(name)] = func

    def remove_action(self, name):
        assert str(name) in self.action
        del self.action[str(name)]
        assert str(name) not in self.action

    def run_all(self, logger=None):
        for key, func in self.action.items():
            if logger:
                logger.info('trigger {}'.format(key))
            func()