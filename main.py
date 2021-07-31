import json
import yaml
import os
import argparse
import torch
import torch.nn as nn
import random
import utils.general_utils as utils
from utils.logger_utils import custom_logger
from data.dataloader import get_loader
from utils.checkpoint_utils import Checkpoint

from train import train_net
from test import test_net

def update_attacker_info(config, attack_config, dataset_name, attacker_type, attacker_set):
    print('==================== Attacker {} ================='.format(attacker_type))
    config['attacker_opt']['attack_type'] = attacker_type
    config['attacker_opt']['attack_set'] = attacker_set
    config['attacker_opt'].update(attack_config[dataset_name][attacker_type][attacker_set])

# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str, help='Indicate the config file used for the training.')
parser.add_argument('--seed', default=25, type=int, help='Fix the random seed for reproduction. Default is 25.')
parser.add_argument('--phase', default='train', type=str, help='Indicate train/val/test/val_all phase.')
parser.add_argument('--dataset', default='imagenet', type=str, help='Name of Dataset used, e.g., imagenet.')
parser.add_argument('--batch_size', default=None, type=int, help='Batch Size.')
parser.add_argument('--output_dir', default=None, type=str, help='Output directory that saves everything.')
parser.add_argument('--load_dir', default=None, type=str, help='Load model from this directory for testing')
parser.add_argument('--require_eval', action='store_true', help='Require evaluation on val set during training.')
parser.add_argument('--logger_name', default='logger_eval', type=str, help='Name of TXT output for the logger.')
parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
args = parser.parse_args()

# ============================================================================
# init logger
if args.output_dir is None:
    print('Please specify output directory')
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
if args.phase != 'train':
    logger = custom_logger(args.output_dir, name='{}.txt'.format(args.logger_name))
else:
    logger = custom_logger(args.output_dir)
logger.info('========================= Start Main =========================')

# ============================================================================
# fix random seed
if args.seed:
    logger.info('=====> Using fixed random seed: ' + str(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# ============================================================================
# load config
logger.info('=====> Load config from yaml: ' + str(args.cfg))
with open(args.cfg) as f:
    config = yaml.load(f)

# merge first for attack type
logger.info('=====> Merge arguments from command')
config = utils.update(config, args, logger)

logger.info('=====> Load attacker config from yaml: config/attacker_config.yaml')
with open('config/attacker_config.yaml') as f:
    attack_config = yaml.load(f)
dataset_name = config['dataset']['name']
attacker_type = config['attacker_opt']['attack_type']
attacker_set  = config['attacker_opt']['attack_set']
update_attacker_info(config, attack_config, dataset_name, attacker_type, attacker_set)

# merge second for details attack settings
logger.info('=====> Merge arguments from command')
config = utils.update(config, args, logger)

logger.info('=====> Save config as config.json')
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(config, f)
utils.print_config(config, logger)

# ============================================================================
# create model
logger.info('=====> Model construction from: ' + str(config['networks']['def_file']))
model_file = config['networks']['def_file']
model_args = config['networks']['params']
model = utils.source_import(model_file).create_model(**model_args)

# ============================================================================
# data parallel
model = nn.DataParallel(model).cuda()

# ============================================================================
# save blackbox transferred data
if config['blackbox_save']:
    checkpoint = Checkpoint(config)
    checkpoint.load(model, args.load_dir, logger)
    config['attacker_opt']['adv_val'] = True
    testing = test_net(args, config, logger, model, blackbox_save=True)
    testing.save_blackbox_adv(epoch=-1)

# ============================================================================
# training
elif args.phase == 'train':
    training = train_net(args, config, logger, model, eval=args.require_eval)
    training.run()

# ============================================================================
# test all attacker
elif args.phase == 'val_all':
    checkpoint = Checkpoint(config)
    checkpoint.load(model, args.load_dir, logger)
    
    # clean val
    config['attacker_opt']['adv_val'] = False
    testing = test_net(args, config, logger, model, val=False)
    testing.run_val(epoch=-1)

    # val on all attackers
    config['attacker_opt']['adv_val'] = True
    
    #update_attacker_info(config, attack_config, dataset_name, 'GN', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'BFS', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'SPSA', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)
    
    #update_attacker_info(config, attack_config, dataset_name, 'ADAPTIVE', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    update_attacker_info(config, attack_config, dataset_name, 'FGSM', 'setting1')
    testing = test_net(args, config, logger, model, val=False)
    testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'FGSM', 'setting2')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    update_attacker_info(config, attack_config, dataset_name, 'PGD', 'setting1')
    testing = test_net(args, config, logger, model, val=False)
    testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'PGD', 'setting2')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    update_attacker_info(config, attack_config, dataset_name, 'PGDL2', 'setting1')
    testing = test_net(args, config, logger, model, val=False)
    testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'PGDL2', 'setting2')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'GN', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'UN', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'BPDAPGD', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

    update_attacker_info(config, attack_config, dataset_name, 'CW', 'setting1')
    testing = test_net(args, config, logger, model, val=False)
    testing.run_val(epoch=-1)

    #update_attacker_info(config, attack_config, dataset_name, 'EOT', 'setting1')
    #testing = test_net(args, config, logger, model, val=False)
    #testing.run_val(epoch=-1)

# normal test
else:
    checkpoint = Checkpoint(config)
    checkpoint.load(model, args.load_dir, logger)
    if args.phase == 'val':
        testing = test_net(args, config, logger, model, val=True)
        if ('targeted_on' in config['attacker_opt']) and config['attacker_opt']['targeted_on']:
            testing.run_targeted_val(epoch=-1)
        else:
            testing.run_val(epoch=-1)
    else:
        testing = test_net(args, config, logger, model, val=False)
        assert args.phase == 'test'
        if ('targeted_on' in config['attacker_opt']) and config['attacker_opt']['targeted_on']:
            testing.run_targeted_val(epoch=-1)
        else:
            testing.run_val(epoch=-1)

logger.info('========================= Complete =========================')
