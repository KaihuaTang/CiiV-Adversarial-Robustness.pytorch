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

from utils.train_utils import *
from utils.test_utils import *

# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str, help='Indicate the config file used for the training.')
parser.add_argument('--seed', default=25, type=int, help='Fix the random seed for reproduction. Default is 25.')
parser.add_argument('--phase', default='train', type=str, help='Indicate train/val/test/val_all phase.')
parser.add_argument('--output_dir', default=None, type=str, help='Output directory that saves everything.')
parser.add_argument('--load_dir', default=None, type=str, help='Load model from this directory for testing')
parser.add_argument('--require_eval', action='store_true', help='Require evaluation on val set during training.')
parser.add_argument('--logger_name', default='logger_eval', type=str, help='Name of TXT output for the logger.')
parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
# ============================================================================
# other important parameters
parser.add_argument('--train_type', default=None, type=str, help='Type of training strategy.')
parser.add_argument('--test_type', default=None, type=str, help='Type of test strategy.')
parser.add_argument('--adv_train', action='store_true', help='Adversarial Training.')
parser.add_argument('--adv_test', action='store_true', help='Adversarial Attacked Testing.')
parser.add_argument('--adv_type', default=None, type=str, help='Adversarial Attack Type.')
parser.add_argument('--adv_setting', default=None, type=str, help='Adversarial Attack Setting.')
parser.add_argument('--rand_aug', action='store_true', help='Use Random Augmentation.')
parser.add_argument('--target_type', default=None, type=str, help='The using of targeted attack and types. random / most (likely) / least (likely)')

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
logger.info('=====> Using fixed random seed: ' + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ============================================================================
# load config
logger.info('=====> Load config from yaml: ' + str(args.cfg))
with open(args.cfg) as f:
    config = yaml.load(f)

# load attacker config
logger.info('=====> Load attacker config from yaml: config/attacker_config.yaml')
with open('config/attacker_config.yaml') as f:
    attack_config = yaml.load(f)

# merge second for details attack settings
logger.info('=====> Merge arguments from command')
config = utils.update(config, args, logger)

# change attack type
dataset_name = config['dataset']['name']
attacker_type = config['attacker_opt']['attack_type']
attacker_set  = config['attacker_opt']['attack_set']
utils.update_attacker_info(config, attack_config, dataset_name, attacker_type, attacker_set)

# save config file
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
model.config = config
model = nn.DataParallel(model).cuda()

# ============================================================================
# training
if args.phase == 'train':
    training_func = get_train_func(config)
    # start training
    training = training_func(args, config, logger, model, eval=args.require_eval)
    training.run()
# normal test
else:
    checkpoint = Checkpoint(config)
    checkpoint.load(model, args.load_dir, logger)
    # start testing
    test_func = get_test_func(config)
    if args.phase == 'val':
        testing = test_func(args, config, logger, model, val=True)
        testing.run_val(epoch=-1)
    elif args.phase == 'test':
        testing = test_func(args, config, logger, model, val=False)
        testing.run_val(epoch=-1)
    else:
        raise ValueError('Wrong Phase')

logger.info('========================= Complete =========================')
