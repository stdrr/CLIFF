import argparse
from yacs.config import CfgNode as CN
from os.path import join
from config import *


# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 4
cfg.SEED_VALUE = -1
cfg.IMG_RES = 256

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.RESUME = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.IS_FINETUNE = False
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.WARMUP_STEPS = 1
cfg.TRAIN.LR_SCHEDULE = []
cfg.TRAIN.LR_DECAY = 1.0
cfg.TRAIN.MAX_STEP = 100000
cfg.TRAIN.CLIP_GRADIENT = False
cfg.TRAIN.CLIP_NORM = 1.0
cfg.TRAIN.LOSS_SCALE = 60
cfg.TRAIN.MASKED_PROB = 0.0
cfg.TRAIN.SCHEDULER = 'default'
cfg.TRAIN.WD = 0.01
cfg.TRAIN.OPT = 'AdamW'

cfg.DATASET = CN()
cfg.DATASET.TEST = 'agora_validation'

cfg.MODEL = CN()
cfg.MODEL.PRETRAINED = None
cfg.MODEL.REG_LAYER = 2
cfg.MODEL.PTYPE = 'marker'
cfg.MODEL.BACKBONE = 'hrnet_w32'
cfg.MODEL.CORR_LAYER = 0

cfg.LOSS = CN()



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    cfg.cfg_file = cfg_file
    return cfg.clone()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    print(args, end='\n\n')

    cfg_file = args.cfg
    if cfg_file is not None:
        cfg = update_cfg(cfg_file)
    else:
        cfg = get_cfg_defaults()

    return cfg
