import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint,\
    NativeScalerWithGradNormCount,\
    auto_resume_helper, is_main_process,\
    get_git_info, run_cmd


from yacs.config import CfgNode as CN
from contextlib import redirect_stdout

def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', required=False,
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    parser.add_argument('--use-sync-bn', action='store_true',
                        default=False, help='sync bn')
    parser.add_argument('--use-wandb', action='store_true',
                        default=False, help='use wandb to record log')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    
    # ea_searching_results
    parser.add_argument("--ea-result-path", type=str, help='path to the evolutionary search results (.tar)')
    args = parser.parse_args()

    config = get_config(args)

    return args, config

def get_config_with_highest_accuracy_from_ea_result(path):
    def parse_ckpt(m):
        X1 = []
        Y1 = []
        Z1 = []
        for k, v in m['vis_dict'].items():
            if 'params' not in v:
                continue
            if 'acc' not in v:
                continue
            X1.append(v['params'])
            Y1.append(v['acc'])
            Z1.append(k)
        
        return np.array(X1), np.array(Y1), np.array(Z1)
    
    ckpt = torch.load(path)
    flops, accs, cfgs = parse_ckpt(ckpt)
    max_acc_config_idx = np.argmax(accs)
    return flops[max_acc_config_idx], accs[max_acc_config_idx], list(cfgs[max_acc_config_idx])

def get_model_type_name(config):
    if 'deit' in config.MODEL.TYPE:
        return "deit"
    elif 'swin' in config.MODEL.TYPE:
        return "swin"
    else:
        raise NotImplementedError("Model type mismatch")

def get_subnet_config(subnet_rank_config, config):
    model_type_name = get_model_type_name(config)
    if model_type_name == 'swin':
        return [[[float(subnet_rank_config[i*4+j]) for j in range(4)] for i in range(depth)] for depth in config.MODEL.SWIN.DEPTH]
    elif model_type_name == 'deit':
        return [[float(subnet_rank_config[i*3+j]) for j in range(3)] for i in range(config.MODEL.DEIT.DEPTH)]
    else:
        raise NotImplementedError("Model type mismatch")


def dump_to_yaml_file(config, model_type_name):
    new_config = CN()
    new_config.MODEL = CN()
    new_config.MODEL.TYPE = config.MODEL.TYPE
    new_config.MODEL.NAME = config.MODEL.NAME
    if model_type_name == 'swin':
        new_config.MODEL.SWIN = config.MODEL.SWIN.clone()
    elif model_type_name == 'deit':
        new_config.MODEL.DEIT = config.MODEL.DEIT.clone()
    print(new_config)
    
    with open(f'{config.MODEL.NAME}.yml', 'w') as f:
        with redirect_stdout(f): print(new_config)


def main(args, config):    
    flops, acc, svd_config = get_config_with_highest_accuracy_from_ea_result(args.ea_result_path)
    svd_config = get_subnet_config(svd_config, config)    
    model_type_name = get_model_type_name(config)
    config.defrost()
    config.MODEL.TYPE = f"lr_{model_type_name}_subnet"
    config.MODEL.NAME = f"lr_{model_type_name}_subnet_{acc:.2f}_{flops:.2f}G"
    if model_type_name == "swin":
        config.MODEL.SWIN.SVD_CONFIG = svd_config
    elif model_type_name == "deit":
        config.MODEL.DEIT.SVD_CONFIG = svd_config
    config.freeze()
    dump_to_yaml_file(config, model_type_name)
    

if __name__ == '__main__':
    args, config = parse_option()
    main(args, config)
