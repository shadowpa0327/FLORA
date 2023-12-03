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
    parser.add_argument('--pretrained', required=True,
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

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    supernet = torch.load(config.MODEL.PRETRAINED)['model']
    subnet = build_model(config).state_dict()

    for k in supernet.keys():
        if k in subnet:
            if supernet[k].shape == subnet[k].shape:
                subnet[k] = supernet[k]
            else:
                shape = subnet[k].shape
                subnet[k] = supernet[k][:shape[0], :shape[1]]
        else:
            if k.endswith("VT.weight"):
                k_supernet_U_weight = k[:-9] + "U.weight"
                k_supernet_U_bias = k[:-9] + "U.bias"
                k_subnet_weight = k[:-9] + "fc.weight"
                k_subnet_bias = k[:-9] + "fc.bias"
                subnet[k_subnet_weight] = supernet[k_supernet_U_weight] @ \
                    supernet[k]
                subnet[k_subnet_bias] = supernet[k_supernet_U_bias]        

    torch.save({'model': subnet}, f"{config['MODEL']['NAME']}.pth")


if __name__ == '__main__':
    args, config = parse_option()

    main(config)
