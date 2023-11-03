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
    random.seed(0)
    np.random.seed(0)
    weight = torch.load(config.MODEL.PRETRAINED, 'cpu')
    if 'model' in weight.keys():
        weight = weight['model']
    supernet = build_model(config).cpu()

    for k in list(weight.keys()):
        if 'qkv.weight' in k:
            m = k[:-7]
            block_id = int(m.split('.')[1])
            qkv = weight[f'{m}.weight']
            num_components = min(qkv.shape)
            VT, S, U = np.linalg.svd((qkv.T).numpy())
            VTS = torch.tensor(VT)@torch.diag(torch.tensor(S))
            
            if len(config.MODEL.CHOICES_BLOCK_CONFIG) == 0:
                weight[f'{m}.VT.weight'] = VTS[:, :num_components].T
                weight[f'{m}.U.weight'] = torch.tensor(U[:num_components, :].T)
                weight[f'{m}.U.bias'] = weight[f'{m}.bias']
            else:
                for i, ratio in enumerate(config.MODEL.CHOICES_BLOCK_CONFIG[block_id][0]):
                    weight[f'{m}.VT.{i}.weight'] = VTS[:, :int(round(num_components*ratio))].T
                    weight[f'{m}.U.{i}.weight'] = torch.tensor(U[:int(round(num_components*ratio)), :].T)
                    weight[f'{m}.U.{i}.bias'] = weight[f'{m}.bias']
                
        if 'fc1.weight' in k:
            m = k[:-7]
            block_id = int(m.split('.')[1])
            fc1 = weight[f'{m}.weight']
            num_components = min(fc1.shape)
            VT, S, U = np.linalg.svd((fc1.T).numpy())
            VTS = torch.tensor(VT)@torch.diag(torch.tensor(S))
            
            if len(config.MODEL.CHOICES_BLOCK_CONFIG) == 0:
                weight[f'{m}.VT.weight'] = VTS[:, :num_components].T
                weight[f'{m}.U.weight'] = torch.tensor(U[:num_components, :].T)
                weight[f'{m}.U.bias'] = weight[f'{m}.bias']
            else:
                for i, ratio in enumerate(config.MODEL.CHOICES_BLOCK_CONFIG[block_id][1]):
                    weight[f'{m}.VT.{i}.weight'] = VTS[:, :int(round(num_components*ratio))].T
                    weight[f'{m}.U.{i}.weight'] = torch.tensor(U[:int(round(num_components*ratio)), :].T)
                    weight[f'{m}.U.{i}.bias'] = weight[f'{m}.bias']
                
        if 'fc2.weight' in k:
            m = k[:-7]
            block_id = int(m.split('.')[1])
            fc2 = weight[f'{m}.weight']
            num_components = min(fc2.shape)
            VT, S, U = np.linalg.svd((fc2.T).numpy())
            SU = torch.diag(torch.tensor(S))@torch.tensor(U)
            if len(config.MODEL.CHOICES_BLOCK_CONFIG) == 0:
                weight[f'{m}.VT.weight'] = torch.tensor(VT[:, :num_components].T)
                weight[f'{m}.U.weight'] = SU[:num_components, :].T
                weight[f'{m}.U.bias'] = weight[f'{m}.bias']
            else:
                for i, ratio in enumerate(config.MODEL.CHOICES_BLOCK_CONFIG[block_id][2]):
                    weight[f'{m}.VT.{i}.weight'] = torch.tensor(VT[:, :int(round(num_components*ratio))].T)
                    weight[f'{m}.U.{i}.weight'] = SU[:int(round(num_components*ratio)), :].T
                    weight[f'{m}.U.{i}.bias'] = weight[f'{m}.bias']
                
                
    info = supernet.load_state_dict(weight, strict=False)
    #print(info)
    torch.save({'model': supernet.state_dict()}, f"{config['MODEL']['NAME']}.pth")


if __name__ == '__main__':
    args, config = parse_option()

    main(config)
