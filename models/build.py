# --------------------------------------------------------
# TinyViT Model Builder
# Copyright (c) 2022 Microsoft
# --------------------------------------------------------


from .deit import VisionTransformer
from .swin_transformer import SwinTransformer
from .lr_swin_transformer import LRSwinTransformer
from .lr_swin_transformer_subnet import LRSwinTransformer as LRSwinTransformerSubnet
from .lr_deit import LRVisionTransformer
from .lr_deit_subnet import LRVisionTransformerSubnet

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS
        )
    elif model_type == 'deit':
        model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DEIT.PATCH_SIZE,
            in_chans=config.MODEL.DEIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DEIT.EMBED_DIM,
            depth=config.MODEL.DEIT.DEPTH,
            num_heads = config.MODEL.DEIT.NUM_HEADS,
            mlp_ratio = config.MODEL.DEIT.MLP_RATIO,
            qkv_bias = config.MODEL.DEIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == 'lr_swin':
        model = LRSwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS
        )
    elif model_type == 'lr_swin_subnet':
        model = LRSwinTransformerSubnet(
            svd_config=config.MODEL.SWIN.SVD_CONFIG,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS
        )
    elif model_type == 'lr_deit':
        model = LRVisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DEIT.PATCH_SIZE,
            in_chans=config.MODEL.DEIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DEIT.EMBED_DIM,
            depth=config.MODEL.DEIT.DEPTH,
            num_heads = config.MODEL.DEIT.NUM_HEADS,
            mlp_ratio = config.MODEL.DEIT.MLP_RATIO,
            qkv_bias = config.MODEL.DEIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            fused_lr=config.MODEL.DEIT.FUSE_LR,
        )
    elif model_type == 'lr_deit_subnet':
        model = LRVisionTransformerSubnet(
            svd_config = config.MODEL.DEIT.SVD_CONFIG,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DEIT.PATCH_SIZE,
            in_chans=config.MODEL.DEIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DEIT.EMBED_DIM,
            depth=config.MODEL.DEIT.DEPTH,
            num_heads = config.MODEL.DEIT.NUM_HEADS,
            mlp_ratio = config.MODEL.DEIT.MLP_RATIO,
            qkv_bias = config.MODEL.DEIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
