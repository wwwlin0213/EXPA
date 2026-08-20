import VisualAD_lib
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import argparse
import torch.nn.functional as F
from utils.loss import FocalLoss, BinaryDiceLoss, ContrastiveLoss
from dataset import Dataset
from utils.logger import get_logger
from utils.training_utils import (
    print_training_parameters, validate_training_setup, setup_model_training,
    create_optimizer, create_optimizer_v2, setup_feature_transforms, check_for_nan,
    compute_segmentation_loss, validate_gradients, save_checkpoint
)
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO
from utils.pe_wrapper import PE_FeatureExtractor

from utils.amrf import build_amrf
from utils.parameter_count import count_total_parameters

torch.use_deterministic_algorithms(True,warn_only=False)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=False)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def generate_anomaly_map_from_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """
    Generate pixel-level anomaly map using token features with numerical stability
    Args:
        anomaly_features: [B, dim] - anomaly token features
        normal_features: [B, dim] - normal token features  
        patch_tokens: [B, num_patches, dim] - patch token features
        image_size: target image size
    Returns:
        anomaly_map: [B, H, W] - pixel-level anomaly map
    """
    B = anomaly_features.shape[0]
    
    # Normalize all features to prevent numerical instability in cosine similarity
    anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
    normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
    patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)
    
    # Compute similarity between each patch and anomaly/normal tokens
    anomaly_sim = torch.cosine_similarity(
        patch_tokens_norm, anomaly_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    normal_sim = torch.cosine_similarity(
        patch_tokens_norm, normal_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    # Anomaly score = anomaly_similarity - normal_similarity
    anomaly_score = anomaly_sim - normal_sim  # [B, num_patches]
    
    # Check for NaN in anomaly scores
    if torch.isnan(anomaly_score).any():
        print(f"Warning: NaN detected in anomaly_score")
        anomaly_score = torch.nan_to_num(anomaly_score, nan=0.0)
    
    # Reshape to spatial dimensions
    num_patches = anomaly_score.shape[1]
    patch_size_from_model = int(np.sqrt(num_patches))
    
    anomaly_map = anomaly_score.reshape(B, patch_size_from_model, patch_size_from_model)
    
    # Resize to target image size
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(1), 
        size=(image_size, image_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)
    
    return anomaly_map

def compute_classification_loss_V2(anomaly_maps_list, labels, device, amrf=None, return_weights=False):
    """
    Compute classification loss with optional AMRF.
    """
    if not anomaly_maps_list:
        zero_loss = torch.tensor(0.0, device=device)
        if return_weights:
            return zero_loss, None, None
        return zero_loss

    if amrf is None:
        # 原始方式：等权求和融合
        final_anomaly_maps = torch.stack(anomaly_maps_list).sum(dim=0)  # [B, H, W]
        fusion_weights = None
    else:
        # AMRF：输入自适应多层响应融合
        final_anomaly_maps, fusion_weights = amrf(anomaly_maps_list)

    seg_scores = reduce_anomaly_map(
        final_anomaly_maps,
        mode="topk_mean",
        topk_ratio=DEFAULT_TOPK_RATIO
    )  # [B]

    labels_float = labels.float().to(device)
    loss = F.binary_cross_entropy_with_logits(seg_scores, labels_float)

    if return_weights:
        return loss, fusion_weights, final_anomaly_maps

    return loss

def build_frozen_backbone(args, device, logger=None):
    """
    Build frozen visual backbone.

    Supported:
        pe_l14_336        -> PE-Core-L14-336
        clip_vitl14_336   -> CLIP ViT-L/14@336px

    Return:
        backbone_model: frozen visual encoder wrapper
        embed_dim: feature dimension
        layer_indices: actual transformer block indices, e.g. [5, 11, 17, 23]
    """
    backbone_name = args.backbone.lower()
    layer_indices = [l - 1 for l in args.features_list]

    if backbone_name in ["pe_l14_336", "pe-core-l14-336", "pe_core_l14_336", "pe"]:
        if logger is not None:
            logger.info("Using backbone: PE-Core-L14-336")

        backbone_model = PE_FeatureExtractor(
            layers=layer_indices,
            device=device
        )

        embed_dim = 1024

    elif backbone_name in ["clip_vitl14_336", "clip-vit-l14-336", "vit-l/14@336px", "clip"]:
        from models.backbone.clip.clip_visualad import CLIP_FeatureExtractor

        if logger is not None:
            logger.info("Using backbone: CLIP ViT-L/14@336px")

        backbone_model = CLIP_FeatureExtractor(
            layers=layer_indices,
            device=device,
            weight_path="models/backbone/clip/weights/ViT-L-14-336px.pt"
        )
        embed_dim = 1024

    elif backbone_name in ["dinov2_vitl14", "dinov2-l/14", "dinov2"]:
        from models.backbone.dinov2.dinov2_wrapper import DINOv2_FeatureExtractor

        if logger is not None:
            logger.info("Using backbone: DINOv2 ViT-L/14")

        backbone_model = DINOv2_FeatureExtractor(
            backbone_name="DINOv2-L/14",
            layers=layer_indices,
            device=device,
            repo_path="./models/backbone/dinov2/official",
            weight_path="./models/backbone/dinov2/weights/dinov2_vitl14_pretrain.pth",
        )

        embed_dim = 1024

    else:
        raise ValueError(
            f"Unsupported backbone: {args.backbone}. "
            f"Use one of: pe_l14_336, clip_vitl14_336"
        )

    backbone_model.eval()

    for param in backbone_model.parameters():
        param.requires_grad = False

    if logger is not None:
        logger.info(f"Feature layers: {args.features_list}")
        logger.info(f"Actual block indices: {layer_indices}")
        logger.info(f"Backbone embed_dim: {embed_dim}")

    return backbone_model, embed_dim, layer_indices

def train(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    # 根据你传入的命令行参数（主要是 args.image_size）来生成数据预处理流水线
    preprocess, target_transform = get_transform(args)

    # 打印超参摘要
    print_training_parameters(args, logger)

    # 1. 初始化冻结视觉骨干网络：PE-Core-L14-336 或 CLIP ViT-L/14@336px
    backbone_model, embed_dim, layer_indices = build_frozen_backbone(
        args=args,
        device=device,
        logger=logger
    )

    # 2. 在外部显式定义全局可学习 Token
    anomaly_token = nn.Parameter(torch.randn(1, embed_dim, device=device) * 0.02)
    normal_token = nn.Parameter(torch.randn(1, embed_dim, device=device) * 0.02)

    cepa = None

    if args.use_cepa:
        from utils.cepa import build_cepa
        cepa = build_cepa(
            layers=args.features_list,
            embed_dim=embed_dim,
            res_scale_init=0.01
        ).to(device)
        logger.info("CEPA enabled.")
    else:
        logger.info("CEPA disabled: using original anomaly/normal prototypes.")

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    layer_transforms = setup_feature_transforms(args.features_list, device, embed_dim)

    if args.use_amrf:
        amrf = build_amrf(
            num_layers=len(args.features_list),
            hidden_dim=args.amrf_hidden_dim,
            topk_ratio=DEFAULT_TOPK_RATIO,
            keep_sum_scale=True
        ).to(device)
    else:
        amrf = None

    # 优化器:AdamW，多组学习率
    optimizer = create_optimizer_v2(
        anomaly_token,
        layer_transforms,
        normal_token,
        args,
        cepa
    )
    if amrf is not None:
        optimizer.add_param_group({
            "params": amrf.parameters(),
            "lr": args.learning_rate,
            "weight_decay": 1e-4
        })

    total_params = count_total_parameters(
        backbone_model=backbone_model,
        layer_transforms=layer_transforms,
        cepa=cepa,
        amrf=amrf,
        anomaly_token=anomaly_token,
        normal_token=normal_token,
    )
    logger.info(
        f"Total parameters: {total_params:,} "
        f"({total_params / 1e6:.2f} M)"
    )

    # 余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.learning_rate * 0.1)

    amp_enabled = False
    scaler = GradScaler(enabled=amp_enabled)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_token_relation = ContrastiveLoss(temperature=0.1, margin=0.5)

    for epoch in tqdm(range(args.epoch)):
        if cepa is not None:
            cepa.train()
        if amrf is not None:
            amrf.train()

        loss_list, final_seg_loss_list, image_loss_list, token_relation_loss_list = [], [], [], []
        fusion_weight_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze(1).to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0

            def compute_losses():
                # 1. 图像进入冻结视觉骨干网络，拿到 patch_tokens_list
                #    PE 和 CLIP wrapper 都必须返回 list of [B, 576, 1024]
                with torch.no_grad():
                    patch_tokens_list = backbone_model(image)

                B = image.shape[0]
                # 2. 将外部全局 Token 扩展到 Batch 维度
                batch_anomaly_features = anomaly_token.expand(B, -1)
                batch_normal_features = normal_token.expand(B, -1)

                # 3. 在外部会议室进行交互
                if cepa is not None:
                    adapted_list = cepa(
                        batch_anomaly_features,
                        batch_normal_features,
                        patch_tokens_list,
                        args.features_list
                    )
                    anomaly_features_list = [adapted['anomaly'] for adapted in adapted_list]
                    normal_features_list = [adapted['normal'] for adapted in adapted_list]
                else:
                    anomaly_features_list = [batch_anomaly_features] * len(patch_tokens_list)
                    normal_features_list = [batch_normal_features] * len(patch_tokens_list)
                    
                final_anomaly_features = anomaly_features_list[-1]
                final_normal_features = normal_features_list[-1]
                final_anomaly_features_norm = F.normalize(final_anomaly_features, dim=1, eps=1e-8)
                final_normal_features_norm = F.normalize(final_normal_features, dim=1, eps=1e-8)

                token_relation_val = loss_token_relation(final_anomaly_features_norm, final_normal_features_norm)

                similarity_map_list = []
                anomaly_maps_list = []

                for idx_layer, patch_feature in enumerate(patch_tokens_list):
                    anomaly_feat_norm = F.normalize(anomaly_features_list[idx_layer], dim=1, eps=1e-8)
                    normal_feat_norm = F.normalize(normal_features_list[idx_layer], dim=1, eps=1e-8)

                    current_layer = args.features_list[idx_layer]
                    transform_key = f'layer_{current_layer}'

                    if transform_key in layer_transforms:
                        batch_size, num_patches, feat_dim = patch_feature.shape
                        patch_feature_flat = patch_feature.reshape(-1, feat_dim)
                        transformed_feature = layer_transforms[transform_key](patch_feature_flat)
                        patch_feature = transformed_feature.reshape(batch_size, num_patches, feat_dim)

                    # 生成热力图 (因为 pe_wrapper 已经剥离了 CLS，所以直接传 patch_feature)
                    anomaly_map = generate_anomaly_map_from_tokens(
                        anomaly_feat_norm, normal_feat_norm,
                        patch_feature, args.image_size
                    )

                    anomaly_map_sigmoid = torch.sigmoid(anomaly_map)
                    similarity_map = torch.stack([1 - anomaly_map_sigmoid, anomaly_map_sigmoid], dim=1)
                    similarity_map_list.append(similarity_map)
                    anomaly_maps_list.append(anomaly_map)

                image_val, fusion_weights, final_anomaly_maps = compute_classification_loss_V2(
                    anomaly_maps_list,
                    label,
                    device,
                    amrf=amrf,
                    return_weights=True
                )
                
                # 原来的每层 anomaly map segmentation loss
                seg_val = compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice)

                # 对 AMRF 输出的 fused anomaly map 计算 segmentation loss
                final_anomaly_map_sigmoid = torch.sigmoid(final_anomaly_maps)

                final_similarity_map = torch.stack(
                    [1 - final_anomaly_map_sigmoid, final_anomaly_map_sigmoid],
                    dim=1
                )

                final_seg_val = compute_segmentation_loss(
                    [final_similarity_map],
                    gt,
                    loss_focal,
                    loss_dice
                )

                total = image_val + token_relation_val + seg_val + args.final_seg_weight * final_seg_val

                return total, seg_val, final_seg_val, image_val, token_relation_val, fusion_weights

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                result = compute_losses()
            if result is None: continue

            total_loss, seg_loss_val, final_seg_loss_val, image_loss_val, token_relation_loss_val, fusion_weights = result

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            # 梯度裁剪防爆
            torch.nn.utils.clip_grad_norm_([anomaly_token, normal_token], max_norm=1.0)
            
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            if fusion_weights is not None:
                fusion_weight_list.append(
                    fusion_weights.detach().mean(dim=0).cpu().numpy()
                )

            loss_list.append(float(seg_loss_val.item()))
            final_seg_loss_list.append(float(final_seg_loss_val.item()))
            image_loss_list.append(float(image_loss_val.item()))
            token_relation_loss_list.append(float(token_relation_loss_val.item()))

        scheduler.step()
        if (epoch + 1) % args.print_freq == 0:
            logger.info(f'Epoch [{epoch+1}/{args.epoch}] - seg: {np.mean(loss_list):.4f}, '
                        f'final_seg: {np.mean(final_seg_loss_list):.4f}, '
                        f'cls: {np.mean(image_loss_list):.4f}, '
                        f'contra: {np.mean(token_relation_loss_list):.4f}')

            if len(fusion_weight_list) > 0:
                mean_weights = np.stack(fusion_weight_list, axis=0).mean(axis=0)
                weight_info = ', '.join([
                    f'L{layer}: {w:.4f}'
                    for layer, w in zip(args.features_list, mean_weights)
                ])
                logger.info(f'AMRF fusion weights: {weight_info}')

        if (epoch + 1) % args.save_freq == 0:
            transform_state_dict = {name: t.state_dict() for name, t in layer_transforms.items()}
            checkpoint_data = {
                "backbone": args.backbone,
                "anomaly_token": anomaly_token.data.clone(),
                "normal_token": normal_token.data.clone(),
                "features_list": args.features_list,
                "image_size": args.image_size,
                "epoch": epoch + 1,
                "final_seg_weight": args.final_seg_weight,
                "layer_transforms": transform_state_dict,
                "use_cepa": args.use_cepa,
                "use_amrf": args.use_amrf,
            }
            if cepa is not None:
                checkpoint_data["cepa"] = cepa.state_dict()
                checkpoint_data["cepa_config"] = {
                    "res_scale_init": 0.01
                }
            if amrf is not None:
                checkpoint_data["amrf"] = amrf.state_dict()
                checkpoint_data["amrf_config"] = {
                    "num_layers": len(args.features_list),
                    "hidden_dim": args.amrf_hidden_dim,
                    "topk_ratio": DEFAULT_TOPK_RATIO,
                    "keep_sum_scale": True
                }
            torch.save(checkpoint_data, os.path.join(args.save_path, f'epoch_{epoch + 1}.pth'))

    logger.info(f'Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("EXPA Training", add_help=True)
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training dataset root.",)
    parser.add_argument("--save_path", type=str, default='./checkpoints', help='path to save results')
    parser.add_argument("--train_dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--backbone",type=str,default="pe_l14_336",choices=["pe_l14_336", "clip_vitl14_336","dinov2_vitl14"],help="Visual backbone to use: pe_l14_336 or clip_vitl14_336"
    )
    parser.add_argument("--feature_config", type=str, default=os.path.join('configs', 'backbone_layers.yaml'),
                        help="YAML file specifying default feature layers per backbone")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24],
                        help="Override feature layers (falls back to YAML config if omitted)")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")
    parser.add_argument("--use_amrf",type=lambda x: x.lower() == "true",default=True,help="whether to use adaptive multi-level response fusion (AMRF)"
) 
    
    parser.add_argument("--amrf_hidden_dim", type=int, default=32,
                    help="hidden dimension of the AMRF weight-prediction network")
    parser.add_argument("--final_seg_weight", type=float, default=0.4,
                    help="weight for segmentation loss on dynamically fused final anomaly map")
    parser.add_argument("--use_cepa",
        type=lambda x: x.lower() == "true",
        default=True,
        help="whether to use context-aware external prototype adaptation"
    )
    args = parser.parse_args()
    if not os.path.isdir(args.train_data_path):
        raise FileNotFoundError(
            f"Training dataset directory not found: {args.train_data_path}"
        )
    setup_seed(args.seed)
    device = torch.device(args.device)
    train(args)
