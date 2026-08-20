import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.metrics import compute_metrics
from scipy.ndimage import gaussian_filter
from utils.feature_transform import create_feature_transform
from utils.analysis import get_classification_from_segmentation, analyze_classification_distribution
from utils.visualization import visualize_anomaly_results
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.pe_wrapper import PE_FeatureExtractor
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def build_frozen_backbone(args, device, logger=None):
    """
    Build frozen visual backbone for test.

    args.backbone:
        pe_core_l14 / pe_l14_336
        clip_vitl14 / clip_vitl14_336

    Return:
        backbone_model
        embed_dim
        layer_indices
    """
    backbone_name = args.backbone.lower()
    layer_indices = [l - 1 for l in args.features_list]

    if backbone_name in ["pe_core_l14", "pe_l14_336", "pe"]:
        if logger is not None:
            logger.info("Using backbone: PE-Core-L14")

        backbone_model = PE_FeatureExtractor(
            layers=layer_indices,
            device=device
        )
        embed_dim = 1024

    elif backbone_name in ["clip_vitl14", "clip_vitl14_336", "clip"]:
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
            f"Supported: pe_core_l14, clip_vitl14"
        )

    backbone_model.eval()

    for p in backbone_model.parameters():
        p.requires_grad = False

    if logger is not None:
        logger.info(f"Backbone: {args.backbone}")
        logger.info(f"Image size: {args.image_size}")
        logger.info(f"Feature layers: {args.features_list}")
        logger.info(f"Actual block indices: {layer_indices}")
        logger.info(f"Embed dim: {embed_dim}")

    return backbone_model, embed_dim, layer_indices
    
def test(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # ===== 优先使用 checkpoint 里的训练配置，保证 train/test 一致 =====
    args.backbone = checkpoint.get("backbone", args.backbone)
    args.image_size = checkpoint.get("image_size", args.image_size)
    args.features_list = checkpoint.get("features_list", args.features_list)
    args.use_amrf = checkpoint.get("use_amrf", args.use_amrf)
    args.use_cepa = checkpoint.get("use_cepa", args.use_cepa)

    preprocess, target_transform = get_transform(args)

    # 1. 初始化冻结 backbone：PE / CLIP
    backbone_model, embed_dim, layer_indices = build_frozen_backbone(
        args=args,
        device=device,
        logger=logger
    )
    
    # 2. 从 checkpoint 加载训练好的 Token
    anomaly_token = checkpoint["anomaly_token"].to(device)
    normal_token = checkpoint["normal_token"].to(device)

    # 3. 加载特征映射变换  SAF
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict['mlp.0.weight'].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                dropout=0.0
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    # 4. 加载 CEPA / prototype calibration module
    cepa = None

    if args.use_cepa:
        if "cepa" not in checkpoint:
            raise ValueError(
                "Checkpoint indicates use_cepa=True, but 'cepa' is missing. "
                "This may silently turn full EXPA into w/o CEPA. "
                "Please check the training script and checkpoint."
            )

        from utils.cepa import build_cepa

        config = checkpoint.get("cepa_config", {})
        logger.info("CEPA enabled: loading CEPA prototype calibration module.")

        cepa = build_cepa(
            layers=args.features_list,
            embed_dim=embed_dim,
            res_scale_init=config.get("res_scale_init", 0.01)
        ).to(device)

        cepa.load_state_dict(checkpoint["cepa"])
        cepa.eval()

    else:
        logger.info("CEPA disabled: using original anomaly/normal prototypes.")

    # 5. 加载 AMRF / anomaly-response-guided layer fusion module
    amrf = None

    if args.use_amrf:
        if "amrf" not in checkpoint:
            raise ValueError(
                "Checkpoint indicates use_amrf=True, but 'amrf' is missing. "
                "This may silently turn full EXPA into w/o AMRF. "
                "Please check the training script and checkpoint."
            )

        from utils.amrf import build_amrf

        amrf_config = checkpoint.get("amrf_config", {})
        logger.info("AMRF enabled: loading AMRF dynamic fusion module.")

        amrf = build_amrf(
            num_layers=amrf_config.get("num_layers", len(args.features_list)),
            hidden_dim=amrf_config.get("hidden_dim", 32),
            topk_ratio=amrf_config.get("topk_ratio", 0.01),
            keep_sum_scale=amrf_config.get("keep_sum_scale", True)
        ).to(device)

        amrf.load_state_dict(checkpoint["amrf"])
        amrf.eval()

    else:
        logger.info("AMRF disabled: using equal-weight fusion.")

    test_data = Dataset(root=args.test_data_path, transform=preprocess,
                        target_transform=target_transform, dataset_name=args.test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}
    all_original_images, all_anomaly_maps, all_gt_masks = [], [], []
    all_cls_names, all_anomaly_labels, all_img_paths = [], [], []

    # === 在循环前初始化计时列表 ===
    e2e_latencies = []
    e2e_batch_sizes = []

    use_cuda = str(device).startswith("cuda")
    warmup_steps = 10

    if logger is not None:
        logger.info("\n" + "=" * 70)
        logger.info("🚀 开始 Ours 图像端到端测速 / Start Ours image-only end-to-end latency measurement")
        logger.info("测速范围：for items in dataloader 之后开始计时")
        logger.info("即 DataLoader 已经完成读图、resize、normalize、transform、组成 batch")
        logger.info("计时范围：image.to(device) → backbone/CEPA/AMRF 推理 → anomaly map / heatmap 生成 → 后处理结束")
        logger.info("不包含：DataLoader取batch、GT mask读取、GT mask二值化、results保存、指标计算")
        logger.info("=" * 70 + "\n")

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=use_cuda):
            for items in tqdm(test_dataloader):
                # ============================================================
                # 不计入测速的部分：GT mask 读取、GT mask 二值化、标签保存
                # ============================================================
                cls_name = items["cls_name"][0]

                gt_mask = items["img_mask"]
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                results[cls_name]["imgs_masks"].append(gt_mask)
                results[cls_name]["gt_sp"].extend(items["anomaly"].detach().cpu())
                    
                # ===== 开始测速 =====
                if use_cuda:
                    torch.cuda.synchronize(device)

                start = time.perf_counter()

                # 1. CPU 到 GPU 传输，计入测速
                image = items["img"].to(device)

                # 2. backbone 提取 patch tokens，计入测速
                patch_tokens_list = backbone_model(image)

                B = image.shape[0]

                # 3. 全局 anomaly / normal token，计入测速
                batch_anomaly_features = anomaly_token.expand(B, -1)
                batch_normal_features = normal_token.expand(B, -1)

                # 4. CEPA / prototype calibration，计入测速
                if cepa is not None:
                    adapted_list = cepa(
                        batch_anomaly_features,
                        batch_normal_features,
                        patch_tokens_list,
                        args.features_list
                    )
                    anomaly_features_list = [a["anomaly"] for a in adapted_list]
                    normal_features_list = [a["normal"] for a in adapted_list]
                else:
                    anomaly_features_list = [batch_anomaly_features] * len(patch_tokens_list)
                    normal_features_list = [batch_normal_features] * len(patch_tokens_list)

                anomaly_map_list = []

                for idx, patch_feature in enumerate(patch_tokens_list):
                    current_layer = args.features_list[idx]

                    anomaly_feat_norm = F.normalize(
                        anomaly_features_list[idx],
                        dim=1,
                        eps=1e-8
                    )
                    normal_feat_norm = F.normalize(
                        normal_features_list[idx],
                        dim=1,
                        eps=1e-8
                    )

                    # SAF / layer transform，计入测速
                    transform_key = f"layer_{current_layer}"
                    if transform_key in layer_transforms:
                        B_val, N_val, D_val = patch_feature.shape
                        patch_feature = layer_transforms[transform_key](
                            patch_feature.reshape(-1, D_val)
                        ).reshape(B_val, N_val, D_val)

                    patch_features_only = patch_feature

                    # anomaly map 生成，计入测速
                    vad_anomaly_map = generate_anomaly_map_from_tokens(
                        anomaly_feat_norm,
                        normal_feat_norm,
                        patch_features_only,
                        args.image_size
                    )

                    anomaly_map_list.append(vad_anomaly_map)

                # 5. AMRF / 等权融合，计入测速
                if amrf is not None:
                    final_anomaly_map, fusion_weights = amrf(anomaly_map_list)
                    final_anomaly_map = final_anomaly_map.cpu()
                else:
                    final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()

                # 6. 后处理，计入测速
                filtered_map = gaussian_filter(
                    final_anomaly_map[0].numpy(),
                    sigma=args.sigma
                )
                final_anomaly_map = torch.from_numpy(filtered_map).unsqueeze(0)

                # 等待本轮所有 GPU 任务完成
                if use_cuda:
                    torch.cuda.synchronize(device)

                end = time.perf_counter()
                # ===== 结束测速 =====

                latency_ms = (end - start) * 1000.0
                e2e_latencies.append(latency_ms)
                e2e_batch_sizes.append(image.size(0))

                # ============================================================
                # 不计入测速的部分：results 保存、指标计算所需缓存、分析图缓存、hook 刷新
                # ============================================================
                results[cls_name]["imgs_masks"][-1] = gt_mask.detach().cpu().to(torch.uint8)
                results[cls_name]["anomaly_maps"].append(final_anomaly_map.half())

                if args.enable_analysis:
                    all_original_images.append(image.detach().cpu().float())
                    all_gt_masks.append(gt_mask.detach().cpu().to(torch.uint8))

                # 分类得分、指标及可视化都需要这些完整样本信息
                all_anomaly_maps.append(final_anomaly_map)
                all_cls_names.append(cls_name)
                all_anomaly_labels.append(items["anomaly"].item())
                all_img_paths.append(items["img_path"][0])

                # 刷新 backbone hooks，避免下一张图重复累积特征
                if hasattr(backbone_model, "remove_hooks"):
                    backbone_model.remove_hooks()

                if hasattr(backbone_model, "_register_hooks"):
                    backbone_model._register_hooks()

    # === 在整个测速循环结束后，计算平均值 ===
    # 速度统计：跳过 warmup
    if len(e2e_latencies) > warmup_steps:
        valid_latencies = e2e_latencies[warmup_steps:]
        valid_batch_sizes = e2e_batch_sizes[warmup_steps:]
    else:
        valid_latencies = e2e_latencies
        valid_batch_sizes = e2e_batch_sizes

    valid_latencies = np.array(valid_latencies, dtype=np.float64)
    valid_batch_sizes = np.array(valid_batch_sizes, dtype=np.float64)

    all_latencies = np.array(e2e_latencies, dtype=np.float64)
    all_batch_sizes = np.array(e2e_batch_sizes, dtype=np.float64)

    if len(valid_latencies) == 0 or np.sum(valid_batch_sizes) == 0:
        if logger is not None:
            logger.warning("[测速警告] 没有有效的测速数据，请检查 test_dataloader 是否为空。")
        return

    batch_size_setting = int(e2e_batch_sizes[0]) if len(e2e_batch_sizes) > 0 else args.batch_size

    avg_batch_latency = np.mean(valid_latencies)
    avg_image_latency = np.sum(valid_latencies) / np.sum(valid_batch_sizes)

    fps = 1000.0 / avg_image_latency
    throughput = 1000.0 * np.sum(valid_batch_sizes) / np.sum(valid_latencies)

    total_time_s = np.sum(all_latencies) / 1000.0
    total_images = int(np.sum(all_batch_sizes))

    median_image_latency = np.median(valid_latencies / valid_batch_sizes)
    std_batch_latency = np.std(valid_latencies)

    # ===== 速度报告 =====
    if logger is not None:
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Ours 图像端到端测速报告 / Ours Image-only End-to-End Latency Report 🚀")
        logger.info("[*] 计时范围 / Timing range:")
        logger.info("    DataLoader 已经完成读图、resize、normalize、transform、组成 batch")
        logger.info("    计时范围：CPU到GPU传输 + backbone/CEPA/AMRF 推理 + 异常热力图生成/后处理")
        logger.info("    Timing starts after dataloader fetching: H2D transfer + inference + anomaly-map generation/post-processing")
        logger.info("[*] 排除内容 / Excluded:")
        logger.info("    DataLoader取batch、GT mask读取、GT mask二值化、results保存、指标计算")
        logger.info("    Dataloader fetching, GT-mask loading, GT-mask binarization, result bookkeeping, metric computation")

        logger.info(f"[*] Batch size 设置 / Batch size setting: {batch_size_setting}")
        logger.info(f"[*] 剔除预热步数 / Warmup steps removed: {warmup_steps}")
        logger.info(f"[*] 有效统计 batch 数 / Valid batches: {len(valid_latencies)}")
        logger.info(f"[*] 总测速图片数 / Total measured images: {total_images}")

        logger.info(f"[*] 平均每 batch 耗时 / Avg batch latency: {avg_batch_latency:.2f} ms/batch")
        logger.info(f"[*] 平均每张图耗时 / Avg image latency: {avg_image_latency:.2f} ms/image")
        logger.info(f"[*] 每张图中位耗时 / Median image latency: {median_image_latency:.2f} ms/image")
        logger.info(f"[*] Batch 耗时标准差 / Std batch latency: {std_batch_latency:.2f} ms")

        logger.info(f"[*] FPS / 每秒处理图片数: {fps:.2f} images/s")
        logger.info(f"[*] 吞吐量 / Throughput: {throughput:.2f} images/s")
        logger.info(f"[*] 总测速耗时 / Total measured time: {total_time_s:.2f} s")
        logger.info("=" * 70 + "\n")

    fused_scores, normalized_anomaly_maps = get_classification_from_segmentation(all_anomaly_maps, all_cls_names, results)
    compute_metrics(results, obj_list, logger)

    # 可视化界面
    if args.enable_analysis:
        analysis_dir = os.path.join(args.save_path, "analysis")

        analyze_classification_distribution(
            fused_scores,
            all_cls_names,
            all_anomaly_labels,
            analysis_dir
        )

        visualize_anomaly_results(
            all_original_images,
            normalized_anomaly_maps,
            all_gt_masks,
            fused_scores,
            all_cls_names,
            all_img_paths,
            all_anomaly_labels,
            args.test_dataset,
            analysis_dir
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser("EXPA Test", add_help=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="test dataset path")
    parser.add_argument("--save_path", type=str, default='./test_results', help='path to save test results')
    parser.add_argument("--test_dataset", type=str, required=True, help="test dataset name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")
    parser.add_argument("--backbone",type=str,default="pe_core_l14",choices=["pe_core_l14", "clip_vitl14", "pe_l14_336", "clip_vitl14_336","dinov2_vitl14"],help="Visual backbone to use. Usually overwritten by checkpoint['backbone']."
    )
    parser.add_argument("--image_size", type=int, default=518, help="image size. Usually overwritten by checkpoint.")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24],
                        help="feature layers. Usually overwritten by checkpoint.")
    parser.add_argument("--enable_analysis", action="store_true", help="enable data analysis and visualization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_amrf",
        type=lambda x: x.lower() == "true",
        default=True,
        help="whether to use AMRF dynamic fusion"
    )
    parser.add_argument(
    "--use_cepa",
    type=lambda x: x.lower() == "true",
    default=True,
    help="whether to use context-aware external prototype adaptation"
    )

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)