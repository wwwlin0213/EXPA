import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc
from skimage import measure
from tabulate import tabulate
import warnings

# 忽略计算 F1 时可能出现的分母为 0 警告
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    """
    优化版的 PRO Score 计算函数 (支持不同尺寸图像，极大提升计算速度)
    """
    if not masks or not amaps:
        return 0.0

    # 1. 获取全局最大最小值
    min_val = min(a.min() for a in amaps)
    max_val = max(a.max() for a in amaps)
    if min_val == max_val:
        return 0.0

    delta = (max_val - min_val) / max_step
    pros, fprs = [], []
    
    # 2. 预计算核心信息 (移出阈值循环外，大幅提速)
    # 为每张真实掩码提前计算好连通域 (regions)
    regions_list = [measure.regionprops(measure.label(mask)) for mask in masks]
    # 提前计算所有图像的反向掩码总和 (正常像素总数)
    inverse_masks = [1 - mask for mask in masks]
    total_inverse_pixels = sum(inv.sum() for inv in inverse_masks)
    
    if total_inverse_pixels == 0:
        return 0.0
        # return float('nan')

    # 3. 遍历阈值
    for th in np.arange(min_val, max_val, delta):
        pro_list = []
        total_fp = 0
        
        for mask, amap, regions, inv_mask in zip(masks, amaps, regions_list, inverse_masks):
            # 当前图像的二值化预测
            binary_amap = amap > th
            
            # 计算 TP 区域重叠率
            for region in regions:
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pro_list.append(tp_pixels / region.area)
            
            # 计算 FP 像素数
            total_fp += np.logical_and(inv_mask, binary_amap).sum()
        
        # 记录当前阈值下的 PRO 和 FPR
        mean_pro = np.mean(pro_list) if len(pro_list) > 0 else 0.0
        fpr = total_fp / total_inverse_pixels
        
        pros.append(mean_pro)
        fprs.append(fpr)
        
    # 4. 计算指定 FPR 上限下的 AUC
    pros, fprs = np.array(pros), np.array(fprs)
    idx = fprs < expect_fpr
    
    if idx.sum() > 1:
        fprs_filtered = fprs[idx]
        pros_filtered = pros[idx]
        # 归一化 FPR 轴
        fpr_range = fprs_filtered.max() - fprs_filtered.min()
        if fpr_range > 0:
            fprs_normalized = (fprs_filtered - fprs_filtered.min()) / fpr_range
            pro_auc = auc(fprs_normalized, pros_filtered)
        else:
            pro_auc = 0.0
    else:
        pro_auc = 0.0
    
    if not np.isfinite(pro_auc):
        pro_auc = 0.0

    return pro_auc


def compute_metrics(results, obj_list, logger):
    """
    统一的评估入口函数，直接对接 test.py 生成的 results 字典。
    支持返回 table_ls 并自动打印规范表格。
    """
    logger.info("开始计算 8 大核心评估指标 (包含 AUPRO 0.3 和 0.05)...")
    
    table_ls = []
    auroc_sp_ls, auroc_px_ls = [], []
    f1_sp_ls, f1_px_ls = [], []
    ap_sp_ls, ap_px_ls = [], []
    aupro03_ls, aupro005_ls = [], []

    for obj in obj_list:
        # 如果该类别在数据集中不存在，跳过
        if len(results[obj]['gt_sp']) == 0:
            continue
            
        table = [obj]
        
        # # 1. 图像级别数据提取 (无需处理形状，本来就是 1D)
        # gt_sp = np.array(results[obj]['gt_sp'])
        # pr_sp = np.array(results[obj]['pr_sp'])

        # 1. 图像级别数据提取
        gt_sp = np.array(results[obj]['gt_sp']).astype(np.int32)
        pr_sp = np.array(results[obj]['pr_sp'], dtype=np.float32)

        # 安全处理 shape，避免 pr_sp 是 [N, 1] 或 [N, 2]
        pr_sp = np.squeeze(pr_sp)

        # 如果 pr_sp 是二分类概率 [N, 2]，取异常类概率
        if pr_sp.ndim == 2 and pr_sp.shape[1] == 2:
            pr_sp = pr_sp[:, 1]

        # BrainMRI / OCT17 上可能因为某些归一化或异常图全常数导致 NaN / Inf
        # sklearn 的 roc_auc_score 不能接收 NaN，所以这里强制清理
        if not np.all(np.isfinite(pr_sp)):
            logger.warning(
                f"[WARN] {obj}: pr_sp contains NaN/Inf, replaced by finite values."
            )
            pr_sp = np.nan_to_num(
                pr_sp,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )
        
        # 2. 像素级别数据提取
        gt_px_list_flat = [] # 用于 AUC, F1, AP (展平)
        pr_px_list_flat = [] 
        gt_px_2d_list = []   # 用于 PRO 计算 (保留空间维度)
        pr_px_2d_list = []   

        for i in range(len(results[obj]['imgs_masks'])):
            mask = results[obj]['imgs_masks'][i]
            amap = results[obj]['anomaly_maps'][i]
            
            # 安全地剥离多余维度，确保最终是 2D (H, W)
            if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
            if isinstance(amap, torch.Tensor): amap = amap.detach().cpu().numpy()
            
            mask = np.squeeze(mask)
            amap = np.squeeze(amap)

            # 如果 mask 是三通道图，转成单通道。
            # 兼容 HWC: (H, W, 3) 和 CHW: (3, H, W)
            if mask.ndim == 3:
                if mask.shape[0] in [1, 3]:
                    mask = np.max(mask, axis=0)
                else:
                    mask = np.max(mask, axis=-1)

            # 如果 anomaly map 不是 2D，也压成 2D。
            if amap.ndim == 3:
                if amap.shape[0] in [1, 3]:
                    amap = np.max(amap, axis=0)
                else:
                    amap = np.max(amap, axis=-1)

            # 分割数据集的 mask 常见为 0/255 或浮点灰度图。
            # 这里统一二值化，保证 pixel-level AUC/F1/AP/PRO 的 gt 是 0/1。
            mask = (mask > 0).astype(np.uint8)

            # anomaly map 也做一次数值清理，避免 NaN/Inf 进入指标计算。
            amap = np.nan_to_num(
                amap.astype(np.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )
            
            # 存入列表
            gt_px_2d_list.append(mask)
            pr_px_2d_list.append(amap)
            gt_px_list_flat.append(mask[::4, ::4].ravel())
            pr_px_list_flat.append(amap[::4, ::4].ravel())

        # 将所有的展平数组拼接成一个巨大的一维数组
        gt_px_all = np.concatenate(gt_px_list_flat)
        pr_px_all = np.concatenate(pr_px_list_flat)

        # ==================== 指标计算区 ====================
        
        """ 工业异常检测
        # 1. 像素级 (Pixel-level)
        auroc_px = roc_auc_score(gt_px_all, pr_px_all)
        ap_px = average_precision_score(gt_px_all, pr_px_all)
        p_precisions, p_recalls, _ = precision_recall_curve(gt_px_all, pr_px_all)
        f1_px = np.max((2 * p_precisions * p_recalls) / (p_precisions + p_recalls + 1e-6))

        # ================== 修改点 2 开始 ==================
        # 新增: 立刻从内存中删除这几个巨大的变量，防止影响后续的 PRO 计算
        del gt_px_all, pr_px_all, p_precisions, p_recalls
        # ================== 修改点 2 结束 ==================
        
        # 2. 图像级 (Image-level)
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        i_precisions, i_recalls, _ = precision_recall_curve(gt_sp, pr_sp)
        f1_sp = np.max((2 * i_precisions * i_recalls) / (i_precisions + i_recalls + 1e-6))
        
        # 3. 区域级 (Region-level PRO)
        aupro03 = cal_pro_score(gt_px_2d_list, pr_px_2d_list, expect_fpr=0.3)
        aupro005 = cal_pro_score(gt_px_2d_list, pr_px_2d_list, expect_fpr=0.05)
        """

        # ==================== 指标计算区 ====================
        
        """ medicial """
        # 1. 像素级 (Pixel-level)
        # 对 OCT17 这类没有真实 pixel mask 的数据，gt_px_all 往往全为 0。
        # 此时不能计算 roc_auc_score / precision_recall_curve / average_precision_score。
        if gt_px_all.size > 0 and len(np.unique(gt_px_all)) > 1:
            auroc_px = roc_auc_score(gt_px_all, pr_px_all)
            ap_px = average_precision_score(gt_px_all, pr_px_all)
            p_precisions, p_recalls, _ = precision_recall_curve(gt_px_all, pr_px_all)
            f1_px = np.max((2 * p_precisions * p_recalls) / (p_precisions + p_recalls + 1e-6))
        else:
            auroc_px = 0.0
            ap_px = 0.0
            f1_px = 0.0

        # 新增: 立刻从内存中删除这几个巨大的变量，防止影响后续的 PRO 计算
        del gt_px_all, pr_px_all
        
        # 2. 图像级 (Image-level)
        # 如果某个数据集或某个类别只有 normal 或只有 anomaly，也不能计算 image-level AUC。
        if (
            gt_sp.size > 0
            and pr_sp.size == gt_sp.size
            and len(np.unique(gt_sp)) > 1
            and np.all(np.isfinite(pr_sp))
        ):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            ap_sp = average_precision_score(gt_sp, pr_sp)
            i_precisions, i_recalls, _ = precision_recall_curve(gt_sp, pr_sp)
            f1_sp = np.max((2 * i_precisions * i_recalls) / (i_precisions + i_recalls + 1e-6))
        else:
            auroc_sp = 0.0
            ap_sp = 0.0
            f1_sp = 0.0
        
        # 3. 区域级 (Region-level PRO)
        # 对没有真实 pixel mask 的数据，PRO 没有实际意义，直接置 0。
        has_positive_pixel = any(np.max(mask) > 0 for mask in gt_px_2d_list)

        if has_positive_pixel:
            aupro03 = cal_pro_score(gt_px_2d_list, pr_px_2d_list, expect_fpr=0.3)
            aupro005 = cal_pro_score(gt_px_2d_list, pr_px_2d_list, expect_fpr=0.05)
        else:
            aupro03 = 0.0
            aupro005 = 0.0
        # ==================== 结果收集区 ====================
        
        # 百分制转换
        metrics_vals = [
            auroc_sp * 100, f1_sp * 100, ap_sp * 100,
            auroc_px * 100, f1_px * 100, ap_px * 100, 
            aupro03 * 100, aupro005 * 100
        ]
        
        table.extend([f"{v:.1f}" for v in metrics_vals])
        table_ls.append(table)
        
        # 存入列表用于计算均值
        auroc_px_ls.append(auroc_px)
        f1_px_ls.append(f1_px)
        ap_px_ls.append(ap_px)
        aupro03_ls.append(aupro03)
        aupro005_ls.append(aupro005)
        auroc_sp_ls.append(auroc_sp)
        f1_sp_ls.append(f1_sp)
        ap_sp_ls.append(ap_sp)

    # ==================== 计算均值并打印表格 ====================
    
    mean_values = [
        'mean',
        f"{np.mean(auroc_sp_ls) * 100:.1f}",
        f"{np.mean(f1_sp_ls) * 100:.1f}",
        f"{np.mean(ap_sp_ls) * 100:.1f}",
        f"{np.mean(auroc_px_ls) * 100:.1f}",
        f"{np.mean(f1_px_ls) * 100:.1f}",
        f"{np.mean(ap_px_ls) * 100:.1f}",
        f"{np.mean(aupro03_ls) * 100:.1f}",
        f"{np.mean(aupro005_ls) * 100:.1f}"
    ]
    table_ls.append(mean_values)

    # 打印排版精美的控制台/日志表格
    # 定义表头（顺序与 table_ls 一致）
    headers = ['Category', 'I-AUC', 'I-F1', 'I-AP', 'P-AUC', 'P-F1', 'P-AP', 'PRO0.3', 'PRO.05']
    # 使用 tabulate 生成 pipe 格式表格
    table_str = tabulate(table_ls, headers=headers, tablefmt='pipe')
    logger.info("\n%s", table_str)

    return table_ls