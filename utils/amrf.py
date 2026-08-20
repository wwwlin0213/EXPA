import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiLevelResponseFusion(nn.Module):
    """Adaptive Multi-level Response Fusion (AMRF).

    AMRF predicts input-dependent fusion weights from four statistical
    descriptors of each layer-level anomaly response map: mean, maximum,
    standard deviation, and top-k mean.

    Args:
        num_layers: Number of layer-level anomaly maps.
        hidden_dim: Hidden dimension of the weight-prediction MLP.
        topk_ratio: Ratio used by the top-k mean descriptor.
        keep_sum_scale: Multiply the weighted sum by ``num_layers`` so that
            its scale remains comparable to equal-weight sum fusion.

    Input:
        anomaly_maps_list: List of ``num_layers`` tensors with shape [B, H, W].

    Returns:
        fused_map: Tensor with shape [B, H, W].
        weights: Input-dependent layer weights with shape [B, num_layers].
    """

    def __init__(
        self,
        num_layers,
        hidden_dim=32,
        topk_ratio=0.01,
        keep_sum_scale=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.topk_ratio = topk_ratio
        self.keep_sum_scale = keep_sum_scale

        self.weight_net = nn.Sequential(
            nn.Linear(num_layers * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_layers),
        )

    def forward(self, anomaly_maps_list):
        if not anomaly_maps_list:
            raise ValueError("anomaly_maps_list is empty")

        # List of [B, H, W] -> [B, L, H, W]
        maps = torch.stack(anomaly_maps_list, dim=1)
        batch_size, num_layers, _, _ = maps.shape

        if num_layers != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} anomaly maps, "
                f"but got {num_layers}."
            )

        flat_maps = maps.flatten(2)  # [B, L, H*W]

        mean_feat = flat_maps.mean(dim=-1)
        max_feat = flat_maps.amax(dim=-1)
        std_feat = flat_maps.std(dim=-1, unbiased=False)

        topk_count = max(1, int(flat_maps.shape[-1] * self.topk_ratio))
        topk_feat = flat_maps.topk(topk_count, dim=-1).values.mean(dim=-1)

        fusion_descriptor = torch.cat(
            [mean_feat, max_feat, std_feat, topk_feat],
            dim=1,
        )  # [B, 4L]

        weight_logits = self.weight_net(fusion_descriptor)
        weights = F.softmax(weight_logits, dim=1)  # [B, L]

        fused_map = (
            maps * weights.view(batch_size, num_layers, 1, 1)
        ).sum(dim=1)

        if self.keep_sum_scale:
            fused_map = fused_map * num_layers

        return fused_map, weights


def build_amrf(
    num_layers,
    hidden_dim=32,
    topk_ratio=0.01,
    keep_sum_scale=True,
):
    """Build the AMRF module."""
    return AdaptiveMultiLevelResponseFusion(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        topk_ratio=topk_ratio,
        keep_sum_scale=keep_sum_scale,
    )
