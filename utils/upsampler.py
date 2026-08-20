import torch
import torch.nn as nn

class FeatureGuidedUpsampler(nn.Module):
    def __init__(self, in_channels=1024, proj_channels=64):
        super().__init__()
        # 1024维 Patch 特征 + 1维低分辨率异常图 = 1025 维输入
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + 1, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True)
        )
        # 放大 14 倍 (因为 336 / 24 = 14)
        self.upsample = nn.ConvTranspose2d(
            proj_channels, 1, 
            kernel_size=14, stride=14
        )

    def forward(self, patch_features, low_res_score):
        """
        patch_features: [B, 576, 1024]
        low_res_score: [B, 24, 24]
        """
        B, N, C = patch_features.shape
        H = W = int(N ** 0.5) # 24
        
        # 转换维度: [B, 1024, 24, 24]
        x = patch_features.transpose(1, 2).view(B, C, H, W)
        score = low_res_score.unsqueeze(1) # [B, 1, 24, 24]
        
        # 拼接在一起: [B, 1025, 24, 24]
        x = torch.cat([x, score], dim=1)
        x = self.proj(x)
        
        # 高清放大: [B, 1, 336, 336]
        high_res_map = self.upsample(x)
        return high_res_map.squeeze(1)