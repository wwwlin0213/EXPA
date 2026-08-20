"""Context-aware external prototype adaptation.

The module adapts normal and abnormal prototypes with multi-level patch
features. Patch features first undergo four-direction spatial shifting and are
then combined with the two external prototypes through bidirectional weighted
key-value aggregation. Only the prototypes are updated; the input patch
features are returned unchanged.
"""

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# -----------------------------------------------------------------------------
# Bi-WKV CUDA extension
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_dir = os.path.join(current_dir, "cuda")

# The target CUDA architecture is inferred from the active PyTorch environment.
wkv_cuda = load(
    name="bi_wkv",
    sources=[
        os.path.join(cuda_dir, "bi_wkv.cpp"),
        os.path.join(cuda_dir, "bi_wkv_kernel.cu")
    ],
    verbose=True, 
    extra_cuda_cflags=[
        '-res-usage',
        '--maxrregcount', '60',
        '--use_fast_math',
        '-O3',
        '-Xptxas', '-O3',
    ]
)
print("[CEPA] Bi-WKV CUDA extension loaded.")

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        ctx.save_for_backward(w, u, k, v)
        w, u, k, v = [t.float().contiguous() for t in (w, u, k, v)]
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        return y.half() if half_mode else y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(
            w.float().contiguous(), u.float().contiguous(),
            k.float().contiguous(), v.float().contiguous(), gy.float().contiguous()
        )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        return (gw, gu, gk, gv)

def run_bidirectional_wkv(w, u, k, v):
    # Launch the extension on the device that stores the input tensors.
    device = k.device
    
    with torch.cuda.device(device):
        return WKV.apply(w.contiguous(), u.contiguous(), k.contiguous(), v.contiguous())

# -----------------------------------------------------------------------------
# Four-direction spatial shift
# -----------------------------------------------------------------------------
def q_shift(input_tensor, shift_pixel=1, gamma=0.25):
    """Apply four-direction shifts to channel groups of patch features."""
    assert gamma <= 0.25
    B, N, C = input_tensor.shape
    H = int(math.sqrt(N))
    W = H
    
    # Convert the token sequence to a spatial feature map.
    x_2d = input_tensor.transpose(1, 2).contiguous().view(B, C, H, W)
    
    c1, c2, c3, c4 = int(C*gamma), int(C*gamma*2), int(C*gamma*3), int(C*gamma*4)
    
    # Shift the first channel group to the right.
    c_right = F.pad(x_2d[:, 0:c1, :, :], (shift_pixel, 0, 0, 0))[:, :, :, :-shift_pixel]
    # Shift the second channel group to the left.
    c_left  = F.pad(x_2d[:, c1:c2, :, :], (0, shift_pixel, 0, 0))[:, :, :, shift_pixel:]
    # Shift the third channel group downward.
    c_down  = F.pad(x_2d[:, c2:c3, :, :], (0, 0, shift_pixel, 0))[:, :, :-shift_pixel, :]
    # Shift the fourth channel group upward.
    c_up    = F.pad(x_2d[:, c3:c4, :, :], (0, 0, 0, shift_pixel))[:, :, shift_pixel:, :]
    
    # Concatenate shifted and unshifted channel groups.
    if c4 < C:
        c_keep = x_2d[:, c4:, :, :]
        out_2d = torch.cat([c_right, c_left, c_down, c_up, c_keep], dim=1)
    else:
        out_2d = torch.cat([c_right, c_left, c_down, c_up], dim=1)
        
    return out_2d.flatten(2).transpose(1, 2).contiguous()

# -----------------------------------------------------------------------------
# Per-layer external prototype adaptation
# -----------------------------------------------------------------------------
class CEPALayer(nn.Module):
    """Adapt external prototypes with patch features from one encoder layer."""

    def __init__(self, embed_dim, res_scale_init=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable interpolation factors for shifted and original features.
        self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        
        # Channel-wise decay and first-position parameters.
        self.spatial_decay = nn.Parameter(torch.zeros(embed_dim))
        self.spatial_first = nn.Parameter(torch.zeros(embed_dim))

        # Key, value, gate, and output projections.
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

        self.res_scale = nn.Parameter(torch.ones(1) * res_scale_init)

    def forward(self, anomaly_token, normal_token, patch_tokens):
        """
        anomaly_token: [B, 1, C]
        normal_token: [B, 1, C]
        patch_tokens: [B, N, C], where N = H_p × W_p
        """
        B, N, C = patch_tokens.shape
        
        # Apply four-direction spatial shifting to patch features.
        shifted_patches = q_shift(patch_tokens, shift_pixel=1, gamma=0.25)
        
        # Interpolate original and shifted patch features.
        xk_p = patch_tokens * self.spatial_mix_k + shifted_patches * (1 - self.spatial_mix_k)
        xv_p = patch_tokens * self.spatial_mix_v + shifted_patches * (1 - self.spatial_mix_v)
        xr_p = patch_tokens * self.spatial_mix_r + shifted_patches * (1 - self.spatial_mix_r)
        
        # Prepend the abnormal and normal prototypes to each feature sequence.
        xk = torch.cat([anomaly_token, normal_token, xk_p], dim=1)
        xv = torch.cat([anomaly_token, normal_token, xv_p], dim=1)
        xr = torch.cat([anomaly_token, normal_token, xr_p], dim=1)
        
        T_total = xk.size(1)
        
        # Generate key, value, and gate representations.
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        # Aggregate context with the bidirectional WKV operator.
        context_features = run_bidirectional_wkv(self.spatial_decay / T_total, self.spatial_first / T_total, k, v)
        
        # Apply gating and output projection.
        context_features = sr * context_features
        context_features = self.output(context_features)
        
        # Update only the two external prototypes through residual adaptation.
        enhanced_anomaly = anomaly_token + self.res_scale * context_features[:, 0:1, :]
        enhanced_normal  = normal_token + self.res_scale * context_features[:, 1:2, :]

        return enhanced_anomaly, enhanced_normal, patch_tokens

class ContextAwareExternalPrototypeAdapter(nn.Module):
    """Apply independent CEPA layers to selected multi-level features."""

    def __init__(self, layers, embed_dim, res_scale_init=0.1):
        super().__init__()
        self.layers = layers
        self.layer_blocks = nn.ModuleDict()
        
        for layer_id in layers:
            self.layer_blocks[f'layer_{layer_id}'] = CEPALayer(
                embed_dim=embed_dim,
                res_scale_init=res_scale_init
            )

    def forward(self, anomaly_token, normal_token, patch_tokens_list, layer_ids):
        B = anomaly_token.shape[0]
        # Convert prototype tensors to [B, 1, C] when necessary.
        if anomaly_token.dim() == 2:
            anomaly_token = anomaly_token.unsqueeze(1)
            normal_token = normal_token.unsqueeze(1)
            
        enhanced_features = []
        for idx, (layer_id, patch_tokens) in enumerate(zip(layer_ids, patch_tokens_list)):
            layer_key = f'layer_{layer_id}'
            cepa_layer = self.layer_blocks[layer_key]
            
            # Adapt the two prototypes with features from the current layer.
            e_anomaly, e_normal, e_patches = cepa_layer(anomaly_token, normal_token, patch_tokens)

            # Return [B, C] prototypes for downstream similarity computation.
            enhanced_features.append({
                'anomaly': e_anomaly.squeeze(1),
                'normal': e_normal.squeeze(1),
                'patches': e_patches  # Patch features remain unchanged.
            })

        return enhanced_features

def build_cepa(layers, embed_dim, res_scale_init=0.1):
    return ContextAwareExternalPrototypeAdapter(
        layers=layers,
        embed_dim=embed_dim,
        res_scale_init=res_scale_init
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Testing the CEPA module with the CUDA extension")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_cepa(layers=[6, 12, 18, 24], embed_dim=1024).to(device)
    
    # 模拟数据
    anomaly_token = torch.randn(4, 1024).to(device)
    normal_token = torch.randn(4, 1024).to(device)
    patch_tokens_list = [torch.randn(4, 576, 1024).to(device) for _ in range(4)]
    
    out = model(anomaly_token, normal_token, patch_tokens_list, [6,12,18,24])
    print("Forward pass completed successfully.")
    print(f"Abnormal prototype shape: {out[-1]['anomaly'].shape}")