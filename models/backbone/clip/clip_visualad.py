import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP_FeatureExtractor(nn.Module):
    def __init__(
        self,
        layers=[5, 11, 17, 23],
        device="cuda",
        weight_path="models/backbone/clip/weights/ViT-L-14-336px.pt",
    ):
        super().__init__()
        self.device = device
        self.layers = layers
        self.weight_path = weight_path

        print(f"Loading CLIP ViT-L/14@336px from local weight: {weight_path}")

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"CLIP weight not found: {weight_path}")

        # 复用 VisualAD_lib 自带的 CLIP 加载逻辑，不依赖 openai clip 包
        import VisualAD_lib.model_load as model_load

        if not hasattr(model_load, "load"):
            raise AttributeError(
                "VisualAD_lib.model_load has no function named 'load'. "
                "Please check VisualAD_lib/model_load.py."
            )

        # 大多数 VisualAD/OpenAI-CLIP 风格 load 支持直接传本地 .pt 路径
        loaded = model_load.load(
            weight_path,
            device=device,
            jit=False
        )

        # 兼容返回 model 或 (model, preprocess)
        if isinstance(loaded, tuple):
            self.model = loaded[0]
        else:
            self.model = loaded

        self.model = self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.intermediate_features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        blocks = self.model.visual.transformer.resblocks

        for layer_idx in self.layers:
            hook = blocks[layer_idx].register_forward_hook(
                self._get_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            feat = output[0] if isinstance(output, tuple) else output
            self.intermediate_features[layer_idx] = feat.detach()
        return hook
    
    def _interpolate_positional_embedding(self, x, grid_h, grid_w):
        """
        x: [B, 1 + grid_h * grid_w, C]

        CLIP ViT-L/14@336px original:
            1 + 24 * 24 = 577 tokens

        Input 518:
            1 + 37 * 37 = 1370 tokens
        """
        visual = self.model.visual

        if hasattr(visual, "positional_embedding"):
            pos_embed = visual.positional_embedding
        elif hasattr(visual, "positional_embedding_frozen"):
            pos_embed = visual.positional_embedding_frozen
        else:
            raise AttributeError(
                "visual has neither positional_embedding nor positional_embedding_frozen"
            )

        # 兼容 [1+N, C] 和 [1, 1+N, C]
        if pos_embed.dim() == 3:
            pos_embed = pos_embed.squeeze(0)

        cls_pos = pos_embed[:1]      # [1, C]
        patch_pos = pos_embed[1:]    # [N, C]

        old_grid = int(math.sqrt(patch_pos.shape[0]))

        if old_grid * old_grid != patch_pos.shape[0]:
            raise RuntimeError(
                f"Cannot infer square position grid from pos_embed shape: {pos_embed.shape}"
            )

        if old_grid == grid_h and old_grid == grid_w:
            new_pos_embed = pos_embed
        else:
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)

            patch_pos = F.interpolate(
                patch_pos.float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False
            )

            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
            patch_pos = patch_pos.to(dtype=pos_embed.dtype, device=pos_embed.device)

            new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)

        return new_pos_embed.to(dtype=x.dtype, device=x.device)

    def _visual_forward_with_pos_interp(self, x):
        """
        VisualAD CLIP visual forward with positional embedding interpolation.
        Supports 518x518 input.
        """
        visual = self.model.visual

        conv_dtype = visual.conv1.weight.dtype
        x = x.to(dtype=conv_dtype)

        # [B, 3, H, W] -> [B, C, H/14, W/14]
        x = visual.conv1(x)

        B, C, grid_h, grid_w = x.shape

        # [B, C, H', W'] -> [B, N, C]
        x = x.reshape(B, C, grid_h * grid_w)
        x = x.permute(0, 2, 1)

        # class token
        cls_token = visual.class_embedding.to(dtype=x.dtype, device=x.device)
        cls_token = cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N, C]

        # position embedding interpolation
        pos_embed = self._interpolate_positional_embedding(x, grid_h, grid_w)
        x = x + pos_embed.unsqueeze(0)

        x = visual.ln_pre(x)

       # Transformer expects [Seq, B, C]
        x = x.permute(1, 0, 2)

        # VisualAD_lib.VisualAD.Transformer.forward 需要 out_layers 参数
        # 它内部可能用 1-based layer index，因此这里用 [6, 12, 18, 24] 这种形式
        visualad_out_layers = [l + 1 for l in self.layers]

        transformer_out = visual.transformer(x, visualad_out_layers)

        # 兼容 Transformer 返回 x 或 (x, features)
        if isinstance(transformer_out, tuple):
            x = transformer_out[0]
        else:
            x = transformer_out

        # [Seq, B, C] -> [B, Seq, C]
        x = x.permute(1, 0, 2)

        return x

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return:
            list of patch tokens, each [B, N, 1024]
        """
        self.intermediate_features.clear()
        batch_size = x.shape[0]

        device_type = "cuda" if x.is_cuda else "cpu"

        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=x.is_cuda):
            _ = self._visual_forward_with_pos_interp(x)

        processed_feats = []

        for layer_idx in self.layers:
            if layer_idx not in self.intermediate_features:
                raise RuntimeError(
                    f"Layer {layer_idx} feature not captured. "
                    f"Captured layers: {list(self.intermediate_features.keys())}"
                )

            feat = self.intermediate_features[layer_idx]

            if feat.shape[0] == batch_size:
                pass
            elif feat.shape[1] == batch_size:
                feat = feat.transpose(0, 1)
            else:
                raise RuntimeError(f"Unexpected CLIP feature shape: {feat.shape}")

            # remove CLS token
            patch_tokens = feat[:, 1:, :].contiguous().float()
            processed_feats.append(patch_tokens)

        return processed_feats

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []