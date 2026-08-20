import torch
import torch.nn as nn
import sys
import os

import models.backbone.pe.pe as pe

class PE_FeatureExtractor(nn.Module):
    def __init__(self, layers=[6, 12, 18], device='cuda'):
        """
        layers: 需要提取特征的网络层索引。
        PE-Core-L14 有 24 层 (0~23)。
        """
        super().__init__()
        self.device = device
        self.layers = layers
        
        print("PE module file:", pe.__file__)
        print("Loading PE Model from local weights...")
        self.model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        weight_path = os.path.join(
            project_root,
            "models",
            "backbone",
            "pe",
            "weights",
            "PE-Core-L14-336.pt"
        )

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"PE weight file not found: {weight_path}"
            )

        print(f"Loading PE weights from: {weight_path}")

        state_dict = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        
        # 彻底冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 注册前向钩子 (Forward Hooks)
        self.intermediate_features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # PE(CLIP) 的视觉 Transformer 块通常在这个路径下
        # 如果报错找不到 resblocks，请 print(self.model.visual) 查看具体结构名字
        blocks = self.model.visual.transformer.resblocks
        
        for layer_idx in self.layers:
            # 给指定的层挂上钩子
            hook = blocks[layer_idx].register_forward_hook(self._get_hook(layer_idx))
            self.hooks.append(hook)

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            # 截获输出并保存到字典中
            feat = output[0] if isinstance(output, tuple) else output
            self.intermediate_features[layer_idx] = feat.clone()
        return hook
        
    def forward(self, x):
        """
        x: [B, 3, H, W]
        return:
            list of patch features, each [B, N, C]
        """
        self.intermediate_features.clear()

        # 只运行一次视觉编码器
        max_layer = max(self.layers)

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            enabled=x.is_cuda
        ):
            _ = self.model.visual(x, layer_idx=max_layer)

        processed_feats = []
        batch_size = x.shape[0]

        for layer_idx in self.layers:
            feat = self.intermediate_features[layer_idx]

            # PE 的 Transformer 使用 batch_first=True，
            # 正常应为 [B, N, C]
            if feat.dim() != 3:
                raise RuntimeError(
                    f"Expected 3D feature at layer {layer_idx}, "
                    f"but got shape {tuple(feat.shape)}"
                )

            # 兼容可能返回 [N, B, C] 的其他实现
            if feat.shape[0] != batch_size and feat.shape[1] == batch_size:
                feat = feat.transpose(0, 1)

            if feat.shape[0] != batch_size:
                raise RuntimeError(
                    f"Unexpected feature shape at layer {layer_idx}: "
                    f"{tuple(feat.shape)}, batch size={batch_size}"
                )

            # 只有模型确实使用 CLS token 时，才删除序列第一个 token
            if self.model.visual.use_cls_token:
                feat = feat[:, 1:, :]

            processed_feats.append(feat.contiguous())

        return processed_feats

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []