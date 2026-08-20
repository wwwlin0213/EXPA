import os
import torch
import torch.nn as nn


DINOV2_ZOO = {
    "DINOv2-S/14": {
        "hub_name": "dinov2_vits14",
        "embed_dim": 384,
        "layers": 12,
        "patch_size": 14,
        "weight_name": "dinov2_vits14_pretrain.pth",
    },
    "DINOv2-B/14": {
        "hub_name": "dinov2_vitb14",
        "embed_dim": 768,
        "layers": 12,
        "patch_size": 14,
        "weight_name": "dinov2_vitb14_pretrain.pth",
    },
    "DINOv2-L/14": {
        "hub_name": "dinov2_vitl14",
        "embed_dim": 1024,
        "layers": 24,
        "patch_size": 14,
        "weight_name": "dinov2_vitl14_pretrain.pth",
    },
    "DINOv2-g/14": {
        "hub_name": "dinov2_vitg14",
        "embed_dim": 1536,
        "layers": 40,
        "patch_size": 14,
        "weight_name": "dinov2_vitg14_pretrain.pth",
    },
}


BACKBONE_ALIASES = {
    "dinov2_vits14": "DINOv2-S/14",
    "dinov2_vitb14": "DINOv2-B/14",
    "dinov2_vitl14": "DINOv2-L/14",
    "dinov2_vitg14": "DINOv2-g/14",

    "dinov2-s/14": "DINOv2-S/14",
    "dinov2-b/14": "DINOv2-B/14",
    "dinov2-l/14": "DINOv2-L/14",
    "dinov2-g/14": "DINOv2-g/14",

    "dinov2_s14": "DINOv2-S/14",
    "dinov2_b14": "DINOv2-B/14",
    "dinov2_l14": "DINOv2-L/14",
    "dinov2_g14": "DINOv2-g/14",
}


class DINOv2_FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name="DINOv2-L/14",
        layers=[5, 11, 17, 23],
        device="cuda",
        repo_path=None,
        weight_path=None,
    ):
        """
        layers: actual 0-based block indices.
                For features_list [6,12,18,24], pass [5,11,17,23].

        return:
            list of patch tokens, each [B, N, C]
        """
        super().__init__()

        canonical_name = self._normalize_backbone_name(backbone_name)

        if canonical_name not in DINOV2_ZOO:
            raise RuntimeError(
                f"Unsupported DINOv2 backbone: {backbone_name}. "
                f"Available: {list(DINOV2_ZOO.keys())}"
            )

        cfg = DINOV2_ZOO[canonical_name]

        self.backbone_name = canonical_name
        self.layers = layers
        self.embed_dim = cfg["embed_dim"]
        self.num_layers = cfg["layers"]
        self.patch_size = cfg["patch_size"]
        self.device = device

        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    f"{self.backbone_name} has {self.num_layers} layers, "
                    f"but got 0-based layer index {layer_idx}."
                )

        base_dir = os.path.dirname(os.path.abspath(__file__))

        default_repo = os.path.join(base_dir, "official")
        default_weight = os.path.join(base_dir, "weights", cfg["weight_name"])

        repo_path = repo_path or os.environ.get("DINOV2_REPO", default_repo)
        weight_path = weight_path or os.environ.get("DINOV2_WEIGHT", default_weight)

        if not os.path.isdir(repo_path):
            raise FileNotFoundError(
                f"[DINOv2] Local repo not found: {repo_path}\n"
                f"Expected official repo at models/backbone/dinov2/official"
            )

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"[DINOv2] Weight file not found: {weight_path}"
            )

        print(f"[DINOv2] Loading local repo: {repo_path}")
        print(f"[DINOv2] Loading local weight: {weight_path}")
        print(f"[DINOv2] Backbone: {self.backbone_name}")
        print(f"[DINOv2] Layers: {self.layers}")
        print(f"[DINOv2] Embed dim: {self.embed_dim}")

        self.dino = torch.hub.load(
            repo_path,
            cfg["hub_name"],
            source="local",
            pretrained=False,
        )

        state_dict = torch.load(weight_path, map_location="cpu")

        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            new_state_dict[k] = v

        msg = self.dino.load_state_dict(new_state_dict, strict=False)
        print(f"[DINOv2] load_state_dict msg: {msg}")

        self.dino = self.dino.to(device)
        self.dino.eval()

        for p in self.dino.parameters():
            p.requires_grad = False

    def _normalize_backbone_name(self, name):
        if name in DINOV2_ZOO:
            return name

        key = name.lower()
        if key in BACKBONE_ALIASES:
            return BACKBONE_ALIASES[key]

        return name

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return:
            processed_feats: list of [B, N, C]
        """
        with torch.no_grad():
            outs = self.dino.get_intermediate_layers(
                x,
                n=self.layers,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        processed_feats = []

        for out in outs:
            if isinstance(out, tuple):
                patch_tokens, cls_token = out
            else:
                patch_tokens = out

            # patch_tokens should be [B, N, C]
            if patch_tokens.dim() != 3:
                raise RuntimeError(
                    f"Unexpected DINOv2 patch token shape: {patch_tokens.shape}"
                )

            processed_feats.append(patch_tokens.float().contiguous())

        return processed_feats

    def remove_hooks(self):
        # DINOv2 uses get_intermediate_layers, no hooks needed.
        pass