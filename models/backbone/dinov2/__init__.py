from .dinov2_wrapper import DINOv2_FeatureExtractor, DINOV2_ZOO


def build_dinov2_visualad(name, device="cuda", download_root=None):
    """
    Compatibility wrapper for VisualAD_lib.model_load import.

    The current Ours framework does not use this function directly.
    We keep it here only because VisualAD_lib.model_load imports
    build_dinov2_visualad during initialization.
    """
    return DINOv2_FeatureExtractor(
        backbone_name=name,
        layers=[5, 11, 17, 23],
        device=device,
        repo_path="./models/backbone/dinov2/official",
        weight_path="./models/backbone/dinov2/weights/dinov2_vitl14_pretrain.pth",
    )