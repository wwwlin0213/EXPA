"""Utilities for counting the parameters used by EXPA."""


def _unique_numel(parameters):
    """Count parameters without double-counting shared tensors."""
    unique_parameters = {}
    for parameter in parameters:
        if parameter is not None:
            unique_parameters[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in unique_parameters.values())


def _active_visual_encoder(backbone_model):
    """Return the visual encoder used during EXPA inference."""
    if (
        hasattr(backbone_model, "model")
        and hasattr(backbone_model.model, "visual")
    ):
        return backbone_model.model.visual

    if hasattr(backbone_model, "visual"):
        return backbone_model.visual

    return backbone_model


def count_total_parameters(
    backbone_model,
    layer_transforms=None,
    cepa=None,
    amrf=None,
    anomaly_token=None,
    normal_token=None,
):
    """Count the total parameters used by the EXPA inference pipeline.

    The count includes the active visual encoder, normal and abnormal
    prototypes, layer projections, CEPA, and AMRF. Parameters belonging to
    unused wrapper components, such as a text encoder, are excluded.
    Shared tensors are counted only once.
    """
    visual_encoder = _active_visual_encoder(backbone_model)

    parameter_groups = [visual_encoder.parameters()]

    if anomaly_token is not None:
        parameter_groups.append([anomaly_token])
    if normal_token is not None:
        parameter_groups.append([normal_token])
    if layer_transforms is not None:
        parameter_groups.append(layer_transforms.parameters())
    if cepa is not None:
        parameter_groups.append(cepa.parameters())
    if amrf is not None:
        parameter_groups.append(amrf.parameters())

    return _unique_numel(
        parameter
        for group in parameter_groups
        for parameter in group
    )