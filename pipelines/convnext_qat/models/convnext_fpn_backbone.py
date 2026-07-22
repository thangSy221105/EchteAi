"""Compatibility shim for the legacy convnext_qat namespace."""

from ...fasterrcnn_qat.models.backbones import (  # noqa: F401
    ConvNeXtFPNBackbone,
    ResNetFPNBackbone,
    build_convnext_fpn_backbone,
    build_fpn_backbone,
)
