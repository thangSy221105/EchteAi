"""Compiler-facing adapters for ResNet50 feature extraction.

These modules deliberately avoid detector control flow. Their job is to expose
stable tensor-only interfaces that a compiler backend can own independently of
RPN / ROI / NMS integration.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class BackboneBodyAdapter(nn.Module):
    """Return ResNet C2-C5 as a tuple of tensors."""

    def __init__(self, backbone):
        super().__init__()
        self.body = backbone.body
        self.feature_indices = tuple(getattr(backbone, "feature_indices", (1, 2, 3, 4)))

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                outputs.append(x)
        return tuple(outputs)


class BackboneBodyFPNAdapter(nn.Module):
    """Return FPN P2-P6 as a tuple of tensors."""

    def __init__(self, backbone):
        super().__init__()
        self.body = BackboneBodyAdapter(backbone)
        self.fpn = backbone.fpn

    def forward(self, x):
        c_features = self.body(x)
        feature_dict = OrderedDict((str(index), tensor) for index, tensor in enumerate(c_features))
        outputs = self.fpn(feature_dict)
        return tuple(outputs.values())


def resolve_compiler_scope(config):
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = str(compiler_cfg.get("scope", "backbone")).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("quantization.compiler.scope must be backbone or backbone_fpn")
    return scope


def build_compiler_target_module(model, config):
    scope = resolve_compiler_scope(config)
    backbone = model.backbone
    if str(getattr(backbone, "pt2e_region_kind", "")).lower() != "resnet50":
        raise ValueError("Compiler-first branch currently supports only ResNet50 backbones")
    return BackboneBodyAdapter(backbone) if scope == "backbone" else BackboneBodyFPNAdapter(backbone)
