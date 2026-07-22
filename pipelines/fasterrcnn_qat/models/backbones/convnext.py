"""ConvNeXt backbone adapters for Faster R-CNN + FPN."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_small,
    convnext_tiny,
)
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


VARIANTS = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, [96, 192, 384, 768]),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, [96, 192, 384, 768]),
}


class ConvNeXtFPNBackbone(nn.Module):
    """Expose ConvNeXt stages as C2-C5 and return P2-P6 in an OrderedDict."""

    def __init__(self, variant="convnext_tiny", out_channels=256, pretrained=True, trainable_layers=4):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported backbone {variant!r}; choose {sorted(VARIANTS)}")
        if not 0 <= trainable_layers <= 4:
            raise ValueError("trainable_backbone_layers must be between 0 and 4")

        constructor, default_weights, stage_channels = VARIANTS[variant]
        network = constructor(weights=default_weights if pretrained else None)
        self.body = network.features
        self.fpn = FeaturePyramidNetwork(stage_channels, out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels
        self.feature_indices = (1, 3, 5, 7)
        self.pt2e_spatial_divisor = 32
        self.pt2e_region_kind = "convnext"

        stage_groups = ((0, 1), (2, 3), (4, 5), (6, 7))
        trainable_stage_indices = {
            index
            for group in stage_groups[-trainable_layers:] if trainable_layers
            for index in group
        }
        for index, child in enumerate(self.body):
            requires_grad = index in trainable_stage_indices
            for parameter in child.parameters():
                parameter.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor):
        features = OrderedDict()
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                features[str(len(features))] = x
        return self.fpn(features)
