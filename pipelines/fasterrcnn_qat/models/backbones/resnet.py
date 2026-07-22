"""ResNet backbone adapters for Faster R-CNN + FPN."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class ResNetStem(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNetFPNBackbone(nn.Module):
    """Expose ResNet stages as C2-C5 and return P2-P6 in an OrderedDict."""

    def __init__(self, variant="resnet50", out_channels=256, pretrained=True, trainable_layers=5):
        super().__init__()
        if variant != "resnet50":
            raise ValueError("Only resnet50 is currently supported for ResNet FPN")
        if not 0 <= trainable_layers <= 5:
            raise ValueError("trainable_backbone_layers must be between 0 and 5 for ResNet")

        network = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(
            ResNetStem(network),
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        stage_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(stage_channels, out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels
        self.feature_indices = (1, 2, 3, 4)
        self.pt2e_spatial_divisor = 32
        self.pt2e_region_kind = "resnet50"

        stage_indices = tuple(range(len(self.body)))
        trainable_stage_indices = set(stage_indices[-trainable_layers:]) if trainable_layers else set()
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
