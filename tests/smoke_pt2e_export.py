"""Smoke-check the tensor-only backbone PT2E export boundary without TorchAO."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.convnext_qat.models.convnext_fpn_backbone import ConvNeXtFPNBackbone, ResNetFPNBackbone
from pipelines.convnext_qat.quantization.pt2e_qat import BackboneBodyRegion, ResNet50BodyRegion, _dynamic_shapes


def run_one(backbone, expected_channels, name):
    if getattr(backbone, "pt2e_region_kind", "") == "resnet50":
        region = ResNet50BodyRegion(backbone.body).eval()
    else:
        region = BackboneBodyRegion(backbone.body, backbone.feature_indices).eval()
    example = torch.randn(2, 3, 256, 320)
    exported = torch.export.export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(2, 2, 224, 1024),
    ).module()
    outputs = exported(torch.randn(1, 3, 288, 352))
    assert len(outputs) == 4
    assert [tensor.shape[1] for tensor in outputs] == expected_channels
    print(f"PT2E dynamic {name} export smoke test passed")


def main():
    run_one(ConvNeXtFPNBackbone(pretrained=False), [96, 192, 384, 768], "ConvNeXt")
    run_one(ResNetFPNBackbone(pretrained=False), [256, 512, 1024, 2048], "ResNet50")


if __name__ == "__main__":
    main()
