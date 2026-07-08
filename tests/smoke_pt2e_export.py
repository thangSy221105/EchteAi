"""Smoke-check the tensor-only ConvNeXt PT2E export boundary without TorchAO."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.convnext_qat.models.convnext_fpn_backbone import ConvNeXtFPNBackbone
from pipelines.convnext_qat.quantization.pt2e_qat import ConvNeXtBodyRegion, _dynamic_shapes


def main():
    backbone = ConvNeXtFPNBackbone(pretrained=False)
    region = ConvNeXtBodyRegion(backbone.body).eval()
    example = torch.randn(2, 3, 256, 320)
    exported = torch.export.export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(2, 2, 224, 1024),
    ).module()
    outputs = exported(torch.randn(1, 3, 288, 352))
    assert len(outputs) == 4
    assert [tensor.shape[1] for tensor in outputs] == [96, 192, 384, 768]
    print("PT2E dynamic ConvNeXt export smoke test passed")


if __name__ == "__main__":
    main()
