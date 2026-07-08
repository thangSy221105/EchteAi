"""Graph-mode PT2E QAT for the ConvNeXt body with an FP32 FPN/head boundary."""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
from torch import nn
from torch.export import Dim, export


def _torchao_pt2e():
    try:
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
        from torchao.quantization.pt2e.quantizer import x86_inductor_quantizer as xiq
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
        from torchao.quantization.pt2e import move_exported_model_to_eval
    except ImportError as error:
        raise RuntimeError(
            "PT2E QAT requires torchao. Install the optional dependency with "
            "`pip install torchao`."
        ) from error
    return prepare_qat_pt2e, convert_pt2e, X86InductorQuantizer, xiq, move_exported_model_to_eval


class ConvNeXtBodyRegion(nn.Module):
    """Tensor-only export boundary returning C2-C5 as a stable tuple."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in (1, 3, 5, 7):
                outputs.append(x)
        return tuple(outputs)


class ConvNeXtBodyFPNRegion(nn.Module):
    """Optional second PT2E scope covering ConvNeXt and the complete FPN."""

    def __init__(self, body, fpn):
        super().__init__()
        self.body = ConvNeXtBodyRegion(body)
        self.fpn = fpn

    def forward(self, x):
        tensors = self.body(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return tuple(self.fpn(features).values())


class PT2EConvNeXtFPNBackbone(nn.Module):
    """PT2E ConvNeXt graph followed by the original FP32 torchvision FPN."""

    def __init__(self, body_region, fpn, out_channels):
        super().__init__()
        self.body_region = body_region
        self.fpn = fpn
        self.out_channels = int(out_channels)

    def train(self, mode: bool = True):
        # Exported GraphModules reject .train()/.eval(). Their ConvNeXt graph is
        # captured in deterministic eval form; QAT fake-quant nodes still update
        # during forward. Only the ordinary FPN needs recursive mode switching.
        self.training = mode
        if self.fpn is not None:
            self.fpn.train(mode)
        return self

    def forward(self, x):
        tensors = self.body_region(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return self.fpn(features) if self.fpn is not None else features


def _dynamic_shapes(example_batch_size, maximum_batch_size, minimum_side, maximum_side):
    # ConvNeXt downsamples by 32. Expressing this relation explicitly avoids
    # torch.export constraint violations while retaining variable image shapes.
    height = 32 * Dim("image_h_div_32", min=max(1, minimum_side // 32), max=maximum_side // 32)
    width = 32 * Dim("image_w_div_32", min=max(1, minimum_side // 32), max=maximum_side // 32)
    if maximum_batch_size > 1 and example_batch_size == 1:
        raise ValueError(
            "Dynamic batch export needs pt2e.example_batch_size >= 2; "
            "PyTorch specializes an example batch of one"
        )
    batch = Dim.STATIC if maximum_batch_size == 1 else Dim(
        "image_batch", min=1, max=maximum_batch_size,
    )
    return ({0: batch, 2: height, 3: width},)


def prepare_pt2e_backbone_qat(model, config, inplace=False):
    """Replace only ConvNeXt body with a prepared x86 PT2E QAT graph."""
    prepared_model = model if inplace else copy.deepcopy(model)
    if isinstance(prepared_model.backbone, PT2EConvNeXtFPNBackbone):
        return prepared_model
    pt2e = config.get("quantization", {}).get("pt2e", {})
    scope = str(pt2e.get("region", "backbone")).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("quantization.pt2e.region must be backbone or backbone_fpn")
    example_batch = int(pt2e.get("example_batch_size", 1))
    maximum_batch = int(pt2e.get("maximum_batch_size", config["training"].get("qat_batch_size", 1)))
    minimum_side = int(pt2e.get("minimum_image_side", 256))
    maximum_side = int(pt2e.get("maximum_image_side", config["model"].get("max_size", 1600)))
    example_height = int(pt2e.get("example_height", min(960, maximum_side)))
    example_width = int(pt2e.get("example_width", min(1280, maximum_side)))
    if any(value <= 0 or value % 32 for value in (minimum_side, maximum_side, example_height, example_width)):
        raise ValueError("PT2E image dimensions must be positive multiples of 32")
    if maximum_batch < example_batch:
        raise ValueError("pt2e.maximum_batch_size must be >= example_batch_size")

    backbone = prepared_model.backbone
    region = (
        ConvNeXtBodyRegion(backbone.body)
        if scope == "backbone"
        else ConvNeXtBodyFPNRegion(backbone.body, backbone.fpn)
    ).cpu().eval()
    example = torch.randn(example_batch, 3, example_height, example_width)
    exported = export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(
            example_batch, maximum_batch, minimum_side, maximum_side,
        ),
    ).module()
    prepare_qat_pt2e, _, quantizer_type, xiq, _ = _torchao_pt2e()
    quantizer = quantizer_type()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
    prepared_region = prepare_qat_pt2e(exported, quantizer)
    prepared_model.backbone = PT2EConvNeXtFPNBackbone(
        prepared_region, backbone.fpn if scope == "backbone" else None, backbone.out_channels,
    )
    prepared_model.pt2e_quantized_region = (
        "backbone.body" if scope == "backbone" else "backbone.body+fpn"
    )
    return prepared_model


def convert_pt2e_backbone(model, inplace=False, compile_region=False):
    """Convert the prepared ConvNeXt graph and optionally compile that graph."""
    converted_model = model if inplace else copy.deepcopy(model)
    if not isinstance(converted_model.backbone, PT2EConvNeXtFPNBackbone):
        raise TypeError("Model has not been prepared by prepare_pt2e_backbone_qat")
    _, convert_pt2e, _, _, move_to_eval = _torchao_pt2e()
    converted_model.cpu()
    converted_region = convert_pt2e(converted_model.backbone.body_region)
    move_to_eval(converted_region)
    if compile_region:
        converted_region = torch.compile(converted_region)
    converted_model.backbone.body_region = converted_region
    converted_model.eval()
    converted_model.pt2e_compiled = bool(compile_region)
    return converted_model
