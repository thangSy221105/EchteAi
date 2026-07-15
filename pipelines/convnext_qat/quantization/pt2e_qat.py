"""Graph-mode PT2E QAT for the backbone body with an FP32 FPN/head boundary."""

from __future__ import annotations

import copy
import sysconfig
import warnings
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export


def _normalized_version(value):
    return tuple(int(part) for part in value.split("+", 1)[0].split(".")[:3])


def _collect_pt2e_fake_quantizers(model):
    required_methods = {
        "enable_observer", "disable_observer", "enable_fake_quant", "disable_fake_quant",
    }
    return [
        module for module in model.modules()
        if required_methods.issubset(dir(module))
    ]


def _torchao_pt2e():
    if _normalized_version(torch.__version__) < (2, 11, 0):
        raise RuntimeError(
            "PT2E QAT in this repo needs torch >= 2.11.0. "
            f"Found torch {torch.__version__}. "
            "Kaggle's default torch 2.10 image is too old for the torchao PT2E path here; "
            "upgrade torch/torchvision first, then reinstall torchao and rerun the notebook."
        )
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
    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        torchao_version = "unknown"
    if not torchao_version.startswith("0.17."):
        raise RuntimeError(
            f"This pipeline is validated with torchao 0.17.x, found {torchao_version}. "
            "Install `torchao==0.17.0`."
        )
    return prepare_qat_pt2e, convert_pt2e, X86InductorQuantizer, xiq, move_exported_model_to_eval


class BackboneBodyRegion(nn.Module):
    """Tensor-only export boundary returning C2-C5 as a stable tuple."""

    def __init__(self, body, feature_indices):
        super().__init__()
        self.body = body
        self.feature_indices = tuple(int(value) for value in feature_indices)

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                outputs.append(x)
        return tuple(outputs)


class ResNet50BodyRegion(nn.Module):
    """Explicit ResNet-50 export boundary with cloned C2-C5 outputs."""

    def __init__(self, body):
        super().__init__()
        self.stem = body[0]
        self.layer1 = body[1]
        self.layer2 = body[2]
        self.layer3 = body[3]
        self.layer4 = body[4]

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Clone stage outputs before returning them so the quantizer sees
        # single-use graph leaves rather than tensors that are both returned
        # and consumed by subsequent stages.
        return (c2.clone(), c3.clone(), c4.clone(), c5.clone())


class SingleTensorRegion(nn.Module):
    """Export a single module with one tensor input and one tensor output."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class BackboneBodyFPNRegion(nn.Module):
    """Optional second PT2E scope covering the backbone body and the complete FPN."""

    def __init__(self, body, fpn, feature_indices):
        super().__init__()
        self.body = BackboneBodyRegion(body, feature_indices)
        self.fpn = fpn

    def forward(self, x):
        tensors = self.body(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return tuple(self.fpn(features).values())


class ResNet50BodyFPNRegion(nn.Module):
    """Optional PT2E scope covering explicit ResNet-50 body and full FPN."""

    def __init__(self, body, fpn):
        super().__init__()
        self.body = ResNet50BodyRegion(body)
        self.fpn = fpn

    def forward(self, x):
        tensors = self.body(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return tuple(self.fpn(features).values())


def build_backbone_body_region(backbone, scope="backbone"):
    """Build the tensor-only PT2E export region for the selected backbone."""
    scope = str(scope).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("scope must be backbone or backbone_fpn")
    feature_indices = tuple(getattr(backbone, "feature_indices", (1, 3, 5, 7)))
    region_kind = str(getattr(backbone, "pt2e_region_kind", "convnext")).lower()
    if region_kind == "resnet50":
        return (
            ResNet50BodyRegion(backbone.body)
            if scope == "backbone"
            else ResNet50BodyFPNRegion(backbone.body, backbone.fpn)
        )
    return (
        BackboneBodyRegion(backbone.body, feature_indices)
        if scope == "backbone"
        else BackboneBodyFPNRegion(backbone.body, backbone.fpn, feature_indices)
    )


class PT2EBackboneFPN(nn.Module):
    """PT2E backbone graph followed by the original FP32 torchvision FPN."""

    def __init__(self, body_region, fpn, out_channels):
        super().__init__()
        self.body_region = body_region
        self.fpn = fpn
        self.out_channels = int(out_channels)

    def train(self, mode: bool = True):
        # Exported GraphModules reject .train()/.eval(). Their backbone graph is
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


class PT2EResNetBodyStages(nn.Module):
    """Stage-wise PT2E wrapper for ResNet50 to avoid multi-output partition issues."""

    def __init__(self, stem, layer1, layer2, layer3, layer4):
        super().__init__()
        self.stem = stem
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return (c2, c3, c4, c5)


def _dynamic_shapes(example_batch_size, maximum_batch_size, minimum_side, maximum_side,
                    spatial_divisor=32):
    # ConvNeXt downsamples by 32. Expressing this relation explicitly avoids
    # torch.export constraint violations while retaining variable image shapes.
    height = spatial_divisor * Dim(
        f"image_h_div_{spatial_divisor}", min=max(1, minimum_side // spatial_divisor),
        max=maximum_side // spatial_divisor,
    )
    width = spatial_divisor * Dim(
        f"image_w_div_{spatial_divisor}", min=max(1, minimum_side // spatial_divisor),
        max=maximum_side // spatial_divisor,
    )
    if maximum_batch_size > 1 and example_batch_size == 1:
        raise ValueError(
            "Dynamic batch export needs pt2e.example_batch_size >= 2; "
            "PyTorch specializes an example batch of one"
        )
    batch = Dim.STATIC if maximum_batch_size == 1 else Dim(
        "image_batch", min=1, max=maximum_batch_size,
    )
    return ({0: batch, 2: height, 3: width},)


def _export_single_region(region, example, example_batch, maximum_batch,
                          minimum_side, maximum_side, spatial_divisor):
    return export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(
            example_batch, maximum_batch, minimum_side, maximum_side, spatial_divisor,
        ),
    ).module()


def _export_single_region_static(region, example):
    """Export with fixed shape to avoid stage-specific dynamic guard blowups."""
    return export(region, (example,)).module()


def _prepare_resnet50_stagewise_qat(backbone, config):
    pt2e = config.get("quantization", {}).get("pt2e", {})
    example_batch = int(pt2e.get("example_batch_size", 1))
    maximum_batch = int(pt2e.get("maximum_batch_size", config["training"].get("qat_batch_size", 1)))
    minimum_side = int(pt2e.get("minimum_image_side", 256))
    maximum_side = int(pt2e.get("maximum_image_side", config["model"].get("max_size", 1600)))
    example_height = int(pt2e.get("example_height", min(960, maximum_side)))
    example_width = int(pt2e.get("example_width", min(1280, maximum_side)))

    prepare_qat_pt2e, _, quantizer_type, xiq, _ = _torchao_pt2e()

    def make_quantizer():
        quantizer = quantizer_type()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
        return quantizer

    def prepare_stage(module, channels, input_divisor):
        if any(value <= 0 or value % input_divisor for value in (minimum_side, maximum_side, example_height, example_width)):
            raise ValueError(
                f"PT2E ResNet50 stage input dimensions must be positive multiples of {input_divisor}"
            )
        example = torch.randn(
            example_batch,
            channels,
            example_height // input_divisor,
            example_width // input_divisor,
        )
        # ResNet stem/downsample stages generate export guards that are much
        # stricter than the generic image-size range. For the first graphized
        # ResNet50 PT2E branch, prefer a static-shape export so we can validate
        # graph-mode quantization before re-introducing dynamic-shape support.
        exported = _export_single_region_static(SingleTensorRegion(module).cpu().eval(), example)
        prepared = prepare_qat_pt2e(exported, make_quantizer())
        fake_quantizers = _collect_pt2e_fake_quantizers(prepared)
        if not fake_quantizers:
            raise RuntimeError(
                "prepare_qat_pt2e completed but inserted no PT2E fake-quant modules for a ResNet50 stage. "
                f"torch={torch.__version__}. Check torch/torchao compatibility."
            )
        return prepared

    prepared_stem = prepare_stage(backbone.body[0], 3, 1)
    prepared_layer1 = prepare_stage(backbone.body[1], 64, 4)
    prepared_layer2 = prepare_stage(backbone.body[2], 256, 4)
    prepared_layer3 = prepare_stage(backbone.body[3], 512, 8)
    prepared_layer4 = prepare_stage(backbone.body[4], 1024, 16)
    return PT2EResNetBodyStages(
        prepared_stem, prepared_layer1, prepared_layer2, prepared_layer3, prepared_layer4,
    )


def prepare_pt2e_backbone_qat(model, config, inplace=False):
    """Replace only backbone body with a prepared x86 PT2E QAT graph."""
    prepared_model = model if inplace else copy.deepcopy(model)
    if isinstance(prepared_model.backbone, PT2EBackboneFPN):
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
    backbone = prepared_model.backbone
    if str(getattr(backbone, "pt2e_region_kind", "convnext")).lower() == "resnet50":
        if scope != "backbone":
            raise ValueError("ResNet50 PT2E currently supports only quantization.pt2e.region=backbone")
        prepared_region = _prepare_resnet50_stagewise_qat(backbone, config)
        prepared_model.backbone = PT2EBackboneFPN(prepared_region, backbone.fpn, backbone.out_channels)
        prepared_model.pt2e_quantized_region = "backbone.body(stagewise)"
        return prepared_model
    default_spatial_divisor = int(getattr(backbone, "pt2e_spatial_divisor", 32))
    spatial_divisor = 64 if scope == "backbone_fpn" else default_spatial_divisor
    if any(
        value <= 0 or value % spatial_divisor
        for value in (minimum_side, maximum_side, example_height, example_width)
    ):
        raise ValueError(
            f"PT2E {scope} image dimensions must be positive multiples of {spatial_divisor}"
        )
    if maximum_batch < example_batch:
        raise ValueError("pt2e.maximum_batch_size must be >= example_batch_size")

    region = build_backbone_body_region(backbone, scope=scope).cpu().eval()
    example = torch.randn(example_batch, 3, example_height, example_width)
    exported = export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(
            example_batch, maximum_batch, minimum_side, maximum_side,
            spatial_divisor,
        ),
    ).module()
    prepare_qat_pt2e, _, quantizer_type, xiq, _ = _torchao_pt2e()
    quantizer = quantizer_type()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
    prepared_region = prepare_qat_pt2e(exported, quantizer)
    fake_quantizers = _collect_pt2e_fake_quantizers(prepared_region)
    if not fake_quantizers:
        raise RuntimeError(
            "prepare_qat_pt2e completed but inserted no PT2E fake-quant modules. "
            "This almost always means the active torch/torchao stack is incompatible. "
            f"torch={torch.__version__}. "
            "On Kaggle, reinstall a PT2E-compatible torch first, then rerun this notebook."
        )
    prepared_model.backbone = PT2EBackboneFPN(
        prepared_region, backbone.fpn if scope == "backbone" else None, backbone.out_channels,
    )
    prepared_model.pt2e_quantized_region = (
        "backbone.body" if scope == "backbone" else "backbone.body+fpn"
    )
    if scope == "backbone_fpn":
        # FPN creates additional parity guards. Faster R-CNN must therefore pad
        # its internal ImageList to 64 instead of the default backbone stride.
        prepared_model.transform.size_divisible = 64
    return prepared_model


def set_pt2e_qat_phase(model, phase):
    """Set observer-only warmup, full fake-quant QAT, or frozen ranges."""
    if phase not in {"observer_warmup", "full", "frozen"}:
        raise ValueError("PT2E phase must be observer_warmup, full, or frozen")
    fake_quantizers = _collect_pt2e_fake_quantizers(model)
    if not fake_quantizers:
        raise RuntimeError(
            "No PT2E fake-quant modules found on the prepared model. "
            "The PT2E graph was likely created without QAT inserts because torch/torchao "
            "are incompatible in this runtime."
        )
    for module in fake_quantizers:
        if phase == "observer_warmup":
            module.enable_observer()
            module.disable_fake_quant()
        elif phase == "full":
            module.enable_observer()
            module.enable_fake_quant()
        else:
            module.disable_observer()
            module.enable_fake_quant()
    return len(fake_quantizers)


def convert_pt2e_backbone(model, inplace=False, compile_region=False):
    """Convert the prepared backbone graph and optionally compile that graph."""
    converted_model = model if inplace else copy.deepcopy(model)
    if not isinstance(converted_model.backbone, PT2EBackboneFPN):
        raise TypeError("Model has not been prepared by prepare_pt2e_backbone_qat")
    _, convert_pt2e, _, _, move_to_eval = _torchao_pt2e()
    converted_model.cpu()
    body_region = converted_model.backbone.body_region
    if isinstance(body_region, PT2EResNetBodyStages):
        for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
            converted_stage = convert_pt2e(getattr(body_region, name))
            move_to_eval(converted_stage)
            setattr(body_region, name, converted_stage)
        converted_region = body_region
    else:
        converted_region = convert_pt2e(body_region)
        move_to_eval(converted_region)
    converted_model.backbone.body_region = converted_region
    converted_model.eval()
    converted_model.pt2e_compiled = False
    if compile_region:
        compile_pt2e_region(converted_model)
    return converted_model


def compile_pt2e_region(model):
    """Compile only the converted tensor graph, leaving detection control flow eager."""
    python_header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if not python_header.is_file():
        raise RuntimeError(
            f"torch.compile x86 requires Python development headers; missing {python_header}. "
            "Install python3-dev (or use a Kaggle image that provides Python.h)."
        )
    model.backbone.body_region = torch.compile(model.backbone.body_region)
    model.pt2e_compiled = True
    return model


def save_pt2e_int8_artifact(path, model, metrics=None, extra=None):
    """Persist converted graph tensors without pickling the non-portable GraphModule."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "metrics": metrics or {},
        "extra": {**(extra or {}), "format": "pt2e_int8_state_dict"},
    }, path)
    return path


def load_pt2e_int8_artifact(path, config, compile_region=False):
    from ..models import build_fasterrcnn_convnext

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("extra", {}).get("format") != "pt2e_int8_state_dict":
        raise ValueError("Not a PT2E INT8 state-dict artifact")
    model = prepare_pt2e_backbone_qat(build_fasterrcnn_convnext(config), config)
    # Conversion establishes the quantized_decomposed topology. Empty observer
    # defaults are immediately replaced by the stored converted tensors below.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        model = convert_pt2e_backbone(model, inplace=True, compile_region=False)
    current_buffers = dict(model.named_buffers())
    for name, value in payload["model"].items():
        current = current_buffers.get(name)
        if current is None or current.shape == value.shape:
            continue
        module_name, _, buffer_name = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        if buffer_name not in module._buffers:
            raise RuntimeError(f"PT2E artifact parameter shape mismatch: {name}")
        # Per-channel qparams are data-dependent. An uncalibrated reconstruction
        # initially creates scalar placeholders; resize those registered buffers
        # before the strict state load.
        module._buffers[buffer_name] = torch.empty_like(value)
    model.load_state_dict(payload["model"], strict=True)
    model.cpu().eval()
    if compile_region:
        compile_pt2e_region(model)
    return model, payload
