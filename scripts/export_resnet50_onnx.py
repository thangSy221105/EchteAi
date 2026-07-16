#!/usr/bin/env python3
"""Export compiler-facing ResNet50 modules to ONNX.

Supported sources:
- fp32: regular PyTorch model
- qat_graph: prepared QAT / fake-quant graph

This script intentionally does not export eager INT8 packed-parameter modules,
because the deployment path is now ONNX Q/DQ -> TensorRT rather than
eager-INT8 -> TVM.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint
from pipelines.convnext_qat.compiler import build_compiler_target_module, resolve_compiler_scope
from pipelines.convnext_qat.config import load_config, quantized_modules_for_variant
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import (
    mixed_precision_policy_from_config,
    module_qconfig_map_from_policy,
    policy_scope_to_quantized_modules,
    prepare_selective_qat,
    set_qat_phase,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_resnet50_hawq_compiler.yaml")
    parser.add_argument("--model", choices=["fp32", "qat_graph"], default="fp32")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--qat-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force-w8a8", action="store_true")
    return parser.parse_args()


def load_source_model(config, model_kind, fp32_checkpoint=None, qat_checkpoint=None, force_w8a8=False):
    model = build_fasterrcnn_convnext(config).cpu().eval()
    if model_kind == "fp32":
        checkpoint = fp32_checkpoint or config["output"].get("fp32_best")
        if checkpoint and Path(checkpoint).is_file():
            print(f"Loading FP32 checkpoint: {checkpoint}", flush=True)
            payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        else:
            print("No FP32 checkpoint loaded; exporting current model weights.", flush=True)
            payload = {}
        return model, payload

    checkpoint = qat_checkpoint or config["output"].get("qat_best") or config["output"].get("qat_last")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("QAT graph export requires --qat-checkpoint or output.qat_best/output.qat_last")
    print(f"Loading QAT graph checkpoint: {checkpoint}", flush=True)
    raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = raw_payload.get("extra", {}) if isinstance(raw_payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
    quantized_modules = metadata.get("quantized_modules", quantized_modules_for_variant(config, variant))
    mixed_precision_policy = None if force_w8a8 else (metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config))
    module_qconfig_map = None
    if mixed_precision_policy is not None:
        quantized_modules = policy_scope_to_quantized_modules(mixed_precision_policy)
        module_qconfig_map = module_qconfig_map_from_policy(mixed_precision_policy)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        model = prepare_selective_qat(
            model,
            variant,
            backend,
            quantized_modules=quantized_modules,
            module_qconfig_map=module_qconfig_map,
        )
    payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    set_qat_phase(model, "frozen")
    return model.cpu().eval(), payload


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = resolve_compiler_scope(config)
    batch_size = int(compiler_cfg.get("example_batch_size", 1))
    height = int(compiler_cfg.get("example_height", 256))
    width = int(compiler_cfg.get("example_width", 320))

    artifact_dir = Path(
        args.artifact_dir
        or compiler_cfg.get("artifact_dir")
        or Path(config["output"]["directory"]) / "onnx_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_source_model(
        config,
        args.model,
        fp32_checkpoint=args.fp32_checkpoint,
        qat_checkpoint=args.qat_checkpoint,
        force_w8a8=args.force_w8a8,
    )
    target_module = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)
    with torch.inference_mode():
        outputs = target_module(sample)
    output_names = [f"output_{index}" for index in range(len(outputs))]

    onnx_path = Path(args.output) if args.output else artifact_dir / f"resnet50_{scope}_{args.model}.onnx"
    metadata_path = onnx_path.with_name(f"{onnx_path.stem}_metadata.json")

    export_kwargs = dict(
        input_names=["input0"],
        output_names=output_names,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    if args.model == "qat_graph":
        torch.onnx.export(
            target_module,
            (sample,),
            str(onnx_path),
            dynamo=False,
            **export_kwargs,
        )
    else:
        torch.onnx.export(
            target_module,
            (sample,),
            str(onnx_path),
            **export_kwargs,
        )

    metadata = {
        "model_kind": args.model,
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "onnx_path": str(onnx_path),
        "input_name": "input0",
        "output_names": output_names,
        "example_shape": list(sample.shape),
        "checkpoint_extra": payload.get("extra", {}) if isinstance(payload, dict) else {},
        "force_w8a8": bool(args.force_w8a8),
        "opset": int(args.opset),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved ONNX model: {onnx_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
