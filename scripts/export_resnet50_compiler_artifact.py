#!/usr/bin/env python3
"""Export a compiler-facing ResNet50 backbone artifact.

This script does not claim a full deployment path by itself. Its role is to:
1. freeze the exact tensor interface the compiler backend will own;
2. store sample output metadata for later integration checks; and
3. optionally save a torch.export program as a neutral interchange artifact.
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
from pipelines.convnext_qat.compiler import describe_tvm_output_shape
from pipelines.convnext_qat.config import load_config, quantized_modules_for_variant
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import (
    convert_selective_qat,
    mixed_precision_policy_from_config,
    module_qconfig_map_from_policy,
    policy_has_non_int8_weights,
    policy_scope_to_quantized_modules,
    prepare_selective_qat,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--int8-checkpoint")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--force-w8a8", action="store_true")
    parser.add_argument("--model", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--format", choices=["torch_export", "state_dict"], default="torch_export")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = resolve_compiler_scope(config)

    artifact_dir = Path(
        args.artifact_dir
        or compiler_cfg.get("artifact_dir")
        or Path(config["output"]["directory"]) / "compiler_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(compiler_cfg.get("example_batch_size", 1))
    height = int(compiler_cfg.get("example_height", 256))
    width = int(compiler_cfg.get("example_width", 320))
    model = build_fasterrcnn_convnext(config).cpu().eval()
    payload = {}

    if args.model == "fp32":
        checkpoint = args.fp32_checkpoint or config["output"].get("fp32_best")
        if checkpoint and Path(checkpoint).is_file():
            print(f"Loading FP32 checkpoint: {checkpoint}", flush=True)
            payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        else:
            print("No FP32 checkpoint loaded; exporting current model weights.", flush=True)
    else:
        checkpoint = args.int8_checkpoint or compiler_cfg.get("int8_reference_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("INT8 export requires --int8-checkpoint or quantization.compiler.int8_reference_checkpoint")
        print(f"Loading INT8 checkpoint: {checkpoint}", flush=True)
        raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = raw_payload.get("extra", {}) if isinstance(raw_payload, dict) else {}
        variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
        backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
        quantized_modules = metadata.get(
            "quantized_modules",
            quantized_modules_for_variant(config, variant),
        )
        mixed_precision_policy = None if args.force_w8a8 else (metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config))
        module_qconfig_map = None
        if mixed_precision_policy is not None:
            if policy_has_non_int8_weights(mixed_precision_policy):
                raise ValueError(
                    "INT8 compiler export cannot use a mixed-precision policy containing sub-8-bit weights. "
                    "Provide a true W8A8 eager INT8 checkpoint."
                )
            quantized_modules = policy_scope_to_quantized_modules(mixed_precision_policy)
            module_qconfig_map = module_qconfig_map_from_policy(mixed_precision_policy)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
            model = convert_selective_qat(
                prepare_selective_qat(
                    model,
                    variant,
                    backend,
                    quantized_modules=quantized_modules,
                    module_qconfig_map=module_qconfig_map,
                )
            )
        payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)

    target = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)
    with torch.inference_mode():
        outputs = target(sample)

    metadata = {
        "format": args.format,
        "model_kind": args.model,
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "example_batch_size": batch_size,
        "example_height": height,
        "example_width": width,
        "output_count": len(outputs),
        "output_shapes": [describe_tvm_output_shape(tensor) for tensor in outputs],
        "checkpoint_extra": payload.get("extra", {}) if isinstance(payload, dict) else {},
        "force_w8a8": bool(args.force_w8a8),
    }

    if args.model == "int8" and args.format == "torch_export":
        raise ValueError("Quantized eager reference artifacts currently support state_dict export only")

    if args.format == "torch_export":
        exported = torch.export.export(target, (sample,))
        artifact_path = artifact_dir / f"resnet50_{scope}.pt2"
        torch.export.save(exported, artifact_path)
    else:
        suffix = "int8" if args.model == "int8" else "fp32"
        artifact_path = artifact_dir / f"resnet50_{scope}_{suffix}_state_dict.pt"
        torch.save(target.state_dict(), artifact_path)

    metadata_suffix = "int8" if args.model == "int8" else "fp32"
    metadata_path = artifact_dir / f"resnet50_{scope}_{metadata_suffix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved artifact: {artifact_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
