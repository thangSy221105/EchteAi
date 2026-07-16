#!/usr/bin/env python3
"""Compile a compiler-facing ResNet50 backbone artifact with TVM."""

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
from pipelines.convnext_qat.compiler import (
    build_compiler_target_module,
    compile_tvm_from_module,
    load_tvm_artifact,
    resolve_compiler_scope,
    run_tvm_module,
    save_tvm_artifact,
)
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
    parser.add_argument("--config", default="configs/seadronessee_resnet50_hawq_compiler.yaml")
    parser.add_argument("--model", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--int8-checkpoint")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--target", default="llvm")
    parser.add_argument("--input-name", default="input0")
    parser.add_argument("--opt-level", type=int, default=3)
    return parser.parse_args()


def load_model_for_compile(config, model_kind, fp32_checkpoint=None, int8_checkpoint=None):
    model = build_fasterrcnn_convnext(config).cpu().eval()
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    if model_kind == "fp32":
        checkpoint = fp32_checkpoint or config["output"].get("fp32_best")
        if checkpoint and Path(checkpoint).is_file():
            print(f"Loading FP32 checkpoint: {checkpoint}", flush=True)
            payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        else:
            payload = {}
            print("No FP32 checkpoint loaded; compiling current model weights.", flush=True)
        return model, payload

    checkpoint = int8_checkpoint or compiler_cfg.get("int8_reference_checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("INT8 compile requires --int8-checkpoint or quantization.compiler.int8_reference_checkpoint")
    print(f"Loading INT8 checkpoint: {checkpoint}", flush=True)
    raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = raw_payload.get("extra", {}) if isinstance(raw_payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
    quantized_modules = metadata.get(
        "quantized_modules",
        quantized_modules_for_variant(config, variant),
    )
    mixed_precision_policy = metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config)
    module_qconfig_map = None
    if mixed_precision_policy is not None:
        if policy_has_non_int8_weights(mixed_precision_policy):
            raise ValueError(
                "TVM INT8 compile path needs a true W8A8 eager INT8 checkpoint. "
                "Mixed policies containing sub-8-bit weights are not supported here."
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
        or Path(config["output"]["directory"]) / "compiler_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_model_for_compile(
        config,
        args.model,
        fp32_checkpoint=args.fp32_checkpoint,
        int8_checkpoint=args.int8_checkpoint,
    )
    target_module = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)
    compiled, tvm_mode = compile_tvm_from_module(
        target_module,
        sample,
        input_name=args.input_name,
        target=args.target,
        opt_level=args.opt_level,
    )

    artifact_name = f"resnet50_{scope}_{args.model}_tvm.so"
    metadata = {
        "model_kind": args.model,
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "tvm_mode": tvm_mode,
        "target": args.target,
        "input_name": args.input_name,
        "example_shape": list(sample.shape),
        "checkpoint_extra": payload.get("extra", {}) if isinstance(payload, dict) else {},
    }
    lib_path, metadata_path = save_tvm_artifact(artifact_dir, compiled, metadata, artifact_name)

    module, _ = load_tvm_artifact(lib_path)
    outputs = run_tvm_module(module, args.input_name, sample)
    metadata["output_count"] = len(outputs)
    metadata["output_shapes"] = [list(output.shape) for output in outputs]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved TVM library: {lib_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
