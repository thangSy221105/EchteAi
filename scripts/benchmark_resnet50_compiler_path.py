#!/usr/bin/env python3
"""Benchmark compiler-facing ResNet50 scopes before detector integration."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force-w8a8", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def benchmark(module, sample, warmup_iters, iters):
    module = module.cpu().eval()
    timings = []
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = module(sample)
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = module(sample)
            timings.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(timings) / len(timings)
    return {
        "avg_ms": avg_ms,
        "fps": 1000.0 / avg_ms if avg_ms > 0 else float("nan"),
        "iters": int(iters),
    }


def load_fp32_model(config, checkpoint):
    model = build_fasterrcnn_convnext(config).cpu().eval()
    if checkpoint and Path(checkpoint).is_file():
        print(f"Loading FP32 checkpoint: {checkpoint}", flush=True)
        load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    else:
        print("No FP32 checkpoint loaded; benchmarking current model weights.", flush=True)
    return model


def load_int8_model(config, checkpoint, force_w8a8=False):
    if not checkpoint or not Path(checkpoint).is_file():
        return None, None
    print(f"Loading INT8 checkpoint: {checkpoint}", flush=True)
    model = build_fasterrcnn_convnext(config).cpu().eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("extra", {}) if isinstance(payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
    quantized_modules = metadata.get(
        "quantized_modules",
        quantized_modules_for_variant(config, variant),
    )

    mixed_precision_policy = None if force_w8a8 else (metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config))
    module_qconfig_map = None
    if mixed_precision_policy is not None:
        if policy_has_non_int8_weights(mixed_precision_policy):
            raise ValueError(
                "INT8 compiler benchmark cannot load a mixed-precision policy containing sub-8-bit weights. "
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
    load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    return model.cpu().eval(), metadata


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = resolve_compiler_scope(config)
    torch.set_num_threads(int(args.threads))

    batch_size = int(args.batch_size or compiler_cfg.get("example_batch_size", 1))
    height = int(args.height or compiler_cfg.get("example_height", 256))
    width = int(args.width or compiler_cfg.get("example_width", 320))

    sample = torch.randn(batch_size, 3, height, width)
    fp32_model = load_fp32_model(config, args.fp32_checkpoint or config["output"].get("fp32_best"))
    fp32_target = build_compiler_target_module(fp32_model, config).cpu().eval()
    result = {
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "threads": int(args.threads),
        "fp32_pytorch": benchmark(fp32_target, sample, args.warmup_iters, args.iters),
    }

    int8_checkpoint = args.int8_checkpoint or compiler_cfg.get("int8_reference_checkpoint")
    int8_model, int8_metadata = load_int8_model(config, int8_checkpoint, force_w8a8=args.force_w8a8)
    if int8_model is not None:
        int8_target = build_compiler_target_module(int8_model, config).cpu().eval()
        result["int8_eager_reference"] = benchmark(int8_target, sample, args.warmup_iters, args.iters)
        result["int8_eager_reference"]["checkpoint"] = str(int8_checkpoint)
        result["int8_eager_reference"]["metadata"] = int8_metadata
        result["int8_eager_reference"]["force_w8a8"] = bool(args.force_w8a8)
        result["speedup_int8_vs_fp32"] = (
            result["fp32_pytorch"]["avg_ms"] / result["int8_eager_reference"]["avg_ms"]
        )
    print(json.dumps(result, indent=2), flush=True)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
