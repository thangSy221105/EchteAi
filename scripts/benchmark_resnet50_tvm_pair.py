#!/usr/bin/env python3
"""Benchmark a pair of TVM-compiled ResNet50 artifacts and report FP32 vs INT8 speedup."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.compiler import describe_tvm_output_shape, load_tvm_artifact, run_tvm_module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-lib", required=True)
    parser.add_argument("--fp32-metadata", required=True)
    parser.add_argument("--other-lib", required=True)
    parser.add_argument("--other-metadata", required=True)
    parser.add_argument("--other-label", default="other")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output")
    return parser.parse_args()


def benchmark(module, input_name, sample, warmup_iters, iters):
    for _ in range(warmup_iters):
        _ = run_tvm_module(module, input_name, sample)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run_tvm_module(module, input_name, sample)
        timings.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(timings) / len(timings)
    return {"avg_ms": avg_ms, "fps": 1000.0 / avg_ms if avg_ms > 0 else float("nan"), "iters": int(iters)}


def run_one(lib_path, metadata_path, warmup_iters, iters):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    sample = torch.randn(*metadata["example_shape"])
    module, _ = load_tvm_artifact(lib_path, metadata_path=metadata_path)
    outputs = run_tvm_module(module, metadata.get("input_name", "input0"), sample)
    return {
        "lib": str(lib_path),
        "metadata": metadata,
        "output_count": len(outputs),
        "output_shapes": [describe_tvm_output_shape(output) for output in outputs],
        "tvm_runtime": benchmark(module, metadata.get("input_name", "input0"), sample, warmup_iters, iters),
    }


def main():
    args = parse_args()
    fp32 = run_one(args.fp32_lib, args.fp32_metadata, args.warmup_iters, args.iters)
    other = run_one(args.other_lib, args.other_metadata, args.warmup_iters, args.iters)
    result = {
        "fp32": fp32,
        str(args.other_label): other,
        f"speedup_{args.other_label}_vs_fp32": fp32["tvm_runtime"]["avg_ms"] / other["tvm_runtime"]["avg_ms"],
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
