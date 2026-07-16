#!/usr/bin/env python3
"""Benchmark compiler-facing ResNet50 scopes before detector integration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint
from pipelines.convnext_qat.compiler import build_compiler_target_module, resolve_compiler_scope
from pipelines.convnext_qat.config import load_config
from pipelines.convnext_qat.models import build_fasterrcnn_convnext


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
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


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = resolve_compiler_scope(config)
    torch.set_num_threads(int(args.threads))

    batch_size = int(args.batch_size or compiler_cfg.get("example_batch_size", 1))
    height = int(args.height or compiler_cfg.get("example_height", 256))
    width = int(args.width or compiler_cfg.get("example_width", 320))

    model = build_fasterrcnn_convnext(config).cpu().eval()
    checkpoint = args.fp32_checkpoint or config["output"].get("fp32_best")
    if checkpoint and Path(checkpoint).is_file():
        print(f"Loading checkpoint: {checkpoint}", flush=True)
        load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    else:
        print("No checkpoint loaded; benchmarking current model weights.", flush=True)

    target = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)
    result = {
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "threads": int(args.threads),
        "fp32_pytorch": benchmark(target, sample, args.warmup_iters, args.iters),
    }
    print(json.dumps(result, indent=2), flush=True)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
