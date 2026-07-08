#!/usr/bin/env python3
"""Compare FP32, eager-island INT8, and PT2E backbone INT8 on x86 CPU."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import benchmark
from evaluate import load_model
from pipelines.convnext_qat.checkpoint import load_checkpoint, model_state_size_mb
from pipelines.convnext_qat.config import load_config, validate_dataset_paths
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import convert_pt2e_backbone, prepare_pt2e_backbone_qat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--pt2e-qat-checkpoint", required=True)
    parser.add_argument("--eager-int8-checkpoint")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_dataset_paths(config, ("test",))
    torch.set_num_threads(int(config.get("benchmark", {}).get("num_threads", 1)))
    images, _ = next(iter(build_coco_loader(config, "test", shuffle=False, batch_size=1)))
    images = [image.cpu() for image in images]
    warmup = int(config.get("benchmark", {}).get("warmup_iterations", 10))
    iterations = int(config.get("benchmark", {}).get("iterations", 50))
    results = {}

    fp32 = load_model(config, "fp32", args.fp32_checkpoint, torch.device("cpu"))
    results["fp32"] = benchmark(fp32, images, warmup, iterations)
    results["fp32"]["model_size_mb"] = model_state_size_mb(fp32)
    del fp32
    gc.collect()

    if args.eager_int8_checkpoint:
        eager = load_model(config, "int8", args.eager_int8_checkpoint, torch.device("cpu"))
        results["eager_m3"] = benchmark(eager, images, warmup, iterations)
        results["eager_m3"]["model_size_mb"] = model_state_size_mb(eager)
        del eager
        gc.collect()

    pt2e = build_fasterrcnn_convnext(config)
    pt2e = prepare_pt2e_backbone_qat(pt2e, config)
    load_checkpoint(args.pt2e_qat_checkpoint, pt2e)
    pt2e = convert_pt2e_backbone(pt2e, inplace=True, compile_region=args.compile)
    results["pt2e_backbone"] = benchmark(pt2e, images, warmup, iterations)
    results["pt2e_backbone"]["model_size_mb"] = model_state_size_mb(pt2e)
    results["pt2e_backbone"]["compiled"] = args.compile
    results["pt2e_speedup_vs_fp32"] = (
        results["fp32"]["latency_ms"] / results["pt2e_backbone"]["latency_ms"]
    )
    output = Path(args.output or Path(config["output"]["directory"]) / "pt2e_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
