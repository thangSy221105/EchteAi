#!/usr/bin/env python3
"""Benchmark TensorRT FP32 and INT8 engines with trtexec."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


_MEAN_PATTERNS = (
    re.compile(r"GPU Compute Time:\s*min = .*? mean = ([0-9.]+) ms", re.IGNORECASE),
    re.compile(r"Latency:\s*min = .*? mean = ([0-9.]+) ms", re.IGNORECASE),
)
_THROUGHPUT_PATTERNS = (
    re.compile(r"Throughput:\s*([0-9.]+)\s*qps", re.IGNORECASE),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--shape", required=True, help="Example: input0:1x3x256x320")
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--output")
    return parser.parse_args()


def find_trtexec(binary_name):
    resolved = shutil.which(binary_name)
    if resolved:
        return resolved
    for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find trtexec ({binary_name}). Install TensorRT or pass --trtexec with a full path."
    )


def parse_metrics(text):
    mean_ms = None
    for pattern in _MEAN_PATTERNS:
        match = pattern.search(text)
        if match:
            mean_ms = float(match.group(1))
            break
    throughput = None
    for pattern in _THROUGHPUT_PATTERNS:
        match = pattern.search(text)
        if match:
            throughput = float(match.group(1))
            break
    if mean_ms is None:
        raise RuntimeError("Could not parse mean latency from trtexec output.")
    return {
        "avg_ms": mean_ms,
        "fps": throughput if throughput is not None else (1000.0 / mean_ms if mean_ms > 0 else float("nan")),
    }


def run_one(trtexec, engine_path, shape_text, warmup_ms, iterations):
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--shapes={shape_text}",
        f"--warmUp={int(warmup_ms)}",
        f"--iterations={int(iterations)}",
        "--duration=0",
        "--noDataTransfers",
        "--useSpinWait",
    ]
    proc = subprocess.run(command, text=True, capture_output=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command, output=proc.stdout, stderr=proc.stderr)
    metrics = parse_metrics(combined)
    metrics["engine"] = str(engine_path)
    return metrics


def main():
    args = parse_args()
    trtexec = find_trtexec(args.trtexec)
    fp32 = run_one(trtexec, Path(args.fp32_engine), args.shape, args.warmup_ms, args.iterations)
    int8 = run_one(trtexec, Path(args.int8_engine), args.shape, args.warmup_ms, args.iterations)
    result = {
        "fp32": fp32,
        "int8": int8,
        "speedup_int8_vs_fp32": fp32["avg_ms"] / int8["avg_ms"],
        "shape": args.shape,
        "warmup_ms": int(args.warmup_ms),
        "iterations": int(args.iterations),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
