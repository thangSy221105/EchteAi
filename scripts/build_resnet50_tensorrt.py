#!/usr/bin/env python3
"""Build TensorRT engines from exported ONNX backbone artifacts via trtexec."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--engine")
    parser.add_argument("--precision", choices=["fp32", "int8"], required=True)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def find_trtexec(binary_name):
    resolved = shutil.which(binary_name)
    if resolved:
        return resolved
    common = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
    ]
    for candidate in common:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find trtexec ({binary_name}). Install TensorRT or pass --trtexec with a full path."
    )


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    metadata_path = Path(args.metadata) if args.metadata else onnx_path.with_name(f"{onnx_path.stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    example_shape = metadata.get("example_shape", [1, 3, 256, 320])
    input_name = metadata.get("input_name", "input0")

    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(f".{args.precision}.engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    shape_text = "x".join(str(int(v)) for v in example_shape)
    trtexec = find_trtexec(args.trtexec)
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{int(args.workspace_mb)}",
        f"--shapes={input_name}:{shape_text}",
    ]
    if args.precision == "int8":
        command.append("--int8")
    else:
        command.append("--fp32")
    command.extend(args.extra_arg)

    print("Running:", " ".join(str(part) for part in command), flush=True)
    proc = subprocess.run(command, text=True, capture_output=True)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") or not proc.stdout else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)

    summary = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "input_name": input_name,
        "example_shape": example_shape,
    }
    summary_path = engine_path.with_suffix(engine_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved engine summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
