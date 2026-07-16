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
    parser.add_argument("--artifact-dir")
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

    checkpoint = args.fp32_checkpoint or config["output"].get("fp32_best")
    if checkpoint and Path(checkpoint).is_file():
        print(f"Loading checkpoint: {checkpoint}", flush=True)
        payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    else:
        payload = {}
        print("No checkpoint loaded; exporting current model weights.", flush=True)

    target = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)
    with torch.inference_mode():
        outputs = target(sample)

    metadata = {
        "format": args.format,
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "example_batch_size": batch_size,
        "example_height": height,
        "example_width": width,
        "output_count": len(outputs),
        "output_shapes": [list(tensor.shape) for tensor in outputs],
        "checkpoint_extra": payload.get("extra", {}) if isinstance(payload, dict) else {},
    }

    if args.format == "torch_export":
        exported = torch.export.export(target, (sample,))
        artifact_path = artifact_dir / f"resnet50_{scope}.pt2"
        torch.export.save(exported, artifact_path)
    else:
        artifact_path = artifact_dir / f"resnet50_{scope}_state_dict.pt"
        torch.save(target.state_dict(), artifact_path)

    metadata_path = artifact_dir / f"resnet50_{scope}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved artifact: {artifact_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
