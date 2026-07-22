#!/usr/bin/env python3
"""Inspect how a checkpoint matches the current ResNet50 Faster R-CNN model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.checkpoint import load_partial_checkpoint
from pipelines.fasterrcnn_qat.config import load_config
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_resnet50_hawq_compiler.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    model = build_fasterrcnn_model(config).cpu().eval()
    payload = load_partial_checkpoint(args.checkpoint, model, map_location="cpu")
    summary = payload.get("extra", {})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
