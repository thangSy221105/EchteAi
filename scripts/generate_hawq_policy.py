#!/usr/bin/env python3
"""Generate a HAWQ-style mixed-precision policy for Faster R-CNN ResNet50-FPN."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.checkpoint import load_checkpoint
from pipelines.fasterrcnn_qat.config import choose_device, load_config
from pipelines.fasterrcnn_qat.data import build_coco_loader
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model
from pipelines.fasterrcnn_qat.quantization import (
    build_resnet50_mixed_precision_policy,
    estimate_resnet50_sensitivity,
    save_hawq_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint", help="optional FP32 checkpoint for sensitivity estimation")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scope", choices=["backbone", "backbone_fpn"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sensitivity-batches", type=int)
    parser.add_argument("--target-average-weight-bits", type=float)
    parser.add_argument("--activation-bits", type=int)
    parser.add_argument("--candidate-weight-bits", nargs="+", type=int, default=None)
    parser.add_argument("--output", help="override quantization.mixed_precision.policy_output")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    model_cfg = config.get("model", {})
    if str(model_cfg.get("backbone", "")).lower() != "resnet50":
        raise ValueError("HAWQ branch currently supports only model.backbone=resnet50")

    mixed_cfg = config.get("quantization", {}).get("mixed_precision", {})
    scope = args.scope or str(mixed_cfg.get("scope", "backbone"))
    target_average_weight_bits = float(
        args.target_average_weight_bits
        if args.target_average_weight_bits is not None
        else mixed_cfg.get("target_average_weight_bits", 6.0)
    )
    activation_bits = int(
        args.activation_bits
        if args.activation_bits is not None
        else mixed_cfg.get("activation_bits", 8)
    )
    candidate_weight_bits = (
        args.candidate_weight_bits
        if args.candidate_weight_bits is not None
        else mixed_cfg.get("candidate_weight_bits", [4, 8])
    )
    sensitivity_batches = int(
        args.sensitivity_batches
        if args.sensitivity_batches is not None
        else mixed_cfg.get("sensitivity_batches", 8)
    )
    output_path = (
        args.output
        or mixed_cfg.get("policy_output")
        or str(Path(config["output"]["directory"]) / "resnet50_hawq_policy.json")
    )

    device = choose_device(config.get("device", "auto"))
    loader = build_coco_loader(
        config,
        args.split,
        shuffle=False,
        limit=args.limit,
        batch_size=int(args.batch_size),
    )
    model = build_fasterrcnn_model(config)
    if args.fp32_checkpoint:
        print(f"Loading FP32 checkpoint: {args.fp32_checkpoint}", flush=True)
        load_checkpoint(args.fp32_checkpoint, model)
    elif Path(config["output"]["fp32_best"]).is_file():
        print(f"Loading default FP32 checkpoint: {config['output']['fp32_best']}", flush=True)
        load_checkpoint(config["output"]["fp32_best"], model)
    else:
        print("No FP32 checkpoint found; using current model weights for sensitivity estimation.", flush=True)

    print(
        f"Estimating HAWQ sensitivity split={args.split} scope={scope} "
        f"batches={sensitivity_batches} device={device}",
        flush=True,
    )
    sensitivities = estimate_resnet50_sensitivity(
        model,
        loader,
        device,
        scope=scope,
        max_batches=sensitivity_batches,
    )
    policy = build_resnet50_mixed_precision_policy(
        sensitivities,
        scope=scope,
        target_average_weight_bits=target_average_weight_bits,
        activation_bits=activation_bits,
        candidate_weight_bits=candidate_weight_bits,
    )
    save_hawq_policy(output_path, policy)

    print(f"Saved HAWQ policy: {output_path}", flush=True)
    print(json.dumps({
        "scope": policy["scope"],
        "assigned_average_weight_bits": policy["assigned_average_weight_bits"],
        "non_deploy_weight_bits_present": policy["non_deploy_weight_bits_present"],
        "top_sensitive_layers": sensitivities[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
