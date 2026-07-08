#!/usr/bin/env python3
"""Fine-tune graph-mode PT2E QAT for ConvNeXt while keeping FPN/heads FP32."""

import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from pipelines.convnext_qat.config import choose_device, load_config
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.engine import benchmark_inference, make_optimizer, train_one_epoch
from pipelines.convnext_qat.metrics import evaluate_model
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import prepare_pt2e_backbone_qat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--epochs-this-run", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def checkpoint_extra(config, best_map):
    return {
        "format": "pt2e_prepared_qat",
        "region": config.get("quantization", {}).get("pt2e", {}).get("region", "backbone"),
        "backend": "x86_inductor",
        "best_map": best_map,
        "anchor_sizes": config["model"].get("anchor_sizes"),
    }


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    device = choose_device(config.get("device", "auto"))
    batch_size = int(config["training"].get("qat_batch_size", 1))
    train_loader = build_coco_loader(config, "train", limit=args.limit, batch_size=batch_size)
    val_loader = build_coco_loader(
        config, "val", shuffle=False, limit=args.limit, batch_size=batch_size,
    )

    model = build_fasterrcnn_convnext(config)
    fp32_checkpoint = args.fp32_checkpoint or config["output"]["fp32_best"]
    print(f"Loading FP32 checkpoint: {fp32_checkpoint}", flush=True)
    load_checkpoint(fp32_checkpoint, model)
    print("Exporting ConvNeXt body and preparing PT2E x86 QAT graph...", flush=True)
    model = prepare_pt2e_backbone_qat(model, config).to(device)
    optimizer = make_optimizer(model, config, qat=True)

    start_epoch, best_map = 0, -1.0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        best_map = float(payload.get("extra", {}).get("best_map", -1.0))
        print(f"Resumed PT2E QAT checkpoint={args.resume} epoch={start_epoch}", flush=True)
    total_epochs = int(config["training"].get("pt2e_qat_epochs", 3))
    if args.epochs_this_run is not None and args.epochs_this_run <= 0:
        raise ValueError("--epochs-this-run must be positive")
    end_epoch = total_epochs
    if args.epochs_this_run is not None:
        end_epoch = min(total_epochs, start_epoch + args.epochs_this_run)

    last_path = config["output"].get(
        "pt2e_qat_last", str(Path(config["output"]["directory"]) / "pt2e_qat_last.pt"),
    )
    best_path = config["output"].get(
        "pt2e_qat_best", str(Path(config["output"]["directory"]) / "pt2e_qat_best.pt"),
    )
    for epoch in range(start_epoch, end_epoch):
        print(f"PT2E QAT epoch={epoch + 1}/{total_epochs}", flush=True)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            float(config["training"].get("grad_clip_norm", 0)),
            int(config["training"].get("print_frequency", 20)),
        )
        save_checkpoint(
            last_path, model, optimizer, epoch + 1,
            {"train": train_metrics, "validation_pending": True},
            checkpoint_extra(config, best_map),
        )
        print(f"Saved pre-validation PT2E checkpoint: {last_path}", flush=True)
        validation = evaluate_model(model, val_loader, device, include_rpn=False)
        timing = benchmark_inference(
            model, val_loader, device,
            int(config["training"].get("epoch_benchmark_images", 100)),
        )
        metrics = {**validation, "benchmark": timing}
        if validation["map_50_95"] > best_map:
            best_map = validation["map_50_95"]
            save_checkpoint(
                best_path, model, optimizer, epoch + 1, metrics,
                checkpoint_extra(config, best_map),
            )
            print(f"Saved new PT2E QAT best: {best_path}", flush=True)
        save_checkpoint(
            last_path, model, optimizer, epoch + 1, metrics,
            checkpoint_extra(config, best_map),
        )
        print(f"PT2E epoch={epoch + 1} train={train_metrics} validation={validation}", flush=True)
    print(f"PT2E QAT run completed at epoch {end_epoch}/{total_epochs}", flush=True)


if __name__ == "__main__":
    main()
