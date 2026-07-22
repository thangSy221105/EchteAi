#!/usr/bin/env python3
"""Train a clean FP32 Faster R-CNN baseline for the current repo pipeline.

This entrypoint is intended for the user's current ResNet50 workflow:
- train the detector in the repo's own pipeline
- optionally warm-start from an external checkpoint
- optionally run only a few hundred steps for smoke testing

It keeps FP32 training separate from QAT so the baseline can be validated
before any quantization work starts.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.checkpoint import (
    load_checkpoint,
    load_partial_checkpoint,
    save_checkpoint,
)
from pipelines.fasterrcnn_qat.config import choose_device, load_config
from pipelines.fasterrcnn_qat.data import build_coco_loader
from pipelines.fasterrcnn_qat.engine import (
    append_epoch_benchmark,
    benchmark_inference,
    make_optimizer,
    train_one_epoch,
)
from pipelines.fasterrcnn_qat.metrics import evaluate_model
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_resnet50_hawq_compiler.yaml")
    parser.add_argument("--limit", type=int, help="limit each split for a quick experiment")
    parser.add_argument("--resume", help="resume a repo-native FP32 checkpoint")
    parser.add_argument(
        "--init-checkpoint",
        help="initialize model weights from another checkpoint before FP32 training",
    )
    parser.add_argument(
        "--partial-init-checkpoint",
        action="store_true",
        help="load only matching keys from --init-checkpoint",
    )
    parser.add_argument(
        "--epochs-this-run",
        type=int,
        help="stop after this many epochs; useful for Colab/Kaggle sessions",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="cap training steps per epoch for a fast smoke test",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="save checkpoints after training epochs without running validation",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="skip epoch inference benchmark on the validation split",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(config.get("device", "auto"))
    fp32_batch_size = int(config["training"].get("fp32_batch_size", config["training"]["batch_size"]))

    train_loader = build_coco_loader(
        config,
        "train",
        limit=args.limit,
        batch_size=fp32_batch_size,
    )
    val_loader = build_coco_loader(config, "val", shuffle=False, limit=args.limit)

    print(
        f"FP32 baseline setup device={device} backbone={config['model']['backbone']} "
        f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)} "
        f"epochs={config['training']['fp32_epochs']} batch_size={fp32_batch_size}",
        flush=True,
    )
    print(f"output.fp32_best={config['output']['fp32_best']}", flush=True)
    print(f"output.fp32_last={config['output']['fp32_last']}", flush=True)

    model = build_fasterrcnn_model(config).to(device)
    print(f"model parameters={model.logical_parameter_count:,}", flush=True)

    optimizer = make_optimizer(model, config, qat=False)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"].get("lr_step_size", 8)),
        gamma=float(config["training"].get("lr_gamma", 0.1)),
    )

    best_map = -1.0
    start_epoch = 0

    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")

    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            strict=True,
        )
        start_epoch = int(payload.get("epoch", 0))
        best_map = float(payload.get("extra", {}).get("best_map", -1.0))
        print(f"resumed checkpoint={args.resume} epoch={start_epoch}", flush=True)
    elif args.init_checkpoint:
        if args.partial_init_checkpoint:
            payload = load_partial_checkpoint(args.init_checkpoint, model, map_location="cpu")
            summary = payload.get("extra", {})
            print(
                "partial init load: "
                f"matched={summary.get('matched_key_count', 0)} "
                f"missing={summary.get('missing_key_count', 0)} "
                f"unexpected={summary.get('unexpected_key_count', 0)} "
                f"shape_mismatches={summary.get('shape_mismatch_count', 0)}",
                flush=True,
            )
        else:
            load_checkpoint(args.init_checkpoint, model, map_location="cpu", strict=True)
            print(f"initialized from checkpoint={args.init_checkpoint}", flush=True)

    total_epochs = int(config["training"]["fp32_epochs"])
    if args.epochs_this_run is not None and args.epochs_this_run <= 0:
        raise ValueError("--epochs-this-run must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    end_epoch = total_epochs
    if args.epochs_this_run is not None:
        end_epoch = min(start_epoch + args.epochs_this_run, total_epochs)

    for epoch in range(start_epoch, end_epoch):
        total_loader_steps = len(train_loader)
        effective_steps = min(total_loader_steps, int(args.max_steps)) if args.max_steps is not None else total_loader_steps
        print(
            f"FP32 epoch={epoch + 1}/{total_epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
        print(
            f"train loop ready: total_loader_steps={total_loader_steps} "
            f"effective_steps={effective_steps} "
            f"print_frequency={int(config['training'].get('print_frequency', 20))} "
            f"max_steps={args.max_steps if args.max_steps is not None else 'none'}",
            flush=True,
        )
        warmup_scheduler = None
        if epoch == 0:
            warmup_iterations = min(
                int(config["training"].get("warmup_iterations", 0)),
                max(len(train_loader) - 1, 0),
            )
            if args.max_steps is not None:
                warmup_iterations = min(warmup_iterations, max(int(args.max_steps) - 1, 0))
            if warmup_iterations:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.001,
                    total_iters=warmup_iterations,
                )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(config["training"].get("grad_clip_norm", 0)),
            int(config["training"].get("print_frequency", 20)),
            warmup_scheduler,
            max_steps=args.max_steps,
        )
        scheduler.step()

        if args.skip_validation:
            save_checkpoint(
                config["output"]["fp32_last"],
                model,
                optimizer,
                epoch + 1,
                {"train": train_metrics},
                {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
                scheduler,
            )
            print(f"saved FP32 checkpoint: {config['output']['fp32_last']}", flush=True)
            print(f"epoch={epoch + 1} train={train_metrics} validation=skipped", flush=True)
            continue

        print("FP32 validation started", flush=True)
        val_metrics = evaluate_model(model, val_loader, device, include_rpn=True)
        print("FP32 validation completed", flush=True)

        payload_metrics = dict(val_metrics)
        payload_metrics["train"] = train_metrics

        save_checkpoint(
            config["output"]["fp32_last"],
            model,
            optimizer,
            epoch + 1,
            payload_metrics,
            {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
            scheduler,
        )
        print(f"saved FP32 checkpoint: {config['output']['fp32_last']}", flush=True)

        benchmark_metrics = None
        if not args.skip_benchmark:
            benchmark_metrics = benchmark_inference(
                model,
                val_loader,
                device,
                int(config["training"].get("epoch_benchmark_images", 100)),
            )
            benchmark_record = {"stage": "fp32", "epoch": epoch + 1, **benchmark_metrics}
            benchmark_history = config["output"].get(
                "epoch_benchmarks",
                str(Path(config["output"]["directory"]) / "epoch_benchmarks.json"),
            )
            append_epoch_benchmark(benchmark_history, benchmark_record)
            print(f"FP32 epoch benchmark={benchmark_record}", flush=True)

        print(f"epoch={epoch + 1} train={train_metrics} validation={val_metrics}", flush=True)

        if val_metrics["map_50_95"] > best_map:
            best_map = float(val_metrics["map_50_95"])
            save_checkpoint(
                config["output"]["fp32_best"],
                model,
                optimizer,
                epoch + 1,
                payload_metrics,
                {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
                scheduler,
            )
            print(f"saved new FP32 best: {config['output']['fp32_best']}", flush=True)

        save_checkpoint(
            config["output"]["fp32_last"],
            model,
            optimizer,
            epoch + 1,
            payload_metrics,
            {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
            scheduler,
        )
        print(f"saved FP32 resume checkpoint: {config['output']['fp32_last']}", flush=True)

    print(
        f"FP32 run completed at epoch {end_epoch}/{total_epochs}. "
        f"resume checkpoint={config['output']['fp32_last']}",
        flush=True,
    )
    if not args.skip_validation and end_epoch == total_epochs:
        print(f"best FP32 checkpoint: {config['output']['fp32_best']} (mAP={best_map:.4f})", flush=True)


if __name__ == "__main__":
    main()
