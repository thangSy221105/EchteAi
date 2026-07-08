#!/usr/bin/env python3
"""Run exactly the next FP32 or QAT epoch, resume from Drive, then exit."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--stage", choices=["fp32", "qat", "pt2e"], required=True)
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def checkpoint_epoch(path):
    path = Path(path)
    if not path.is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    if args.stage == "fp32":
        last = Path(config["output"]["fp32_last"])
        total = int(config["training"]["fp32_epochs"])
        use_ddp = torch.cuda.is_available() and torch.cuda.device_count() >= 2
        script = "scripts/train_fp32_ddp.py" if use_ddp else "scripts/train_fp32.py"
    elif args.stage == "qat":
        last = Path(config["output"]["qat_last"])
        total = int(config["training"]["qat_epochs"])
        script = "scripts/train_qat.py"
        fp32_best = Path(config["output"]["fp32_best"])
        fp32_completed = checkpoint_epoch(config["output"]["fp32_last"])
        fp32_total = int(config["training"]["fp32_epochs"])
        if fp32_completed < fp32_total:
            raise RuntimeError(
                f"finish FP32 first: checkpoint is epoch {fp32_completed}/{fp32_total}"
            )
        if not fp32_best.is_file():
            raise FileNotFoundError(f"QAT requires the completed FP32 best checkpoint: {fp32_best}")
    else:
        last = Path(config["output"].get(
            "pt2e_qat_last", Path(config["output"]["directory"]) / "pt2e_qat_last.pt",
        ))
        total = int(config["training"].get("pt2e_qat_epochs", 3))
        script = "scripts/train_pt2e_qat.py"
        fp32_best = Path(config["output"]["fp32_best"])
        fp32_completed = checkpoint_epoch(config["output"]["fp32_last"])
        fp32_total = int(config["training"]["fp32_epochs"])
        if fp32_completed < fp32_total:
            raise RuntimeError(
                f"finish FP32 first: checkpoint is epoch {fp32_completed}/{fp32_total}"
            )
        if not fp32_best.is_file():
            raise FileNotFoundError(f"PT2E QAT requires the completed FP32 best checkpoint: {fp32_best}")

    completed = checkpoint_epoch(last)
    if completed >= total:
        print(f"{args.stage.upper()} already completed: {completed}/{total} epochs", flush=True)
        return

    command = [
        sys.executable, "-u", script, "--config", args.config,
        "--epochs-this-run", "1",
    ]
    if last.is_file():
        command += ["--resume", str(last)]
    if args.stage == "fp32" and script.endswith("_ddp.py"):
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={min(2, torch.cuda.device_count())}",
            script,
            "--config", args.config,
            "--epochs-this-run", "1",
        ]
        if last.is_file():
            command += ["--resume", str(last)]
    if args.stage == "qat" and args.variant:
        command += ["--variant", args.variant]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]

    print(
        f"Starting {args.stage.upper()} epoch {completed + 1}/{total}\n"
        f"Checkpoint: {last}\nCommand: {' '.join(command)}",
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    saved_epoch = checkpoint_epoch(last)
    if saved_epoch != completed + 1:
        raise RuntimeError(
            f"checkpoint verification failed: expected epoch {completed + 1}, got {saved_epoch}"
        )
    print(
        f"Finished {args.stage.upper()} epoch {saved_epoch}/{total}; "
        f"checkpoint verified on Drive: {last}",
        flush=True,
    )


if __name__ == "__main__":
    main()
