# EchteAI

Selective-QAT Faster R-CNN for SeaDronesSee, now flattened around a single
root-level `pipelines/` package.

## What this repo does

The current pipeline trains a ConvNeXt-FPN Faster R-CNN in two stages:

1. FP32 training to get a strong baseline checkpoint.
2. Selective QAT for the quantized backbone / FPN / RPN parts, then INT8
   conversion for CPU deployment.

The project also includes evaluation, benchmark, visualization, and Kaggle /
Colab notebook flows for running the same pipeline on limited hardware.

## Main folders

- `configs/` – YAML configs for local, Colab, and Kaggle runs.
- `pipelines/` – core model, data, training, checkpoint, metric, and QAT code.
- `scripts/` – CLI entry points for train, eval, benchmark, video inference,
  comparison, and visualization.
- `colab/` – one-click notebook for Colab-style runs.
- `kaggle/` – Kaggle notebooks for staged training, benchmarking, and
  visualization.
- `weights/` – saved checkpoints and exported quantized weights.
- `tests/` – smoke tests for the selective-QAT pipeline.

The old `Python/` tree has been removed.

## Typical workflow

```bash
cd /home/nguyenducthang/EchteAI

# 1) Train FP32
python scripts/train_fp32.py --config configs/fasterrcnn_convnext_qat.yaml

# 2) Continue with QAT
python scripts/train_qat.py --config configs/fasterrcnn_convnext_qat.yaml

# 3) Evaluate or benchmark
python scripts/evaluate.py --model fp32 --split test
python scripts/evaluate.py --model int8 --split test
python scripts/benchmark.py --config configs/fasterrcnn_convnext_qat.yaml

# 4) Visualize FP32 vs INT8 outputs
python scripts/visualize_fp32_int8.py --config configs/fasterrcnn_convnext_qat.yaml
```

## ConvNeXt selective QAT

The ConvNeXt pipeline lives under `pipelines/convnext_qat/`. It contains:

- backbone construction
- Faster R-CNN assembly
- selective quantization wrappers
- checkpoint save / resume helpers
- training / validation engine
- metric computation for detection and benchmark reporting

By default the model keeps detection post-processing, image transforms, anchors,
ROIAlign, and regression in FP32, while quantizing the selected backbone/FPN/RPN
parts only.

## Notes

- INT8 checkpoints are CPU-oriented.
- If you run on Colab or Kaggle, prefer the notebook entry points under
  `colab/` and `kaggle/` because they keep durable outputs in Drive / dataset
  storage.
- `pycocotools` is optional; if it is missing, the repository falls back to the
  bundled detection metrics.

