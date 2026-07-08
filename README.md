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

## Small-object anchors and overlapping tiles

The SeaDronesSee config derives its five P2-P6 anchor scales from the resized
training bounding boxes (`model.anchor_sizes: auto`). Inspect the exact values:

```bash
python scripts/analyze_anchors.py --config configs/seadronessee_colab.yaml
```

Overlapping training crops are configured under `augmentation.tiling`. Crops
containing object centres are retained together with a small, configurable
sample of background crops. Full-image tiled evaluation shifts every crop's
boxes back to global coordinates and performs class-aware NMS:

```bash
python scripts/evaluate.py --config configs/seadronessee_colab.yaml \
  --model fp32 --split val --tiled --skip-rpn-metrics
```

Changing anchor scales changes RPN training behaviour, so start a new FP32 run
rather than resuming a checkpoint trained with the old anchors. Tiling preserves
small-object pixels but costs additional forward passes; compare it against the
normal full-image evaluation as an ablation.

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

## PT2E graph-mode QAT (x86 CPU)

The eager M3 implementation remains available as the operator-island baseline.
For lower Q/DQ and Python overhead, the repository also provides an experimental
TorchAO PT2E path. It exports the ConvNeXt body as one tensor-to-tuple graph,
applies `X86InductorQuantizer` QAT, and leaves FPN/RPN/ROI/NMS in FP32. After
the backbone experiment is stable, set `quantization.pt2e.region` to
`backbone_fpn` for the second ablation; RPN/ROI/NMS still remain FP32.

```bash
pip install -e '.[pt2e]'

python scripts/train_pt2e_qat.py \
  --config configs/seadronessee_colab.yaml \
  --fp32-checkpoint /path/to/fp32_best.pt \
  --epochs-this-run 1

# Or use the common one-epoch/resume launcher:
python scripts/train_next_epoch.py \
  --config configs/seadronessee_colab.yaml --stage pt2e

python scripts/benchmark_pt2e.py \
  --config configs/seadronessee_colab.yaml \
  --fp32-checkpoint /path/to/fp32_best.pt \
  --pt2e-qat-checkpoint /path/to/pt2e_qat_best.pt \
  --compile
```

`torch.compile` is intentionally used only for the converted backbone graph;
the dynamic proposal, ROIAlign and NMS code remains eager FP32. PT2E checkpoints
are separate from eager-QAT checkpoints and cannot be resumed interchangeably.

## Notes

- INT8 checkpoints are CPU-oriented.
- If you run on Colab or Kaggle, prefer the notebook entry points under
  `colab/` and `kaggle/` because they keep durable outputs in Drive / dataset
  storage.
- `pycocotools` is optional; if it is missing, the repository falls back to the
  bundled detection metrics.
