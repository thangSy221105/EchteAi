"""CPU end-to-end PT2E QAT, resume, convert, artifact and inference smoke test."""

import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import (
    convert_pt2e_backbone, load_pt2e_int8_artifact, prepare_pt2e_backbone_qat,
    save_pt2e_int8_artifact, set_pt2e_qat_phase,
)


def config():
    return {
        "dataset": {"num_classes": 3},
        "model": {
            "backbone": "convnext_tiny", "pretrained_backbone": False,
            "trainable_backbone_layers": 4, "fpn_out_channels": 256,
            "min_size": 64, "max_size": 64,
            "anchor_sizes": [8, 16, 32, 64, 128], "aspect_ratios": [0.5, 1.0, 2.0],
            "rpn_pre_nms_top_n_train": 40, "rpn_pre_nms_top_n_test": 20,
            "rpn_post_nms_top_n_train": 20, "rpn_post_nms_top_n_test": 10,
        },
        "training": {"qat_batch_size": 1},
        "quantization": {"pt2e": {
            "region": "backbone", "example_batch_size": 1, "maximum_batch_size": 1,
            "minimum_image_side": 64, "maximum_image_side": 128,
            "example_height": 64, "example_width": 64,
        }},
    }


def prepared(cfg):
    return prepare_pt2e_backbone_qat(build_fasterrcnn_convnext(cfg), cfg)


def main():
    torch.set_num_threads(1)
    cfg = config()
    image = torch.rand(3, 64, 64)
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 40.0, 48.0]]),
        "labels": torch.tensor([1]),
    }
    model = prepared(cfg)
    assert set_pt2e_qat_phase(model, "observer_warmup") > 0
    model([image], [target])
    set_pt2e_qat_phase(model, "full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss = sum(model([image], [target]).values())
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "qat.pt"
        artifact = Path(directory) / "int8.pt"
        save_checkpoint(checkpoint, model, optimizer, epoch=1)
        resumed = prepared(cfg)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
        load_checkpoint(checkpoint, resumed, resumed_optimizer)
        set_pt2e_qat_phase(resumed, "frozen")
        resumed.eval()
        resumed([image])
        converted = convert_pt2e_backbone(resumed, inplace=True)
        output = converted([image])[0]
        assert {"boxes", "scores", "labels"} <= output.keys()
        save_pt2e_int8_artifact(artifact, converted, {"map_50_95": 0.0})
        loaded, payload = load_pt2e_int8_artifact(artifact, cfg)
        assert payload["extra"]["format"] == "pt2e_int8_state_dict"
        loaded([image])
        if os.environ.get("RUN_PT2E_COMPILE") == "1":
            from pipelines.convnext_qat.quantization import compile_pt2e_region
            compile_pt2e_region(loaded)
            loaded([image])
    print("PT2E QAT end-to-end smoke test passed")


if __name__ == "__main__":
    main()
