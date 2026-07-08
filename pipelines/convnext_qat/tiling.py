"""Overlapping tiled inference with class-aware global NMS."""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms


def tile_origins(length, tile_size, overlap):
    tile_size = min(int(tile_size), int(length))
    stride = max(1, int(round(tile_size * (1.0 - float(overlap)))))
    origins = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = max(length - tile_size, 0)
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


@torch.inference_mode()
def predict_tiled(model, image, tile_size=960, overlap=0.25, batch_size=1,
                  score_threshold=0.05, nms_threshold=0.5, max_detections=300):
    if not 0.0 <= overlap < 1.0:
        raise ValueError("tile overlap must be in [0, 1)")
    height, width = image.shape[-2:]
    windows, offsets = [], []
    for top in tile_origins(height, tile_size, overlap):
        for left in tile_origins(width, tile_size, overlap):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            windows.append(image[:, top:bottom, left:right])
            offsets.append((left, top))

    boxes, scores, labels = [], [], []
    for start in range(0, len(windows), int(batch_size)):
        outputs = model(windows[start:start + int(batch_size)])
        for output, (left, top) in zip(outputs, offsets[start:start + int(batch_size)]):
            keep = output["scores"] >= float(score_threshold)
            shifted = output["boxes"][keep].clone()
            shifted[:, 0::2] += left
            shifted[:, 1::2] += top
            boxes.append(shifted)
            scores.append(output["scores"][keep])
            labels.append(output["labels"][keep])
    device = image.device
    if not boxes or not any(len(item) for item in boxes):
        return {
            "boxes": torch.empty((0, 4), device=device),
            "scores": torch.empty((0,), device=device),
            "labels": torch.empty((0,), dtype=torch.int64, device=device),
        }
    boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
    keep = batched_nms(boxes, scores, labels, float(nms_threshold))[:int(max_detections)]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


class TiledDetector:
    """Minimal callable adapter accepted by the existing evaluation loop."""

    def __init__(self, model, **settings):
        self.model = model
        self.settings = settings

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, images):
        return [predict_tiled(self.model, image, **self.settings) for image in images]


def validation_detector(model, config):
    """Apply the configured deployment tiling path during best-checkpoint selection."""
    if not config.get("validation", {}).get("tiled", False):
        return model
    tiling = config.get("inference", {}).get("tiling", {})
    return TiledDetector(
        model,
        tile_size=tiling.get("tile_size", 960),
        overlap=tiling.get("overlap", 0.25),
        batch_size=tiling.get("batch_size", 1),
        score_threshold=tiling.get("score_threshold", 0.05),
        nms_threshold=tiling.get("nms_threshold", 0.5),
        max_detections=tiling.get("max_detections", 300),
    )
