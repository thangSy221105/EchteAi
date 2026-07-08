import json

import torch
from PIL import Image

from pipelines.convnext_qat.anchors import infer_anchor_statistics
from pipelines.convnext_qat.data import CocoDetectionDataset, TiledCocoDetectionDataset
from pipelines.convnext_qat.tiling import predict_tiled, tile_origins


def _coco_fixture(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (120, 80), "black").save(image_dir / "sample.jpg")
    boxes = [[5, 5, 4, 4], [20, 10, 8, 6], [45, 20, 12, 10], [70, 30, 20, 15], [85, 45, 30, 25]]
    annotation = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 120, "height": 80}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {"id": index, "image_id": 1, "category_id": 1, "bbox": box, "area": box[2] * box[3]}
            for index, box in enumerate(boxes, 1)
        ],
    }
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    return image_dir, annotation_path


def test_anchor_statistics_are_sorted_and_data_driven(tmp_path):
    _, annotation_path = _coco_fixture(tmp_path)
    result = infer_anchor_statistics(annotation_path, target_min_size=80, max_size=120)
    assert result["boxes"] == 5
    assert len(result["anchor_sizes"]) == 5
    assert result["anchor_sizes"] == sorted(set(result["anchor_sizes"]))


def test_tiled_dataset_clips_boxes(tmp_path):
    image_dir, annotation_path = _coco_fixture(tmp_path)
    base = CocoDetectionDataset(image_dir, annotation_path, training=True)
    tiled = TiledCocoDetectionDataset(
        base, tile_size=64, overlap=0.25, keep_empty_probability=1.0,
        min_visible_fraction=0.25,
    )
    image, target = tiled[0]
    assert image.shape[-2] <= 64 and image.shape[-1] <= 64
    assert (target["boxes"] >= 0).all()
    assert (target["boxes"][:, 0::2] <= image.shape[-1]).all()
    assert (target["boxes"][:, 1::2] <= image.shape[-2]).all()


def test_tiled_prediction_global_nms():
    class Detector:
        def __call__(self, images):
            return [
                {
                    "boxes": torch.tensor([[20.0, 10.0, 40.0, 30.0]], device=image.device),
                    "scores": torch.tensor([0.9], device=image.device),
                    "labels": torch.tensor([1], device=image.device),
                }
                for image in images
            ]

    image = torch.zeros((3, 64, 96))
    output = predict_tiled(Detector(), image, tile_size=64, overlap=0.5, batch_size=2)
    assert len(tile_origins(96, 64, 0.5)) == 2
    assert len(output["boxes"]) == 2
    assert float(output["boxes"][:, 2].max()) > 64
