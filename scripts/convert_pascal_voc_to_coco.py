#!/usr/bin/env python3
"""Convert Pascal VOC 2007/2012 annotations into COCO-format JSON files.

Typical usage for the combined VOC benchmark protocol:

    python scripts/convert_pascal_voc_to_coco.py \
        --voc-root /kaggle/input/pascal-voc-2007-and-2012/VOCdevkit \
        --output-dir /kaggle/working/pascal_voc_coco \
        --train-sets VOC2007:trainval VOC2012:trainval \
        --val-sets VOC2007:test

The generated JSON files are compatible with the repo's existing COCO loader.
Images stay in-place; the converter writes relative paths such as
VOC2007/JPEGImages/000001.jpg into the COCO `file_name` field.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_CLASS_TO_ID = {name: index + 1 for index, name in enumerate(VOC_CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc-root",
        required=True,
        help="Path to VOCdevkit or its parent directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where instances_train.json and instances_val.json will be written.",
    )
    parser.add_argument(
        "--train-sets",
        nargs="+",
        default=["VOC2007:trainval", "VOC2012:trainval"],
        help="List of YEAR:SPLIT pairs for the training set.",
    )
    parser.add_argument(
        "--val-sets",
        nargs="+",
        default=["VOC2007:test"],
        help="List of YEAR:SPLIT pairs for the validation/test set.",
    )
    parser.add_argument(
        "--keep-difficult",
        action="store_true",
        help="Keep difficult objects instead of dropping them.",
    )
    return parser.parse_args()


def normalize_voc_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "VOC2007").is_dir() or (path / "VOC2012").is_dir():
        return path
    if (path / "VOCdevkit").is_dir():
        return (path / "VOCdevkit").resolve()
    raise FileNotFoundError(
        f"Could not find VOC2007/VOC2012 under {path}. "
        "Expected either VOCdevkit itself or a parent directory containing it."
    )


def parse_year_split(token: str) -> tuple[str, str]:
    if ":" not in token:
        raise ValueError(f"Invalid YEAR:SPLIT token: {token!r}")
    year, split = token.split(":", 1)
    year = year.strip()
    split = split.strip()
    if year not in {"VOC2007", "VOC2012"}:
        raise ValueError(f"Unsupported year {year!r}; expected VOC2007 or VOC2012")
    if not split:
        raise ValueError(f"Missing split in token: {token!r}")
    return year, split


def load_image_ids(voc_root: Path, year: str, split: str) -> list[str]:
    split_path = voc_root / year / "ImageSets" / "Main" / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    ids = []
    for raw in split_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        ids.append(parts[0])
    if not ids:
        raise ValueError(f"No image ids found in split file: {split_path}")
    return ids


def parse_annotation(xml_path: Path, keep_difficult: bool) -> tuple[int, int, list[dict]]:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects: list[dict] = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name")
        if cls_name not in VOC_CLASS_TO_ID:
            continue
        difficult = int(obj.findtext("difficult", default="0"))
        if difficult and not keep_difficult:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
        x1 = max(0.0, xmin - 1.0)
        y1 = max(0.0, ymin - 1.0)
        x2 = max(x1, xmax - 1.0)
        y2 = max(y1, ymax - 1.0)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0.0 or h <= 0.0:
            continue
        objects.append(
            {
                "category_id": VOC_CLASS_TO_ID[cls_name],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "difficult": difficult,
            }
        )
    return width, height, objects


def build_coco_json(voc_root: Path, set_tokens: list[str], keep_difficult: bool) -> tuple[dict, dict]:
    image_entries = []
    annotation_entries = []
    image_id = 1
    annotation_id = 1
    stats = {"images": 0, "annotations": 0}

    for token in set_tokens:
        year, split = parse_year_split(token)
        ids = load_image_ids(voc_root, year, split)
        for stem in ids:
            xml_path = voc_root / year / "Annotations" / f"{stem}.xml"
            image_path = voc_root / year / "JPEGImages" / f"{stem}.jpg"
            if not xml_path.is_file():
                raise FileNotFoundError(f"Annotation not found: {xml_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            width, height, objects = parse_annotation(xml_path, keep_difficult)
            image_entries.append(
                {
                    "id": image_id,
                    "file_name": f"{year}/JPEGImages/{stem}.jpg",
                    "width": width,
                    "height": height,
                }
            )
            for obj in objects:
                annotation_entries.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": obj["category_id"],
                        "bbox": obj["bbox"],
                        "area": obj["area"],
                        "iscrowd": obj["iscrowd"],
                    }
                )
                annotation_id += 1

            image_id += 1
            stats["images"] += 1
            stats["annotations"] += len(objects)

    coco = {
        "images": image_entries,
        "annotations": annotation_entries,
        "categories": [{"id": VOC_CLASS_TO_ID[name], "name": name} for name in VOC_CLASSES],
    }
    return coco, stats


def main():
    args = parse_args()
    voc_root = normalize_voc_root(Path(args.voc_root))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_json, train_stats = build_coco_json(voc_root, args.train_sets, args.keep_difficult)
    val_json, val_stats = build_coco_json(voc_root, args.val_sets, args.keep_difficult)

    train_path = output_dir / "instances_train.json"
    val_path = output_dir / "instances_val.json"
    train_path.write_text(json.dumps(train_json), encoding="utf-8")
    val_path.write_text(json.dumps(val_json), encoding="utf-8")

    summary = {
        "voc_root": str(voc_root),
        "train_sets": args.train_sets,
        "val_sets": args.val_sets,
        "keep_difficult": bool(args.keep_difficult),
        "train": train_stats,
        "val": val_stats,
        "train_annotations": str(train_path),
        "val_annotations": str(val_path),
        "image_root": str(voc_root),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
