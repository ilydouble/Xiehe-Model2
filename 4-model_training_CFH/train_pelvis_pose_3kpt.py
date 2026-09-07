#!/usr/bin/env python3
"""Train the combined three-keypoint pelvis YOLO11 Pose model on AutoDL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


NUM_CLASSES = 1
KEYPOINT_COUNT = 3
EXPECTED_FIELDS = 1 + 4 + KEYPOINT_COUNT * 3
COORDINATE_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11)
VISIBILITY_INDICES = (6, 9, 12)


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    """Validate structure, labels, and patient isolation before training."""
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    yaml_text = data_yaml.read_text(encoding="utf-8")
    if "nc: 1" not in yaml_text or "kpt_shape: [3, 3]" not in yaml_text:
        raise ValueError("Dataset YAML must declare nc: 1 and kpt_shape: [3, 3]")
    if "flip_idx: [0, 2, 1]" not in yaml_text:
        raise ValueError("Dataset YAML must declare flip_idx: [0, 2, 1]")

    root = data_yaml.parent
    report: dict[str, Any] = {"root": str(root), "splits": {}, "objects": 0}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing image/label directory for split: {split}")
        images = {path.stem: path for path in image_dir.glob("*.png")}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if not images:
            raise ValueError(f"Split has no images: {split}")
        if set(images) != set(labels):
            image_only = sorted(set(images) - set(labels))[:5]
            label_only = sorted(set(labels) - set(images))[:5]
            raise ValueError(
                f"Image/label mismatch in {split}: image_only={image_only}, label_only={label_only}"
            )
        for path in labels.values():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError(f"{path}: expected exactly one pelvis object, got {len(lines)}")
            parts = lines[0].split()
            if len(parts) != EXPECTED_FIELDS:
                raise ValueError(f"{path}: expected {EXPECTED_FIELDS} fields, got {len(parts)}")
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"{path}: non-numeric label value") from exc
            if class_id != 0:
                raise ValueError(f"{path}: expected class id 0, got {class_id}")
            if any(not 0.0 <= values[index] <= 1.0 for index in COORDINATE_INDICES):
                raise ValueError(f"{path}: coordinate outside [0, 1]")
            if values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"{path}: non-positive bounding-box size")
            if [values[index] for index in VISIBILITY_INDICES] != [2.0, 2.0, 2.0]:
                raise ValueError(f"{path}: expected three visible keypoints")
        report["splits"][split] = {"images": len(images), "objects": len(labels)}
        report["objects"] += len(labels)

    manifest = root / "manifest.csv"
    if manifest.is_file():
        groups: dict[str, set[str]] = {}
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            group_id, split = (row.get("group_id") or "").strip(), (row.get("split") or "").strip()
            if not group_id or split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid manifest row: {row}")
            groups.setdefault(group_id, set()).add(split)
        leaked = sorted(group_id for group_id, splits in groups.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"Patient groups cross splits: {leaked[:5]}")
        if len(rows) != report["objects"]:
            raise ValueError(
                f"Manifest/label count mismatch: manifest={len(rows)}, labels={report['objects']}"
            )
        report["patients"] = len(groups)
    return report


def build_train_args(args: argparse.Namespace) -> dict[str, Any]:
    cache: str | bool = False if args.cache == "none" else args.cache
    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "patience": args.patience,
        "cos_lr": True,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 1.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.20,
        "degrees": 2.0,
        "translate": 0.05,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "rect": True,
        "cache": cache,
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": True,
        "close_mosaic": 0,
        "max_det": 5,
        "save": True,
        "save_period": args.save_period,
        "val": True,
        "plots": True,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=project_root / "datasets/yolo_pelvis_3kpt_all/data.yaml",
    )
    parser.add_argument(
        "--model", default="yolo11m-pose.pt",
        help=(
            "YOLO pose checkpoint/model name. The just-trained 23-class lateral-pose best.pt "
            "may be supplied to transfer its backbone; the 1-class/3-keypoint head is rebuilt."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=project_root / "runs/pose")
    parser.add_argument("--name", default="yolo11m_pelvis_3kpt_all")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--resume", nargs="?", const="auto",
        help="Resume from a checkpoint path; without a path, use project/name/weights/last.pt",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print settings without importing Ultralytics or starting training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_report = validate_dataset(args.data)
    train_args = build_train_args(args)
    print("Dataset validation passed:")
    print(json.dumps(dataset_report, ensure_ascii=False, indent=2))

    if args.resume:
        resume_path = (
            args.project / args.name / "weights/last.pt"
            if args.resume == "auto"
            else Path(args.resume)
        ).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resume checkpoint: {resume_path}")
    else:
        resume_path = None
        print(f"Initialization model: {args.model}")
        print("Training arguments:")
        print(json.dumps(train_args, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("Dry run complete; training was not started.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r 4-model_training_CFH/requirements.txt"
        ) from exc

    if resume_path is not None:
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(**train_args)


if __name__ == "__main__":
    main()
