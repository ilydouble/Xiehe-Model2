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


def _yaml_scalar(value: str) -> str:
    return value.split("#", 1)[0].strip().strip("'\"")


def dataset_paths(data_yaml: Path) -> tuple[Path, dict[str, list[Path]]]:
    """Parse the small path subset used by our dataset YAML without PyYAML."""
    text = data_yaml.read_text(encoding="utf-8")
    root_value = ""
    entries: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    current_list = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")) and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_list = ""
            if key == "path":
                root_value = _yaml_scalar(value)
            elif key in entries:
                scalar = _yaml_scalar(value)
                if scalar:
                    entries[key].append(scalar)
                else:
                    current_list = key
        elif current_list and stripped.startswith("-"):
            scalar = _yaml_scalar(stripped[1:])
            if scalar:
                entries[current_list].append(scalar)
    root = (data_yaml.parent / root_value).resolve() if root_value else data_yaml.parent.resolve()
    for split in entries:
        if not entries[split]:
            entries[split] = [f"images/{split}"]
    return root, {
        split: [(root / value).resolve() for value in values]
        for split, values in entries.items()
    }


def label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image path has no images component: {image_dir}")
    parts[indices[-1]] = "labels"
    return Path(*parts)


def _validate_manifest_groups(path: Path) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, set[str]] = {}
    for row in rows:
        group_id = (row.get("group_id") or "").strip()
        split = (row.get("split") or row.get("source_split") or "").strip()
        if not group_id or split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid manifest row: {row}")
        groups.setdefault(group_id, set()).add(split)
    leaked = sorted(group_id for group_id, splits in groups.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Patient groups cross splits: {leaked[:5]}")
    return rows, groups


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

    root, split_paths = dataset_paths(data_yaml)
    report: dict[str, Any] = {"root": str(root), "splits": {}, "objects": 0}
    dataset_roots: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_total = 0
        views: dict[str, dict[str, int]] = {}
        all_stems: set[str] = set()
        for image_dir in split_paths[split]:
            label_dir = label_dir_for(image_dir)
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"Missing image/label directory for split: {split}: {image_dir}")
            dataset_root = image_dir.parent.parent
            dataset_roots[dataset_root.name] = dataset_root
            images = {path.stem: path for path in image_dir.glob("*.png")}
            labels = {path.stem: path for path in label_dir.glob("*.txt")}
            if not images:
                raise ValueError(f"Split has no images: {split}: {image_dir}")
            if set(images) != set(labels):
                image_only = sorted(set(images) - set(labels))[:5]
                label_only = sorted(set(labels) - set(images))[:5]
                raise ValueError(
                    f"Image/label mismatch in {split}: image_only={image_only}, label_only={label_only}"
                )
            duplicated = sorted(all_stems & set(images))
            if duplicated:
                raise ValueError(f"Duplicate stems across {split} views: {duplicated[:5]}")
            all_stems.update(images)
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
            count = len(images)
            views[dataset_root.name] = {"images": count, "objects": count}
            split_total += count
        report["splits"][split] = {"images": split_total, "objects": split_total, "views": views}
        report["objects"] += split_total

    source_root = dataset_roots.get("yolo_pelvis_3kpt_all", data_yaml.parent)
    source_manifest = source_root / "manifest.csv"
    if source_manifest.is_file():
        rows, groups = _validate_manifest_groups(source_manifest)
        source_objects = sum(
            view["objects"]
            for split in report["splits"].values()
            for name, view in split["views"].items()
            if name == source_root.name
        )
        if len(rows) != source_objects:
            raise ValueError(
                f"Manifest/label count mismatch: manifest={len(rows)}, source_labels={source_objects}"
            )
        report["patients"] = len(groups)
        source_train_groups = {
            row["group_id"] for row in rows if (row.get("split") or "").strip() == "train"
        }
    else:
        source_train_groups = set()

    roi_root = dataset_roots.get("yolo_pelvis_3kpt_roi_views")
    if roi_root is not None:
        roi_manifest = roi_root / "manifest.csv"
        if not roi_manifest.is_file():
            raise FileNotFoundError(f"Missing ROI manifest: {roi_manifest}")
        roi_rows, roi_groups = _validate_manifest_groups(roi_manifest)
        roi_count = report["splits"]["train"]["views"][roi_root.name]["images"]
        if len(roi_rows) != roi_count:
            raise ValueError(f"ROI manifest/image count mismatch: {len(roi_rows)} != {roi_count}")
        if any((row.get("source_split") or "").strip() != "train" for row in roi_rows):
            raise ValueError("ROI manifest contains a non-train source")
        if source_train_groups and not set(roi_groups).issubset(source_train_groups):
            raise ValueError("ROI patient group is not contained in source train split")
        if (roi_root / "images/val").exists() or (roi_root / "images/test").exists():
            raise ValueError("ROI dataset must not contain validation or test image directories")
        report["roi_patients"] = len(roi_groups)
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
        default=project_root / "4-model_training_CFH/pelvis_3kpt_roi_mixed.yaml",
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
