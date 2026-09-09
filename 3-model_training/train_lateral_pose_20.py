#!/usr/bin/env python3
"""Train the reviewed combined 20-class lateral-spine YOLO Pose model."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLASS_NAMES = ["C2", "C7", *(f"T{i}" for i in range(1, 14)), *(f"L{i}" for i in range(1, 6))]
NUM_CLASSES = len(CLASS_NAMES)
KEYPOINT_COUNT = 4
EXPECTED_FIELDS = 1 + 4 + KEYPOINT_COUNT * 3
FOCUS_CLASS_IDS = [0, 1]


def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def validate_convex_keypoints(points: list[tuple[float, float]], location: str) -> None:
    if len(set(points)) != 4:
        raise ValueError(f"{location}: keypoints are not unique")
    signs = [cross(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) for i in range(4)]
    if any(abs(value) <= 1e-12 for value in signs) or not (all(value > 0 for value in signs) or all(value < 0 for value in signs)):
        raise ValueError(f"{location}: keypoints are not a strictly convex quadrilateral")


def parse_yaml_names(yaml_text: str) -> list[str]:
    pairs = [
        (int(match.group(1)), match.group(2))
        for match in re.finditer(r"^\s+(\d+):\s+([A-Za-z0-9_-]+)\s*$", yaml_text, re.MULTILINE)
    ]
    return [name for _, name in sorted(pairs)]


def yaml_scalar(value: str) -> str:
    return value.split("#", 1)[0].strip().strip("'\"")


def dataset_paths(data_yaml: Path) -> tuple[Path, dict[str, list[Path]]]:
    """Parse the path/train/val/test subset used by the dataset YAML."""
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
                root_value = yaml_scalar(value)
            elif key in entries:
                scalar = yaml_scalar(value)
                if scalar:
                    entries[key].append(scalar)
                else:
                    current_list = key
        elif current_list and stripped.startswith("-"):
            scalar = yaml_scalar(stripped[1:])
            if scalar:
                entries[current_list].append(scalar)
    root = (data_yaml.parent / root_value).resolve() if root_value else data_yaml.parent.resolve()
    for split in entries:
        if not entries[split]:
            entries[split] = [f"images/{split}"]
    return root, {split: [(root / value).resolve() for value in values] for split, values in entries.items()}


def label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image path has no images component: {image_dir}")
    parts[indices[-1]] = "labels"
    return Path(*parts)


def validate_patient_splits(root: Path) -> dict[str, int]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    patients: dict[str, set[str]] = defaultdict(set)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        patients[row["split"]].add(row["patient_id"])
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[first] & patients[second]
        if overlap:
            raise ValueError(f"Patient leakage between {first}/{second}: {sorted(overlap)[:5]}")
    return {split: len(values) for split, values in patients.items()}


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    """Validate class contract, pairs, geometry and patient isolation before GPU use."""
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    yaml_text = data_yaml.read_text(encoding="utf-8")
    if "nc: 20" not in yaml_text or "kpt_shape: [4, 3]" not in yaml_text:
        raise ValueError("Dataset YAML must declare nc: 20 and kpt_shape: [4, 3]")
    names = parse_yaml_names(yaml_text)
    if names != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {names}")
    if any(name in names for name in ("C3", "C4", "C5", "C6")):
        raise ValueError("C3-C6 must not be present")

    root, split_paths = dataset_paths(data_yaml)
    report: dict[str, Any] = {"root": str(root), "splits": {}, "objects": 0}
    all_ids: set[int] = set()
    split_class_counts: dict[str, Counter[int]] = {}
    dataset_roots: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        counts: Counter[int] = Counter()
        images_total = 0
        views: dict[str, dict[str, int]] = {}
        all_stems: set[str] = set()
        for image_dir in split_paths[split]:
            label_dir = label_dir_for(image_dir)
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"Missing image/label directory for split: {split}: {image_dir}")
            dataset_root = image_dir.parent.parent
            dataset_roots[dataset_root.name] = dataset_root
            images = {path.stem: path for path in image_dir.glob("*.png") if not path.name.startswith("._")}
            labels = {path.stem: path for path in label_dir.glob("*.txt") if not path.name.startswith("._")}
            if set(images) != set(labels):
                raise ValueError(f"Image/label mismatch in {split}: image_only={sorted(set(images) - set(labels))[:5]}, label_only={sorted(set(labels) - set(images))[:5]}")
            duplicated = sorted(all_stems & set(images))
            if duplicated:
                raise ValueError(f"Duplicate stems across {split} views: {duplicated[:5]}")
            all_stems.update(images)
            view_objects = 0
            for path in labels.values():
                lines = path.read_text(encoding="utf-8").splitlines()
                if not lines:
                    raise ValueError(f"Empty label file: {path}")
                image_classes: set[int] = set()
                for line_number, line in enumerate(lines, 1):
                    location = f"{path}:{line_number}"
                    parts = line.split()
                    if len(parts) != EXPECTED_FIELDS:
                        raise ValueError(f"{location}: expected {EXPECTED_FIELDS} fields, got {len(parts)}")
                    try:
                        class_id = int(parts[0])
                        values = [float(value) for value in parts[1:]]
                    except ValueError as exc:
                        raise ValueError(f"{location}: non-numeric value") from exc
                    if not 0 <= class_id < NUM_CLASSES:
                        raise ValueError(f"{location}: class id {class_id} is outside [0, {NUM_CLASSES - 1}]")
                    if class_id in image_classes:
                        raise ValueError(f"{location}: duplicate class in one image")
                    image_classes.add(class_id)
                    coordinate_indices = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14]
                    if any(not 0.0 <= values[index] <= 1.0 for index in coordinate_indices):
                        raise ValueError(f"{location}: coordinate outside [0, 1]")
                    if values[2] <= 0 or values[3] <= 0:
                        raise ValueError(f"{location}: non-positive bounding-box size")
                    if [values[index] for index in (6, 9, 12, 15)] != [2.0] * 4:
                        raise ValueError(f"{location}: expected four visible keypoints")
                    points = [(values[index], values[index + 1]) for index in (4, 7, 10, 13)]
                    validate_convex_keypoints(points, location)
                    counts[class_id] += 1
                    all_ids.add(class_id)
                    view_objects += 1
            views[dataset_root.name] = {"images": len(images), "objects": view_objects}
            images_total += len(images)
        split_class_counts[split] = counts
        split_objects = sum(counts.values())
        report["splits"][split] = {
            "images": images_total,
            "objects": split_objects,
            "classes": {CLASS_NAMES[class_id]: counts[class_id] for class_id in range(NUM_CLASSES)},
            "views": views,
        }
        report["objects"] += split_objects
    if all_ids != set(range(NUM_CLASSES)):
        raise ValueError(f"Dataset does not contain every class id: found {sorted(all_ids)}")
    for split in ("val", "test"):
        if any(split_class_counts[split][class_id] == 0 for class_id in FOCUS_CLASS_IDS):
            raise ValueError(f"{split} must contain both C2 and C7 for focused evaluation")
    source_root = dataset_roots.get("yolo_lateral_reviewed_combined_20cls", data_yaml.parent)
    report["patient_groups"] = validate_patient_splits(source_root)
    roi_root = dataset_roots.get("yolo_lateral_reviewed_combined_20cls_black_roi")
    if roi_root is not None:
        with (source_root / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        source_train_patients = {row["patient_id"] for row in source_rows if row["split"] == "train"}
        with (roi_root / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            roi_rows = list(csv.DictReader(handle))
        if len(roi_rows) != report["splits"]["train"]["views"][roi_root.name]["images"]:
            raise ValueError("ROI manifest/image count mismatch")
        if any(row["source_split"] != "train" for row in roi_rows):
            raise ValueError("ROI manifest contains a non-train source")
        roi_patients = {row["patient_id"] for row in roi_rows}
        if not roi_patients <= source_train_patients:
            raise ValueError("ROI patient is not contained in source train split")
        if (roi_root / "images/val").exists() or (roi_root / "images/test").exists():
            raise ValueError("ROI dataset must not contain validation or test directories")
        report["roi_patient_groups"] = len(roi_patients)
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
        "pretrained": False,
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
        "max_det": 30,
        "save": True,
        "save_period": args.save_period,
        "val": True,
        "plots": True,
    }


def build_test_args(args: argparse.Namespace, *, focused: bool) -> dict[str, Any]:
    """Build final test arguments; full-spine evaluation is the primary report."""
    result: dict[str, Any] = {
        "data": str(args.data.resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": f"{args.name}_test_C2_C7" if focused else f"{args.name}_test_all_spine",
        "plots": True,
    }
    if focused:
        result["classes"] = FOCUS_CLASS_IDS
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "3-model_training/lateral_pose_20_black_roi_mixed.yaml")
    parser.add_argument("--model", default="yolo11l-pose.yaml", help="Model architecture or checkpoint; defaults to randomly initialized YOLO11l-Pose")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=root / "3-model_training/runs/pose")
    parser.add_argument("--name", default="yolo11l_lateral_20cls_scratch_best")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true", help="Skip both full-spine and focused C2/C7 final test evaluations")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume from a checkpoint path; without a path use project/name/weights/last.pt")
    parser.add_argument("--dry-run", action="store_true", help="Validate all labels and print settings without importing Ultralytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_report = validate_dataset(args.data)
    train_args = build_train_args(args)
    print("Dataset validation passed:")
    print(json.dumps(dataset_report, ensure_ascii=False, indent=2))
    resume_path = None
    if args.resume:
        resume_path = (args.project / args.name / "weights/last.pt" if args.resume == "auto" else Path(args.resume)).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resume checkpoint: {resume_path}")
    else:
        if not Path(args.model).is_file() and args.model.endswith(".pt") and "/" in args.model:
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.model}")
        print(f"Initialization model: {args.model}")
        print(json.dumps(train_args, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete; training and final full-spine/C2-C7 tests were not started.")
        return
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r 3-model_training/requirements.txt") from exc
    if resume_path is not None:
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(**train_args)
    best_path = args.project.resolve() / args.name / "weights/best.pt"
    print(f"Best checkpoint: {best_path}")
    if not args.skip_final_test:
        best_model = YOLO(str(best_path))
        full_metrics = best_model.val(**build_test_args(args, focused=False))
        print("Primary final test metrics for all labeled spine classes:")
        print(json.dumps(getattr(full_metrics, "results_dict", {}), ensure_ascii=False, indent=2))
        focused_metrics = best_model.val(**build_test_args(args, focused=True))
        print("Auxiliary final test metrics restricted to C2 and C7:")
        print(json.dumps(getattr(focused_metrics, "results_dict", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
