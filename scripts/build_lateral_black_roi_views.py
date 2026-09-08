#!/usr/bin/env python3
"""Build train-only black-border ROI views for the lateral 20-class dataset.

Only continuous, notable black edge bands are removed. The crop is constrained
to contain every source bounding box and keypoint, so the full annotated spine
remains visible. Validation and test images are never derived or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


CLASS_NAMES = ["C2", "C7", *(f"T{i}" for i in range(1, 14)), *(f"L{i}" for i in range(1, 6))]
EXPECTED_FIELDS = 17
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SPLITS = ("train", "val", "test")
SIDES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class PoseObject:
    class_id: int
    bbox: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def load_appearance_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "1-check_data/analyze_image_appearance.py"
    spec = importlib.util.spec_from_file_location("lateral_image_appearance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load appearance analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


APPEARANCE = load_appearance_module()


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions: {path}")
    return width, height


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def validate_pose_object(obj: PoseObject, location: str) -> None:
    if not 0 <= obj.class_id < len(CLASS_NAMES):
        raise ValueError(f"{location}: class id outside range")
    cx, cy, width, height = obj.bbox
    if width <= 0 or height <= 0 or any(not 0 <= value <= 1 for value in obj.bbox):
        raise ValueError(f"{location}: invalid bbox")
    if cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6:
        raise ValueError(f"{location}: bbox x extent outside image")
    if cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
        raise ValueError(f"{location}: bbox y extent outside image")
    if len(obj.keypoints) != 4 or len({(x, y) for x, y, _ in obj.keypoints}) != 4:
        raise ValueError(f"{location}: expected four unique keypoints")
    if any(visibility != 2 or not 0 <= x <= 1 or not 0 <= y <= 1 for x, y, visibility in obj.keypoints):
        raise ValueError(f"{location}: invalid keypoint")
    points = [(x, y) for x, y, _ in obj.keypoints]
    signs = [cross(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) for i in range(4)]
    if any(abs(value) <= 1e-12 for value in signs) or not (all(value > 0 for value in signs) or all(value < 0 for value in signs)):
        raise ValueError(f"{location}: keypoints are not strictly convex")


def parse_pose_labels(path: Path) -> list[PoseObject]:
    objects: list[PoseObject] = []
    class_ids: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != EXPECTED_FIELDS:
            raise ValueError(f"{path}:{line_number}: expected {EXPECTED_FIELDS} fields")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        keypoints = tuple(
            (values[offset], values[offset + 1], int(values[offset + 2]))
            for offset in (4, 7, 10, 13)
        )
        obj = PoseObject(class_id, tuple(values[:4]), keypoints)  # type: ignore[arg-type]
        validate_pose_object(obj, f"{path}:{line_number}")
        if class_id in class_ids:
            raise ValueError(f"{path}:{line_number}: duplicate class")
        class_ids.add(class_id)
        objects.append(obj)
    if not objects:
        raise ValueError(f"Empty pose label: {path}")
    return objects


def format_pose_labels(objects: Sequence[PoseObject]) -> str:
    lines: list[str] = []
    for obj in objects:
        fields = [str(obj.class_id), *(f"{value:.8f}" for value in obj.bbox)]
        for x, y, visibility in obj.keypoints:
            fields.extend((f"{x:.8f}", f"{y:.8f}", str(visibility)))
        lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def object_bounds(obj: PoseObject, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, box_width, box_height = obj.bbox
    left = max(0.0, (cx - box_width / 2) * width)
    right = min(float(width), (cx + box_width / 2) * width)
    top = max(0.0, (cy - box_height / 2) * height)
    bottom = min(float(height), (cy + box_height / 2) * height)
    for x, y, visibility in obj.keypoints:
        if visibility > 0:
            left, right = min(left, x * width), max(right, x * width)
            top, bottom = min(top, y * height), max(bottom, y * height)
    return left, top, right, bottom


def all_target_bounds(objects: Sequence[PoseObject], width: int, height: int) -> tuple[float, float, float, float]:
    bounds = [object_bounds(obj, width, height) for obj in objects]
    return (
        min(value[0] for value in bounds), min(value[1] for value in bounds),
        max(value[2] for value in bounds), max(value[3] for value in bounds),
    )


def detect_black_ratios(
    image: Path,
    ffmpeg: str,
    sample_width: int,
    near_black_threshold: int,
    required_fraction: float,
) -> dict[str, float]:
    width, height, pixels = APPEARANCE.decode_small_grayscale(image, ffmpeg, sample_width)
    bands = APPEARANCE.edge_black_bands(width, height, pixels, near_black_threshold, required_fraction)
    return {
        "top": bands["top"] / height,
        "bottom": bands["bottom"] / height,
        "left": bands["left"] / width,
        "right": bands["right"] / width,
    }


def compute_black_crop(
    ratios: dict[str, float],
    objects: Sequence[PoseObject],
    width: int,
    height: int,
    notable_ratio: float = 0.01,
) -> CropBox | None:
    if not any(ratios[side] >= notable_ratio for side in SIDES):
        return None
    left = round(ratios["left"] * width) if ratios["left"] >= notable_ratio else 0
    right = width - (round(ratios["right"] * width) if ratios["right"] >= notable_ratio else 0)
    top = round(ratios["top"] * height) if ratios["top"] >= notable_ratio else 0
    bottom = height - (round(ratios["bottom"] * height) if ratios["bottom"] >= notable_ratio else 0)
    target_left, target_top, target_right, target_bottom = all_target_bounds(objects, width, height)
    left = min(left, max(0, math.floor(target_left)))
    top = min(top, max(0, math.floor(target_top)))
    right = max(right, min(width, math.ceil(target_right)))
    bottom = max(bottom, min(height, math.ceil(target_bottom)))
    box = CropBox(left, top, right, bottom)
    if box.width < 2 or box.height < 2:
        raise ValueError(f"Invalid black-border crop: {box}")
    if box == CropBox(0, 0, width, height):
        return None
    validate_crop_contains_all(box, objects, width, height)
    return box


def validate_crop_contains_all(box: CropBox, objects: Sequence[PoseObject], width: int, height: int) -> None:
    left, top, right, bottom = all_target_bounds(objects, width, height)
    if box.left > left + 1e-6 or box.top > top + 1e-6 or box.right < right - 1e-6 or box.bottom < bottom - 1e-6:
        raise ValueError("Black-border crop excludes an annotated target")


def transform_object(obj: PoseObject, box: CropBox, width: int, height: int) -> PoseObject:
    cx, cy, box_width, box_height = obj.bbox
    source_left = max(0.0, (cx - box_width / 2) * width)
    source_right = min(float(width), (cx + box_width / 2) * width)
    source_top = max(0.0, (cy - box_height / 2) * height)
    source_bottom = min(float(height), (cy + box_height / 2) * height)
    left = (source_left - box.left) / box.width
    right = (source_right - box.left) / box.width
    top = (source_top - box.top) / box.height
    bottom = (source_bottom - box.top) / box.height
    bbox = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    keypoints = tuple(
        ((x * width - box.left) / box.width, (y * height - box.top) / box.height, visibility)
        for x, y, visibility in obj.keypoints
    )
    transformed = PoseObject(obj.class_id, bbox, keypoints)
    validate_pose_object(transformed, "transformed label")
    return transformed


def load_manifest(dataset_root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with (dataset_root / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_image: dict[str, dict[str, str]] = {}
    patients: dict[str, set[str]] = {}
    for row in rows:
        split, patient, image = row["split"], row["patient_id"], row["image"]
        if split not in SPLITS or not patient or not image:
            raise ValueError(f"Invalid source manifest row: {row}")
        if image in by_image:
            raise ValueError(f"Duplicate source manifest image: {image}")
        by_image[image] = row
        patients.setdefault(patient, set()).add(split)
    leaked = [patient for patient, splits in patients.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Source patients cross splits: {leaked[:5]}")
    return rows, by_image


def plan_views(
    dataset_root: Path,
    *,
    workers: int = 8,
    ffmpeg: str = "ffmpeg",
    sample_width: int = 256,
    near_black_threshold: int = 5,
    required_fraction: float = 0.98,
    notable_ratio: float = 0.01,
    detector: Callable[[Path, str, int, int, float], dict[str, float]] = detect_black_ratios,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, by_image = load_manifest(dataset_root)
    images = sorted((dataset_root / "images/train").glob("*.png"))
    labels = {path.stem: path for path in (dataset_root / "labels/train").glob("*.txt")}
    if not images or {path.stem for path in images} != set(labels):
        raise ValueError("Source train image/label pairing is incomplete")

    def analyze(image: Path) -> dict[str, Any] | None:
        relative_image = str(image.relative_to(dataset_root))
        manifest_row = by_image.get(relative_image)
        if manifest_row is None or manifest_row["split"] != "train":
            raise ValueError(f"Train image missing from train manifest: {relative_image}")
        label = labels[image.stem]
        width, height = read_png_size(image)
        objects = parse_pose_labels(label)
        ratios = detector(image, ffmpeg, sample_width, near_black_threshold, required_fraction)
        box = compute_black_crop(ratios, objects, width, height, notable_ratio)
        if box is None:
            return None
        transformed = [transform_object(obj, box, width, height) for obj in objects]
        return {
            "source_image": image,
            "source_label": label,
            "source_image_relative": relative_image,
            "source_label_relative": str(label.relative_to(dataset_root)),
            "source_stem": image.stem,
            "source_batch": manifest_row["source_batch"],
            "patient_id": manifest_row["patient_id"],
            "width": width,
            "height": height,
            "objects": objects,
            "transformed": transformed,
            "ratios": ratios,
            "crop_box": box,
            "crop_area_fraction": box.width * box.height / (width * height),
            "output_stem": f"blackroi__{image.stem}",
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        analyzed = list(executor.map(analyze, images))
    records = [record for record in analyzed if record is not None]
    areas = sorted(record["crop_area_fraction"] for record in records)
    summary = {
        "source_train_images": len(images),
        "roi_train_images": len(records),
        "mixed_train_views": len(images) + len(records),
        "roi_patient_groups": len({record["patient_id"] for record in records}),
        "roi_by_batch": dict(sorted(Counter(record["source_batch"] for record in records).items())),
        "crop_area_fraction": {
            "min": min(areas),
            "median": areas[len(areas) // 2],
            "max": max(areas),
        },
    }
    return records, summary


def ffmpeg_crop(source: Path, output: Path, box: CropBox, ffmpeg: str) -> None:
    command = [
        ffmpeg, "-v", "error", "-threads", "1", "-i", str(source),
        "-vf", f"crop={box.width}:{box.height}:{box.left}:{box.top}",
        "-frames:v", "1", "-compression_level", "4", "-y", str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)


def audit_leakage(dataset_root: Path, output_root: Path, *, write_report: bool = False) -> dict[str, Any]:
    source_rows, source_by_image = load_manifest(dataset_root)
    with (output_root / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        roi_rows = list(csv.DictReader(handle))
    errors: list[str] = []
    source_hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in source_rows:
        image = dataset_root / row["image"]
        if not image.is_file():
            errors.append(f"missing source image: {row['image']}")
            continue
        source_hashes_by_split[row["split"]].add(sha256_file(image))
    roi_hashes: set[str] = set()
    roi_patients: set[str] = set()
    for row in roi_rows:
        source = source_by_image.get(row["source_image"])
        if row["source_split"] != "train" or source is None or source["split"] != "train":
            errors.append(f"ROI source is not train: {row['source_image']}")
        elif source["patient_id"] != row["patient_id"]:
            errors.append(f"ROI patient mismatch: {row['output_image']}")
        source_path = dataset_root / row["source_image"]
        output_image = output_root / row["output_image"]
        output_label = output_root / row["output_label"]
        if not source_path.is_file() or sha256_file(source_path) != row["source_image_sha256"]:
            errors.append(f"source hash mismatch: {row['source_image']}")
        if not output_image.is_file() or sha256_file(output_image) != row["output_image_sha256"]:
            errors.append(f"ROI image hash mismatch: {row['output_image']}")
        else:
            roi_hashes.add(row["output_image_sha256"])
        if not output_label.is_file() or sha256_file(output_label) != row["output_label_sha256"]:
            errors.append(f"ROI label hash mismatch: {row['output_label']}")
        else:
            try:
                parse_pose_labels(output_label)
            except ValueError as exc:
                errors.append(str(exc))
        roi_patients.add(row["patient_id"])
    exact_holdout = roi_hashes & (source_hashes_by_split["val"] | source_hashes_by_split["test"])
    if exact_holdout:
        errors.append(f"ROI exactly matches holdout images: {len(exact_holdout)}")
    for split in ("val", "test"):
        if (output_root / f"images/{split}").exists() or (output_root / f"labels/{split}").exists():
            errors.append(f"ROI output contains forbidden {split} directory")
    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "counts": {
            "source_images": len(source_rows),
            "roi_images": len(roi_rows),
            "roi_patient_groups": len(roi_patients),
            "roi_exact_matches_to_val_test": len(exact_holdout),
        },
        "errors": errors,
    }
    if write_report:
        (output_root / "leakage_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    workers: int = 8,
    ffmpeg: str = "ffmpeg",
    sample_width: int = 256,
    near_black_threshold: int = 5,
    required_fraction: float = 0.98,
    notable_ratio: float = 0.01,
    apply: bool = False,
    detector: Callable[[Path, str, int, int, float], dict[str, float]] = detect_black_ratios,
    cropper: Callable[[Path, Path, CropBox, str], None] = ffmpeg_crop,
) -> dict[str, Any]:
    dataset_root, output_root = dataset_root.resolve(), output_root.resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    records, summary = plan_views(
        dataset_root, workers=workers, ffmpeg=ffmpeg, sample_width=sample_width,
        near_black_threshold=near_black_threshold, required_fraction=required_fraction,
        notable_ratio=notable_ratio, detector=detector,
    )
    if not apply:
        return {**summary, "mode": "dry-run", "output_written": False}
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    image_dir, label_dir = staging / "images/train", staging / "labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    def materialize(record: dict[str, Any]) -> dict[str, Any]:
        box: CropBox = record["crop_box"]
        output_image = image_dir / f"{record['output_stem']}.png"
        output_label = label_dir / f"{record['output_stem']}.txt"
        cropper(record["source_image"], output_image, box, ffmpeg)
        output_label.write_text(format_pose_labels(record["transformed"]), encoding="utf-8")
        if read_png_size(output_image) != (box.width, box.height):
            raise ValueError(f"Saved crop size mismatch: {output_image}")
        parse_pose_labels(output_label)
        return {
            "source_batch": record["source_batch"],
            "patient_id": record["patient_id"],
            "source_split": "train",
            "source_image": record["source_image_relative"],
            "source_label": record["source_label_relative"],
            "source_width": record["width"],
            "source_height": record["height"],
            "source_image_sha256": sha256_file(record["source_image"]),
            **{f"black_{side}_ratio": f"{record['ratios'][side]:.8f}" for side in SIDES},
            "crop_box": json.dumps(asdict(box), separators=(",", ":")),
            "crop_width": box.width,
            "crop_height": box.height,
            "crop_area_fraction": f"{record['crop_area_fraction']:.8f}",
            "objects": len(record["transformed"]),
            "output_image": str(output_image.relative_to(staging)),
            "output_label": str(output_label.relative_to(staging)),
            "output_image_sha256": sha256_file(output_image),
            "output_label_sha256": sha256_file(output_label),
        }

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = []
            for index, row in enumerate(executor.map(materialize, records), 1):
                rows.append(row)
                if index % 25 == 0 or index == len(records):
                    print(f"Created {index}/{len(records)} black-border ROI views", flush=True)
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "schema_version": 1,
            "source_dataset": str(dataset_root),
            "output": str(output_root),
            "configuration": {
                "sample_width": sample_width,
                "near_black_threshold": near_black_threshold,
                "required_near_black_fraction": required_fraction,
                "notable_side_ratio": notable_ratio,
                "crop_rule": "remove only notable continuous black edge bands while containing every bbox and keypoint",
            },
            "summary": summary,
            "leakage_contract": {
                "derived_splits": ["train"],
                "validation_and_test_views": "original_only",
                "source_patient_split_immutable": True,
            },
        }
        (staging / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            "# Lateral 20-class black-border ROI views\n\n"
            "Train-only derived views with continuous black edge bands removed. All vertebral boxes and four keypoints are transformed.\n\n"
            f"- Source train images: {summary['source_train_images']}\n"
            f"- ROI train images: {summary['roi_train_images']}\n"
            f"- Original + ROI train views: {summary['mixed_train_views']}\n"
            "- Validation/test: original images only\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "datasets/yolo_lateral_reviewed_combined_20cls")
    parser.add_argument("--output-root", type=Path, default=root / "datasets/yolo_lateral_reviewed_combined_20cls_black_roi")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--sample-width", type=int, default=256)
    parser.add_argument("--near-black-threshold", type=int, default=5)
    parser.add_argument("--required-fraction", type=float, default=0.98)
    parser.add_argument("--notable-ratio", type=float, default=0.01)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        result = audit_leakage(args.dataset_root.resolve(), args.output_root.resolve(), write_report=True)
    else:
        result = build_dataset(
            args.dataset_root, args.output_root, workers=args.workers, ffmpeg=args.ffmpeg,
            sample_width=args.sample_width, near_black_threshold=args.near_black_threshold,
            required_fraction=args.required_fraction, notable_ratio=args.notable_ratio,
            apply=args.apply,
        )
        if args.apply:
            result["leakage_audit"] = audit_leakage(args.dataset_root.resolve(), args.output_root.resolve(), write_report=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
