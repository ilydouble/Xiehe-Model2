#!/usr/bin/env python3
"""Build train-only ROI views for the combined three-keypoint pelvis dataset.

The source patient split is immutable.  Every derived image comes from one
source train image, while validation and test images are never cropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_FIELDS = 14
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PoseLabel:
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pose_label(path: Path) -> PoseLabel:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one pelvis object, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != EXPECTED_FIELDS:
        raise ValueError(f"{path}: expected {EXPECTED_FIELDS} fields, found {len(fields)}")
    try:
        class_value = float(fields[0])
        values = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric label value") from exc
    if class_value != 0:
        raise ValueError(f"{path}: expected class 0, found {fields[0]}")
    bbox = tuple(values[:4])
    if len(bbox) != 4 or any(not 0.0 <= value <= 1.0 for value in bbox):
        raise ValueError(f"{path}: invalid bounding box")
    cx, cy, bbox_w, bbox_h = bbox
    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError(f"{path}: non-positive bounding box")
    if cx - bbox_w / 2 < -1e-6 or cx + bbox_w / 2 > 1 + 1e-6:
        raise ValueError(f"{path}: bounding box x extent outside image")
    if cy - bbox_h / 2 < -1e-6 or cy + bbox_h / 2 > 1 + 1e-6:
        raise ValueError(f"{path}: bounding box y extent outside image")
    keypoints = []
    for offset in range(4, len(values), 3):
        x, y, visibility = values[offset:offset + 3]
        if visibility != 2 or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"{path}: expected three visible in-range keypoints")
        keypoints.append((x, y, int(visibility)))
    if len(keypoints) != 3:
        raise ValueError(f"{path}: expected three keypoints")
    return PoseLabel(0, bbox, tuple(keypoints))  # type: ignore[arg-type]


def format_pose_label(label: PoseLabel) -> str:
    fields = [str(label.class_id), *(f"{value:.8f}" for value in label.bbox)]
    for x, y, visibility in label.keypoints:
        fields.extend((f"{x:.8f}", f"{y:.8f}", str(visibility)))
    return " ".join(fields) + "\n"


def target_bounds(label: PoseLabel, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bbox_w, bbox_h = label.bbox
    left = (cx - bbox_w / 2) * width
    right = (cx + bbox_w / 2) * width
    top = (cy - bbox_h / 2) * height
    bottom = (cy + bbox_h / 2) * height
    for x, y, visibility in label.keypoints:
        if visibility > 0:
            left, right = min(left, x * width), max(right, x * width)
            top, bottom = min(top, y * height), max(bottom, y * height)
    return max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)


def deterministic_rng(filename: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def compute_crop_box(
    label: PoseLabel,
    width: int,
    height: int,
    filename: str,
    seed: int = 20260907,
    margin: float = 2.5,
    shift_jitter: float = 0.05,
    scale_jitter: float = 0.10,
    safety_margin: float = 0.03,
    minimum_size: int = 64,
) -> CropBox:
    if width < 2 or height < 2 or margin < 0 or minimum_size < 2:
        raise ValueError("Invalid image size or crop configuration")
    target_left, target_top, target_right, target_bottom = target_bounds(label, width, height)
    target_w = max(1.0, target_right - target_left)
    target_h = max(1.0, target_bottom - target_top)
    rng = deterministic_rng(filename, seed)
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    crop_w = min(float(width), max(float(minimum_size), target_w * (1 + 2 * margin) * scale))
    crop_h = min(float(height), max(float(minimum_size), target_h * (1 + 2 * margin) * scale))
    center_x = (target_left + target_right) / 2 + rng.uniform(-shift_jitter, shift_jitter) * target_w
    center_y = (target_top + target_bottom) / 2 + rng.uniform(-shift_jitter, shift_jitter) * target_h
    guard_x, guard_y = safety_margin * target_w, safety_margin * target_h
    left = min(center_x - crop_w / 2, target_left - guard_x)
    right = max(center_x + crop_w / 2, target_right + guard_x)
    top = min(center_y - crop_h / 2, target_top - guard_y)
    bottom = max(center_y + crop_h / 2, target_bottom + guard_y)
    box = CropBox(
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(width, math.ceil(right)),
        min(height, math.ceil(bottom)),
    )
    if box.width < 2 or box.height < 2:
        raise ValueError(f"Computed invalid crop: {box}")
    validate_crop_contains_target(box, label, width, height)
    return box


def validate_crop_contains_target(box: CropBox, label: PoseLabel, width: int, height: int) -> None:
    left, top, right, bottom = target_bounds(label, width, height)
    if not (box.left <= left + 1e-6 and box.top <= top + 1e-6):
        raise ValueError("Crop excludes target minimum extent")
    if not (box.right >= right - 1e-6 and box.bottom >= bottom - 1e-6):
        raise ValueError("Crop excludes target maximum extent")


def transform_label(label: PoseLabel, box: CropBox, width: int, height: int) -> PoseLabel:
    cx, cy, bbox_w, bbox_h = label.bbox
    # Six-decimal YOLO serialization can place an otherwise valid box a tiny
    # fraction of a pixel beyond an image edge. Clamp those source extents
    # before applying the crop transform.
    source_left = max(0.0, (cx - bbox_w / 2) * width)
    source_right = min(float(width), (cx + bbox_w / 2) * width)
    source_top = max(0.0, (cy - bbox_h / 2) * height)
    source_bottom = min(float(height), (cy + bbox_h / 2) * height)
    left = (source_left - box.left) / box.width
    right = (source_right - box.left) / box.width
    top = (source_top - box.top) / box.height
    bottom = (source_bottom - box.top) / box.height
    bbox = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    keypoints = tuple(
        ((x * width - box.left) / box.width, (y * height - box.top) / box.height, visibility)
        for x, y, visibility in label.keypoints
    )
    transformed = PoseLabel(label.class_id, bbox, keypoints)
    validate_transformed_label(transformed)
    return transformed


def validate_transformed_label(label: PoseLabel) -> None:
    cx, cy, bbox_w, bbox_h = label.bbox
    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError("Transformed bounding box is non-positive")
    if cx - bbox_w / 2 < -1e-6 or cx + bbox_w / 2 > 1 + 1e-6:
        raise ValueError("Transformed bounding box x extent outside crop")
    if cy - bbox_h / 2 < -1e-6 or cy + bbox_h / 2 > 1 + 1e-6:
        raise ValueError("Transformed bounding box y extent outside crop")
    for x, y, visibility in label.keypoints:
        if visibility > 0 and not (-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6):
            raise ValueError("Transformed keypoint outside crop")


def load_source_records(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    groups: dict[str, set[str]] = {}
    by_image: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        split = (row.get("split") or "").strip()
        group_id = (row.get("group_id") or "").strip()
        image = (row.get("image") or "").strip()
        if split not in SPLITS or not group_id or not image:
            raise ValueError(f"Invalid source manifest row: {row}")
        groups.setdefault(group_id, set()).add(split)
        if image in by_image:
            raise ValueError(f"Duplicate source manifest image: {image}")
        by_image[image] = row
    leaked = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Source patient groups cross splits: {leaked[:5]}")

    image_dir, label_dir = dataset_root / "images/train", dataset_root / "labels/train"
    images = sorted(image_dir.glob("*.png"))
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    if not images or {path.stem for path in images} != set(labels):
        raise ValueError("Source train image/label pairing is incomplete")
    records = []
    for image_path in images:
        relative_image = str(image_path.relative_to(dataset_root))
        row = by_image.get(relative_image)
        if row is None or row["split"] != "train":
            raise ValueError(f"Source train image missing from train manifest: {relative_image}")
        label_path = labels[image_path.stem]
        width, height = read_png_size(image_path)
        records.append({
            "source_image": image_path,
            "source_label": label_path,
            "source_image_relative": relative_image,
            "source_label_relative": str(label_path.relative_to(dataset_root)),
            "width": width,
            "height": height,
            "label": parse_pose_label(label_path),
            "batch": row["batch"],
            "patient_id": row["patient_id"],
            "group_id": row["group_id"],
        })
    return records


def plan_views(
    dataset_root: Path,
    seed: int = 20260907,
    margin: float = 2.5,
    shift_jitter: float = 0.05,
    scale_jitter: float = 0.10,
) -> list[dict[str, Any]]:
    records = load_source_records(dataset_root.resolve())
    for record in records:
        box = compute_crop_box(
            record["label"], record["width"], record["height"],
            record["source_image"].name, seed, margin, shift_jitter, scale_jitter,
        )
        record["crop_box"] = box
        record["transformed_label"] = transform_label(
            record["label"], box, record["width"], record["height"]
        )
        record["output_stem"] = f"roi__{record['source_image'].stem}"
        record["crop_area_fraction"] = box.width * box.height / (record["width"] * record["height"])
    return records


def ffmpeg_crop(source: Path, output: Path, box: CropBox, ffmpeg: str) -> None:
    command = [
        ffmpeg, "-v", "error", "-threads", "1", "-i", str(source),
        "-vf", f"crop={box.width}:{box.height}:{box.left}:{box.top}",
        "-frames:v", "1", "-compression_level", "4", "-y", str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = round((len(ordered) - 1) * probability)
    return ordered[position]


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    areas = [record["crop_area_fraction"] for record in records]
    return {
        "source_train_images": len(records),
        "roi_train_images": len(records),
        "mixed_train_views": len(records) * 2,
        "patients": len({record["group_id"] for record in records}),
        "images_by_batch": dict(sorted(Counter(record["batch"] for record in records).items())),
        "crop_area_fraction": {
            "min": min(areas),
            "p10": _percentile(areas, 0.10),
            "median": _percentile(areas, 0.50),
            "p90": _percentile(areas, 0.90),
            "max": max(areas),
        },
        "full_frame_crops": sum(value >= 0.999 for value in areas),
    }


def audit_leakage(
    dataset_root: Path, output_root: Path, *, write_report: bool = False
) -> dict[str, Any]:
    """Audit patient lineage and exact-image leakage across immutable splits."""
    dataset_root, output_root = dataset_root.resolve(), output_root.resolve()
    with (dataset_root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with (output_root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        roi_rows = list(csv.DictReader(handle))

    errors: list[str] = []
    source_by_image: dict[str, dict[str, str]] = {}
    source_groups: dict[str, set[str]] = {}
    source_hash_splits: dict[str, set[str]] = {}
    source_hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in source_rows:
        image_value = (row.get("image") or "").strip()
        split = (row.get("split") or "").strip()
        group_id = (row.get("group_id") or "").strip()
        if split not in SPLITS or not image_value or not group_id:
            errors.append(f"invalid source row: {image_value}")
            continue
        if image_value in source_by_image:
            errors.append(f"duplicate source image manifest key: {image_value}")
            continue
        image_path = dataset_root / image_value
        if not image_path.is_file():
            errors.append(f"missing source image: {image_value}")
            continue
        digest = sha256_file(image_path)
        source_by_image[image_value] = row
        source_groups.setdefault(group_id, set()).add(split)
        source_hash_splits.setdefault(digest, set()).add(split)
        source_hashes_by_split[split].add(digest)

    for group_id, splits in source_groups.items():
        if len(splits) > 1:
            errors.append(f"source patient crosses splits: {group_id}:{sorted(splits)}")
    cross_split_source_hashes = {
        digest: splits for digest, splits in source_hash_splits.items() if len(splits) > 1
    }
    if cross_split_source_hashes:
        errors.append(f"source exact images cross splits: {len(cross_split_source_hashes)}")

    roi_groups: set[str] = set()
    roi_output_hashes: set[str] = set()
    for row in roi_rows:
        source_image = (row.get("source_image") or "").strip()
        source_split = (row.get("source_split") or "").strip()
        group_id = (row.get("group_id") or "").strip()
        output_image = (row.get("output_image") or "").strip()
        output_label = (row.get("output_label") or "").strip()
        source = source_by_image.get(source_image)
        if source_split != "train":
            errors.append(f"ROI has non-train source_split: {output_image}:{source_split}")
        if source is None or source.get("split") != "train":
            errors.append(f"ROI source is not in source train: {source_image}")
        elif source.get("group_id") != group_id:
            errors.append(f"ROI/source group mismatch: {output_image}")
        source_path = dataset_root / source_image
        image_path, label_path = output_root / output_image, output_root / output_label
        if not source_path.is_file() or sha256_file(source_path) != row.get("source_image_sha256"):
            errors.append(f"ROI source hash mismatch: {source_image}")
        if not image_path.is_file() or sha256_file(image_path) != row.get("output_image_sha256"):
            errors.append(f"ROI output image hash mismatch: {output_image}")
        else:
            roi_output_hashes.add(row["output_image_sha256"])
        if not label_path.is_file() or sha256_file(label_path) != row.get("output_label_sha256"):
            errors.append(f"ROI output label hash mismatch: {output_label}")
        roi_groups.add(group_id)

    non_train_groups = sorted(
        group for group in roi_groups if source_groups.get(group, set()) != {"train"}
    )
    if non_train_groups:
        errors.append(f"ROI groups are not train-only: {non_train_groups[:5]}")
    exact_roi_vs_holdout = roi_output_hashes & (
        source_hashes_by_split["val"] | source_hashes_by_split["test"]
    )
    if exact_roi_vs_holdout:
        errors.append(f"ROI images exactly match holdout originals: {len(exact_roi_vs_holdout)}")
    for split in ("val", "test"):
        if (output_root / f"images/{split}").exists() or (output_root / f"labels/{split}").exists():
            errors.append(f"ROI output contains forbidden {split} directory")

    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "source_dataset": str(dataset_root),
        "roi_dataset": str(output_root),
        "counts": {
            "source_images": len(source_rows),
            "source_patient_groups": len(source_groups),
            "roi_images": len(roi_rows),
            "roi_patient_groups": len(roi_groups),
            "source_cross_split_patient_groups": sum(
                len(splits) > 1 for splits in source_groups.values()
            ),
            "source_cross_split_exact_image_groups": len(cross_split_source_hashes),
            "roi_non_train_patient_groups": len(non_train_groups),
            "roi_exact_matches_to_val_test": len(exact_roi_vs_holdout),
        },
        "errors": errors,
    }
    if write_report:
        (output_root / "leakage_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def build_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    seed: int = 20260907,
    margin: float = 2.5,
    shift_jitter: float = 0.05,
    scale_jitter: float = 0.10,
    workers: int = 4,
    ffmpeg: str = "ffmpeg",
    apply: bool = False,
    cropper: Callable[[Path, Path, CropBox, str], None] = ffmpeg_crop,
) -> dict[str, Any]:
    dataset_root, output_root = dataset_root.resolve(), output_root.resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    records = plan_views(dataset_root, seed, margin, shift_jitter, scale_jitter)
    summary = summarize(records)
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
        output_label.write_text(format_pose_label(record["transformed_label"]), encoding="utf-8")
        if read_png_size(output_image) != (box.width, box.height):
            raise ValueError(f"Saved crop size mismatch: {output_image}")
        parse_pose_label(output_label)
        return {
            "batch": record["batch"],
            "patient_id": record["patient_id"],
            "group_id": record["group_id"],
            "source_split": "train",
            "source_image": record["source_image_relative"],
            "source_label": record["source_label_relative"],
            "source_width": record["width"],
            "source_height": record["height"],
            "source_image_sha256": sha256_file(record["source_image"]),
            "crop_box": json.dumps(asdict(box), separators=(",", ":")),
            "crop_width": box.width,
            "crop_height": box.height,
            "crop_area_fraction": f"{record['crop_area_fraction']:.8f}",
            "output_image": str(output_image.relative_to(staging)),
            "output_label": str(output_label.relative_to(staging)),
            "output_image_sha256": sha256_file(output_image),
            "output_label_sha256": sha256_file(output_label),
        }

    try:
        manifest_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, row in enumerate(executor.map(materialize, records), 1):
                manifest_rows.append(row)
                if index % 50 == 0 or index == len(records):
                    print(f"Created {index}/{len(records)} ROI views", flush=True)
        fields = list(manifest_rows[0])
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest_rows)
        report = {
            "schema_version": 1,
            "source_dataset": str(dataset_root),
            "output": str(output_root),
            "configuration": {
                "seed": seed,
                "margin": margin,
                "shift_jitter": shift_jitter,
                "scale_jitter": scale_jitter,
                "safety_margin": 0.03,
                "roi_prefix": "roi__",
                "source_split": "train",
            },
            "summary": summary,
            "leakage_contract": {
                "derived_splits": ["train"],
                "validation_and_test_views": "original_only",
                "source_patient_split_immutable": True,
            },
        }
        (staging / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# Pelvis Three-Keypoint ROI Training Views\n\n"
            "This directory contains train-only derived ROI views. It does not contain val/test data.\n\n"
            f"- Source train images: {summary['source_train_images']}\n"
            f"- ROI train images: {summary['roi_train_images']}\n"
            f"- Mixed train views: {summary['mixed_train_views']}\n"
            "- Validation/test: original images only\n"
            "- Keypoints and bounding boxes are transformed into crop coordinates.\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=project_root / "datasets/yolo_pelvis_3kpt_all",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=project_root / "datasets/yolo_pelvis_3kpt_roi_views",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--margin", type=float, default=2.5)
    parser.add_argument("--shift-jitter", type=float, default=0.05)
    parser.add_argument("--scale-jitter", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--apply", action="store_true", help="write the derived dataset")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="audit an existing ROI dataset and write leakage_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        result = audit_leakage(args.dataset_root, args.output_root, write_report=True)
    else:
        result = build_dataset(
            args.dataset_root, args.output_root,
            seed=args.seed, margin=args.margin, shift_jitter=args.shift_jitter,
            scale_jitter=args.scale_jitter, workers=args.workers,
            ffmpeg=args.ffmpeg, apply=args.apply,
        )
        if args.apply:
            result["leakage_audit"] = audit_leakage(
                args.dataset_root, args.output_root, write_report=True
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
