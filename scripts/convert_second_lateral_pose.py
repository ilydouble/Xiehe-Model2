#!/usr/bin/env python3
"""Convert the 2026-09-03 lateral LabelMe export to YOLO Pose.

The converter is intentionally dependency-free. It pairs PNG/JSON files by
their exact stem, keeps the source export read-only, orders every vertebral
quadrilateral as upper-left, upper-right, lower-right, lower-left, and splits
patients rather than individual images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


CLASS_NAMES = [
    *(f"C{i}" for i in range(2, 8)),
    *(f"T{i}" for i in range(1, 13)),
    *(f"L{i}" for i in range(1, 6)),
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
KEYPOINT_ORDER = ["upper_left", "upper_right", "lower_right", "lower_left"]
FLIP_IDX = [1, 0, 3, 2]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_png_size(path: Path) -> tuple[int, int]:
    """Read PNG width and height without Pillow."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def patient_id_from_stem(stem: str) -> str:
    """Return the patient identifier shared by repeated acquisitions."""
    patient_id = stem.split("_", 1)[0].strip()
    if not patient_id:
        raise ValueError(f"Cannot derive patient id from stem: {stem!r}")
    return patient_id


def source_prefix(patient_id: str) -> str:
    match = re.match(r"[A-Za-z]+", patient_id)
    return match.group(0).upper() if match else "OTHER"


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


def order_quadrilateral(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Order four image-space vertices as UL, UR, LR, LL.

    Sorting by polar angle yields clockwise order in image coordinates. The
    cycle is then rotated to begin at the upper-left vertex (minimum x + y).
    """
    if len(points) != 4:
        raise ValueError(f"Expected exactly four points, got {len(points)}")
    converted = [(float(point[0]), float(point[1])) for point in points]
    if len(set(converted)) != 4 or polygon_area(converted) <= 0:
        raise ValueError("Degenerate quadrilateral")
    center_x = sum(point[0] for point in converted) / 4.0
    center_y = sum(point[1] for point in converted) / 4.0
    clockwise = sorted(
        converted,
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
    )
    start = min(range(4), key=lambda index: sum(clockwise[index]))
    ordered = clockwise[start:] + clockwise[:start]
    if ordered[1][0] < ordered[-1][0]:
        ordered = [ordered[0], *reversed(ordered[1:])]
    return ordered


def clip_point(point: tuple[float, float], width: int, height: int) -> tuple[tuple[float, float], bool]:
    x, y = point
    clipped = (min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height)))
    return clipped, clipped != point


def yolo_pose_line(
    class_id: int,
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[str, int]:
    """Convert one four-point polygon to one YOLO Pose label line."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    ordered = order_quadrilateral(points)
    clipped_count = 0
    clipped_points: list[tuple[float, float]] = []
    for point in ordered:
        clipped, changed = clip_point(point, width, height)
        clipped_points.append(clipped)
        clipped_count += int(changed)
    if polygon_area(clipped_points) <= 0:
        raise ValueError("Quadrilateral became degenerate after clipping")

    xs = [point[0] for point in clipped_points]
    ys = [point[1] for point in clipped_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box = [
        (x_min + x_max) / 2.0 / width,
        (y_min + y_max) / 2.0 / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    ]
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError("Zero-area bounding box")
    values = [str(class_id), *(f"{value:.6f}" for value in box)]
    for x, y in clipped_points:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}", "2"))
    return " ".join(values), clipped_count


def load_review_actions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    actions: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stem = (row.get("stem") or "").strip()
            action = (row.get("action") or "").strip().lower()
            reason = (row.get("reason") or "").strip()
            if not stem or action not in {"exclude", "include", "warn"}:
                raise ValueError(f"Invalid review row: {row}")
            if stem in actions:
                raise ValueError(f"Duplicate review stem: {stem}")
            actions[stem] = {"action": action, "reason": reason}
    return actions


def discover_pairs(source: Path) -> list[tuple[Path, Path]]:
    images: dict[tuple[Path, str], Path] = {}
    annotations: dict[tuple[Path, str], Path] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        key = (path.parent, path.stem)
        if path.suffix.lower() == ".png":
            if key in images:
                raise ValueError(f"Duplicate image stem: {path}")
            images[key] = path
        elif path.suffix.lower() == ".json":
            if key in annotations:
                raise ValueError(f"Duplicate JSON stem: {path}")
            annotations[key] = path
    paired_keys = sorted(set(images) & set(annotations), key=lambda key: str(key[1]))
    if not paired_keys:
        raise ValueError(f"No paired PNG/JSON samples found under {source}")
    return [(images[key], annotations[key]) for key in paired_keys]


def stable_group_split(
    patient_ids: Iterable[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique_ids = sorted(set(patient_ids))
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than one")
    ordered = sorted(
        unique_ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    test_count = round(len(ordered) * test_ratio)
    val_count = round(len(ordered) * val_ratio)
    if test_count + val_count >= len(ordered):
        raise ValueError("Not enough patient groups for requested split ratios")
    result: dict[str, str] = {}
    for patient_id in ordered[:test_count]:
        result[patient_id] = "test"
    for patient_id in ordered[test_count:test_count + val_count]:
        result[patient_id] = "val"
    for patient_id in ordered[test_count + val_count:]:
        result[patient_id] = "train"
    return result


def parse_target_shapes(annotation_path: Path) -> tuple[dict[str, list[list[float]]], int, int]:
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    targets: dict[str, list[list[float]]] = {}
    for shape in data.get("shapes", []):
        label = str(shape.get("label") or "").strip().upper()
        if label not in CLASS_TO_ID:
            continue
        if label in targets:
            raise ValueError(f"Duplicate target label {label}")
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"Target {label} is not a polygon")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError(f"Target {label} does not have exactly four points")
        targets[label] = points
    return targets, width, height


def write_data_yaml(path: Path) -> None:
    lines = [
        "# YOLO11 Pose: 2026-09-03 lateral spine dataset",
        "# The dataset root is inferred from this YAML file for portability.",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)),
        "",
        "kpt_shape: [4, 3]",
        "flip_idx: [1, 0, 3, 2]  # UL<->UR, LR<->LL",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_readme(path: Path, report: dict[str, Any]) -> None:
    counts = report["counts"]
    path.write_text(
        "# 2026-09-03 Lateral Spine YOLO Pose Dataset\n\n"
        f"- Images: {counts['included_images']}\n"
        f"- Patient groups: {counts['included_patients']}\n"
        f"- Classes: {len(CLASS_NAMES)} (`C2-C7`, `T1-T12`, `L1-L5`)\n"
        "- Keypoints: upper-left, upper-right, lower-right, lower-left\n"
        f"- Split seed: {report['split']['seed']}\n"
        "- T13, S1 and femoral-head annotations are not part of this model.\n\n"
        "See `conversion_report.json`, `manifest.csv`, and `excluded_samples.csv` "
        "for provenance and validation details.\n",
        encoding="utf-8",
    )


def convert_dataset(
    source: Path,
    output: Path,
    review_csv: Path | None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    pairs = discover_pairs(source)
    review_actions = load_review_actions(review_csv)
    discovered_stems = {image.stem for image, _ in pairs}
    unknown_reviews = sorted(set(review_actions) - discovered_stems)
    if unknown_reviews:
        raise ValueError(f"Review CSV references unknown stems: {unknown_reviews}")

    included_pairs = [
        (image, annotation)
        for image, annotation in pairs
        if review_actions.get(image.stem, {}).get("action") != "exclude"
    ]
    patient_splits = stable_group_split(
        (patient_id_from_stem(image.stem) for image, _ in included_pairs),
        val_ratio,
        test_ratio,
        seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    split_image_counts: Counter[str] = Counter()
    split_object_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    missing_class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    clipped_points = 0
    manifest_rows: list[dict[str, str | int]] = []
    excluded_rows: list[dict[str, str]] = []
    try:
        for split in ("train", "val", "test"):
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)

        for image_path, annotation_path in pairs:
            review = review_actions.get(image_path.stem, {})
            if review.get("action") == "exclude":
                excluded_rows.append({
                    "stem": image_path.stem,
                    "action": "exclude",
                    "reason": review.get("reason", ""),
                })
                continue

            patient_id = patient_id_from_stem(image_path.stem)
            split = patient_splits[patient_id]
            actual_width, actual_height = read_png_size(image_path)
            targets, json_width, json_height = parse_target_shapes(annotation_path)
            if (json_width, json_height) != (actual_width, actual_height):
                raise ValueError(
                    f"Image/JSON size mismatch for {image_path.stem}: "
                    f"PNG={actual_width}x{actual_height}, JSON={json_width}x{json_height}"
                )
            if not targets:
                raise ValueError(f"No target vertebrae in {annotation_path}")

            label_lines: list[str] = []
            for label in CLASS_NAMES:
                if label not in targets:
                    missing_class_counts[label] += 1
                    continue
                try:
                    line, clipped = yolo_pose_line(
                        CLASS_TO_ID[label], targets[label], actual_width, actual_height
                    )
                except ValueError as exc:
                    raise ValueError(f"{annotation_path.name} {label}: {exc}") from exc
                label_lines.append(line)
                clipped_points += clipped
                class_counts[label] += 1

            destination_image = temporary / "images" / split / image_path.name
            destination_label = temporary / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, destination_image)
            destination_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            split_image_counts[split] += 1
            split_object_counts[split] += len(label_lines)
            source_counts[source_prefix(patient_id)] += 1
            manifest_rows.append({
                "stem": image_path.stem,
                "patient_id": patient_id,
                "source": source_prefix(patient_id),
                "split": split,
                "image": f"images/{split}/{image_path.name}",
                "label": f"labels/{split}/{image_path.stem}.txt",
                "objects": len(label_lines),
                "missing_classes": ";".join(label for label in CLASS_NAMES if label not in targets),
                "review_action": review.get("action", ""),
                "review_reason": review.get("reason", ""),
            })

        write_data_yaml(temporary / "data.yaml")
        with (temporary / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(sorted(manifest_rows, key=lambda row: str(row["stem"])))
        with (temporary / "excluded_samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stem", "action", "reason"])
            writer.writeheader()
            writer.writerows(sorted(excluded_rows, key=lambda row: row["stem"]))

        split_patients = defaultdict(set)
        for row in manifest_rows:
            split_patients[str(row["split"])].add(str(row["patient_id"]))
        if any(split_patients[a] & split_patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise AssertionError("Patient leakage detected between splits")

        report: dict[str, Any] = {
            "schema_version": 1,
            "source": str(source),
            "output": str(output),
            "classes": CLASS_NAMES,
            "keypoint_order": KEYPOINT_ORDER,
            "flip_idx": FLIP_IDX,
            "split": {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio},
            "counts": {
                "discovered_pairs": len(pairs),
                "included_images": len(manifest_rows),
                "included_patients": len({str(row["patient_id"]) for row in manifest_rows}),
                "excluded_images": len(excluded_rows),
                "clipped_points": clipped_points,
                "images_by_split": dict(sorted(split_image_counts.items())),
                "patients_by_split": {key: len(value) for key, value in sorted(split_patients.items())},
                "objects_by_split": dict(sorted(split_object_counts.items())),
                "objects_by_class": {name: class_counts[name] for name in CLASS_NAMES},
                "missing_by_class": {name: missing_class_counts[name] for name in CLASS_NAMES},
                "images_by_source": dict(sorted(source_counts.items())),
            },
            "excluded_samples": excluded_rows,
            "notes": [
                "T13 is absent from this source batch and is not included in this 23-class dataset.",
                "S1 line and CFH/FH point annotations are intentionally excluded.",
                "Original PNG bytes are copied unchanged; Ultralytics decodes them as three-channel images.",
            ],
        }
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_dataset_readme(temporary / "README.md", report)
        temporary.rename(output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Volumes/E/spine_data/20260903-侧面数据第二批/labelme-export/P202607076308"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "datasets/yolo_lateral_20260903_23cls",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=project_root / "2-build_dataset/second_lateral_review.csv",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        source=args.source,
        output=args.output,
        review_csv=args.review_csv,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    counts = report["counts"]
    print(f"Created: {report['output']}")
    print(f"Images: {counts['included_images']} ({counts['images_by_split']})")
    print(f"Patients: {counts['included_patients']} ({counts['patients_by_split']})")
    print(f"Excluded: {counts['excluded_images']}; clipped points: {counts['clipped_points']}")


if __name__ == "__main__":
    main()
