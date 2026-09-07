#!/usr/bin/env python3
"""Build a combined three-keypoint pelvis dataset in YOLO Pose format.

Each image contains one ``pelvis`` instance with keypoints ordered as CFH,
the image-left S1 endpoint, and the image-right S1 endpoint.  The old batch
supplies CFH directly.  The new batch may supply CFH directly or FH-1/FH-2,
whose midpoint is converted to CFH.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KEYPOINT_ORDER = ["CFH", "S1_left", "S1_right"]
FLIP_IDX = [0, 2, 1]
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Candidate:
    batch: str
    image_path: Path
    annotation_path: Path
    patient_id: str
    group_id: str
    width: int
    height: int
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    cfh_source: str


def read_png_size(path: Path) -> tuple[int, int]:
    """Read PNG width and height without an image-library dependency."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def discover_assets(source: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    images: dict[tuple[Path, str], Path] = {}
    annotations: dict[tuple[Path, str], Path] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        key = (path.parent, path.stem)
        if path.suffix.lower() == ".png":
            if key in images:
                raise ValueError(f"Duplicate image stem in one directory: {path}")
            images[key] = path
        elif path.suffix.lower() == ".json":
            if key in annotations:
                raise ValueError(f"Duplicate JSON stem in one directory: {path}")
            annotations[key] = path
    shared = sorted(set(images) & set(annotations), key=lambda key: str(images[key]))
    image_only = [images[key] for key in sorted(set(images) - set(annotations), key=str)]
    json_only = [annotations[key] for key in sorted(set(annotations) - set(images), key=str)]
    return [(images[key], annotations[key]) for key in shared], image_only, json_only


def stable_group_split(
    group_ids: Iterable[str], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, str]:
    unique_ids = sorted(set(group_ids))
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
    result = {group_id: "train" for group_id in ordered}
    result.update({group_id: "test" for group_id in ordered[:test_count]})
    result.update({group_id: "val" for group_id in ordered[test_count:test_count + val_count]})
    return result


def _one_point(shape: dict[str, Any], label: str) -> tuple[float, float]:
    points = shape.get("points")
    if shape.get("shape_type") != "point" or not isinstance(points, list) or len(points) != 1:
        raise ValueError(f"{label}_not_single_point")
    point = points[0]
    if not isinstance(point, list) or len(point) < 2:
        raise ValueError(f"{label}_invalid_point")
    return float(point[0]), float(point[1])


def extract_pelvis_annotation(
    annotation_path: Path, batch: str
) -> tuple[int, int, tuple[tuple[float, float], ...], str]:
    """Extract CFH and ordered S1 endpoints from one LabelMe annotation."""
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("invalid_json_image_size")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shape in data.get("shapes", []):
        label = str(shape.get("label") or "").strip().upper().replace("_", "-")
        by_label[label].append(shape)

    s1_shapes = by_label.get("S1", [])
    valid_s1_lines = [
        shape for shape in s1_shapes
        if shape.get("shape_type") == "line"
        and isinstance(shape.get("points"), list)
        and len(shape["points"]) == 2
    ]
    if len(valid_s1_lines) != 1:
        if not s1_shapes:
            raise ValueError("missing_S1")
        raise ValueError("missing_or_duplicate_valid_S1_line")
    # Some old annotations also contain an S1 circle.  It is an auxiliary
    # shape, not an endpoint definition, so the unique valid line wins.
    s1 = valid_s1_lines[0]
    s1_points = s1["points"]
    endpoints = [(float(point[0]), float(point[1])) for point in s1_points]
    if endpoints[0] == endpoints[1]:
        raise ValueError("degenerate_S1_line")
    left, right = sorted(endpoints, key=lambda point: (point[0], point[1]))

    cfh_shapes = by_label.get("CFH", [])
    if len(cfh_shapes) == 1:
        cfh = _one_point(cfh_shapes[0], "CFH")
        cfh_source = "direct"
    elif len(cfh_shapes) > 1:
        raise ValueError("duplicate_CFH")
    elif batch == "new":
        fh1_shapes = by_label.get("FH-1", [])
        fh2_shapes = by_label.get("FH-2", [])
        if len(fh1_shapes) != 1 or len(fh2_shapes) != 1:
            raise ValueError("missing_CFH_or_FH_pair")
        fh1 = _one_point(fh1_shapes[0], "FH-1")
        fh2 = _one_point(fh2_shapes[0], "FH-2")
        cfh = ((fh1[0] + fh2[0]) / 2.0, (fh1[1] + fh2[1]) / 2.0)
        cfh_source = "FH_midpoint"
    else:
        raise ValueError("missing_CFH")
    return width, height, (cfh, left, right), cfh_source


def clip_point(
    point: tuple[float, float], width: int, height: int
) -> tuple[tuple[float, float], bool]:
    clipped = (
        min(max(point[0], 0.0), float(width)),
        min(max(point[1], 0.0), float(height)),
    )
    return clipped, clipped != point


def yolo_pose_line(
    points: Sequence[tuple[float, float]], width: int, height: int, padding: float = 0.20
) -> tuple[str, int]:
    """Serialize one three-keypoint pelvis instance as a YOLO Pose label."""
    if len(points) != 3 or width <= 0 or height <= 0:
        raise ValueError("Expected three keypoints and a positive image size")
    clipped_points: list[tuple[float, float]] = []
    clipped_count = 0
    for point in points:
        clipped, changed = clip_point(point, width, height)
        clipped_points.append(clipped)
        clipped_count += int(changed)

    xs = [point[0] for point in clipped_points]
    ys = [point[1] for point in clipped_points]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    pad_x = max(span_x * padding, width * 0.02)
    pad_y = max(span_y * padding, height * 0.02)
    x_min, x_max = max(0.0, min(xs) - pad_x), min(float(width), max(xs) + pad_x)
    y_min, y_max = max(0.0, min(ys) - pad_y), min(float(height), max(ys) + pad_y)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Zero-area pelvis bounding box")
    box = (
        (x_min + x_max) / 2.0 / width,
        (y_min + y_max) / 2.0 / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    )
    values = ["0", *(f"{value:.6f}" for value in box)]
    for x, y in clipped_points:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}", "2"))
    return " ".join(values), clipped_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_candidates(
    candidates: Sequence[Candidate],
) -> tuple[list[Candidate], list[tuple[Candidate, Candidate]]]:
    """Remove byte-identical PNGs, hashing only equal-size candidate groups."""
    by_size: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_size[candidate.image_path.stat().st_size].append(candidate)
    kept: list[Candidate] = []
    duplicates: list[tuple[Candidate, Candidate]] = []
    for size in sorted(by_size):
        size_group = sorted(
            by_size[size], key=lambda item: (item.batch != "old", str(item.image_path))
        )
        if len(size_group) == 1:
            kept.extend(size_group)
            continue
        by_digest: dict[str, Candidate] = {}
        for candidate in size_group:
            digest = _sha256(candidate.image_path)
            if digest in by_digest:
                duplicates.append((candidate, by_digest[digest]))
            else:
                by_digest[digest] = candidate
                kept.append(candidate)
    return sorted(kept, key=lambda item: (item.batch, str(item.image_path))), duplicates


def _safe_stem(value: str) -> str:
    result = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
    if not result:
        raise ValueError(f"Cannot create destination name from {value!r}")
    return result


def write_data_yaml(path: Path) -> None:
    path.write_text(
        "# Combined old/new lateral pelvis landmarks in YOLO Pose format\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: pelvis\n\n"
        "kpt_shape: [3, 3]\n"
        "flip_idx: [0, 2, 1]  # CFH fixed; swap image-left/right S1 endpoints\n",
        encoding="utf-8",
    )


def _excluded_row(batch: str, path: Path, reason: str, detail: str = "") -> dict[str, str]:
    return {
        "batch": batch,
        "stem": path.stem,
        "source_path": str(path),
        "reason": reason,
        "detail": detail,
    }


def convert_dataset(
    old_source: Path,
    new_source: Path,
    output: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    old_source, new_source, output = old_source.resolve(), new_source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    sources = {"old": old_source, "new": new_source}
    discovery: dict[str, dict[str, int]] = {}
    excluded_rows: list[dict[str, str]] = []
    candidates: list[Candidate] = []
    for batch, source in sources.items():
        pairs, image_only, json_only = discover_assets(source)
        discovery[batch] = {
            "paired": len(pairs), "image_without_json": len(image_only), "json_without_image": len(json_only)
        }
        excluded_rows.extend(_excluded_row(batch, path, "image_without_json") for path in image_only)
        excluded_rows.extend(_excluded_row(batch, path, "json_without_image") for path in json_only)
        for image_path, annotation_path in pairs:
            try:
                width, height, points, cfh_source = extract_pelvis_annotation(annotation_path, batch)
                actual_width, actual_height = read_png_size(image_path)
                if (width, height) != (actual_width, actual_height):
                    raise ValueError(
                        f"image_json_size_mismatch:{actual_width}x{actual_height}!={width}x{height}"
                    )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                reason, _, detail = str(exc).partition(":")
                excluded_rows.append(_excluded_row(batch, image_path, reason, detail))
                continue
            patient_id = image_path.parent.name if batch == "old" else image_path.stem.split("_", 1)[0]
            patient_id = patient_id.strip()
            if not patient_id:
                excluded_rows.append(_excluded_row(batch, image_path, "invalid_patient_id"))
                continue
            candidates.append(Candidate(
                batch=batch,
                image_path=image_path,
                annotation_path=annotation_path,
                patient_id=patient_id,
                group_id=f"{batch}:{patient_id}",
                width=width,
                height=height,
                points=points,  # type: ignore[arg-type]
                cfh_source=cfh_source,
            ))

    unique_candidates, duplicates = deduplicate_candidates(candidates)
    for duplicate, original in duplicates:
        excluded_rows.append(_excluded_row(
            duplicate.batch, duplicate.image_path, "exact_duplicate_image", str(original.image_path)
        ))

    patient_splits: dict[str, str] = {}
    for batch in sources:
        patient_splits.update(stable_group_split(
            (item.group_id for item in unique_candidates if item.batch == batch),
            val_ratio, test_ratio, seed,
        ))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    manifest_rows: list[dict[str, str | int]] = []
    split_images: Counter[str] = Counter()
    split_patients: dict[str, set[str]] = defaultdict(set)
    batch_images: Counter[str] = Counter()
    batch_split_images: Counter[tuple[str, str]] = Counter()
    cfh_sources: Counter[str] = Counter()
    clipped_points = 0
    used_names: set[str] = set()
    try:
        for split in SPLITS:
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)

        for item in unique_candidates:
            split = patient_splits[item.group_id]
            base_name = _safe_stem(f"{item.batch}__{item.patient_id}__{item.image_path.stem}")
            destination_stem = base_name
            if destination_stem in used_names:
                suffix = hashlib.sha256(str(item.image_path).encode()).hexdigest()[:10]
                destination_stem = f"{base_name}__{suffix}"
            used_names.add(destination_stem)

            label_line, clipped = yolo_pose_line(item.points, item.width, item.height)
            destination_image = temporary / "images" / split / f"{destination_stem}.png"
            destination_label = temporary / "labels" / split / f"{destination_stem}.txt"
            shutil.copy2(item.image_path, destination_image)
            destination_label.write_text(label_line + "\n", encoding="utf-8")

            clipped_points += clipped
            split_images[split] += 1
            split_patients[split].add(item.group_id)
            batch_images[item.batch] += 1
            batch_split_images[(item.batch, split)] += 1
            cfh_sources[item.cfh_source] += 1
            manifest_rows.append({
                "batch": item.batch,
                "patient_id": item.patient_id,
                "group_id": item.group_id,
                "source_stem": item.image_path.stem,
                "source_image": str(item.image_path),
                "source_annotation": str(item.annotation_path),
                "cfh_source": item.cfh_source,
                "split": split,
                "image": f"images/{split}/{destination_stem}.png",
                "label": f"labels/{split}/{destination_stem}.txt",
            })

        if any(
            split_patients[a] & split_patients[b]
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise AssertionError("Patient leakage detected between splits")

        write_data_yaml(temporary / "data.yaml")
        manifest_fields = [
            "batch", "patient_id", "group_id", "source_stem", "source_image",
            "source_annotation", "cfh_source", "split", "image", "label",
        ]
        with (temporary / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows(sorted(manifest_rows, key=lambda row: str(row["image"])))
        excluded_fields = ["batch", "stem", "source_path", "reason", "detail"]
        with (temporary / "excluded_samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=excluded_fields)
            writer.writeheader()
            writer.writerows(sorted(excluded_rows, key=lambda row: (row["batch"], row["source_path"])))

        report: dict[str, Any] = {
            "schema_version": 1,
            "sources": {key: str(value) for key, value in sources.items()},
            "output": str(output),
            "class_names": ["pelvis"],
            "keypoint_order": KEYPOINT_ORDER,
            "flip_idx": FLIP_IDX,
            "bbox_rule": "three-keypoint extent padded 20% per axis, with 2% image-size minimum",
            "split": {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio, "per_batch": True},
            "discovery": discovery,
            "counts": {
                "eligible_before_deduplication": len(candidates),
                "included_images": len(manifest_rows),
                "included_patients": len({str(row["group_id"]) for row in manifest_rows}),
                "excluded_records": len(excluded_rows),
                "exact_duplicate_images": len(duplicates),
                "clipped_points": clipped_points,
                "images_by_batch": dict(sorted(batch_images.items())),
                "images_by_split": {split: split_images[split] for split in SPLITS},
                "patients_by_split": {split: len(split_patients[split]) for split in SPLITS},
                "images_by_batch_and_split": {
                    batch: {split: batch_split_images[(batch, split)] for split in SPLITS}
                    for batch in sources
                },
                "cfh_by_source": dict(sorted(cfh_sources.items())),
                "excluded_by_reason": dict(sorted(Counter(row["reason"] for row in excluded_rows).items())),
            },
            "notes": [
                "Each image has one pelvis object with three visible keypoints.",
                "Old S1 polygons/circles are excluded; only two-point S1 lines are accepted.",
                "Original PNG bytes are copied unchanged.",
                "Patient splits are deterministic and performed separately within each source batch.",
            ],
        }
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            "# Combined Pelvis Three-Keypoint YOLO Pose Dataset\n\n"
            f"- Images: {len(manifest_rows)}\n"
            f"- Patient groups: {report['counts']['included_patients']}\n"
            "- Object class: `pelvis`\n"
            "- Keypoints: `CFH`, `S1_left`, `S1_right` (image coordinates)\n"
            "- New-batch `FH-1`/`FH-2` annotations are represented by their midpoint.\n"
            "- Every label contains one object and three visible keypoints.\n\n"
            "See `conversion_report.json`, `manifest.csv`, and `excluded_samples.csv` for provenance.\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", type=Path, default=Path("/Volumes/E/spine_data/LAT202511"))
    parser.add_argument(
        "--new-source", type=Path,
        default=Path("/Volumes/E/spine_data/20260903-侧面数据第二批/labelme-export/P202607076308"),
    )
    parser.add_argument(
        "--output", type=Path, default=project_root / "datasets/yolo_pelvis_3kpt_all"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        args.old_source, args.new_source, args.output,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    counts = report["counts"]
    print(f"Created: {report['output']}")
    print(f"Images: {counts['included_images']} ({counts['images_by_split']})")
    print(f"Patients: {counts['included_patients']} ({counts['patients_by_split']})")
    print(f"Batches: {counts['images_by_batch']}; CFH sources: {counts['cfh_by_source']}")
    print(f"Excluded records: {counts['excluded_records']}; exact duplicates: {counts['exact_duplicate_images']}")


if __name__ == "__main__":
    main()
