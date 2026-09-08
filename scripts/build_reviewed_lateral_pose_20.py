#!/usr/bin/env python3
"""Build the reviewed 20-class lateral-spine YOLO Pose dataset.

The first batch contributes only human-accepted, source-clean images. Its C2
comes from the accepted pseudo-label review, while C7/T1-T13/L1-L5 are rebuilt
from the exact same-stem LabelMe JSON. Dense polygons are reduced to four
unique convex corners. The already-cleaned second batch is remapped from 23 to
20 classes, dropping C3-C6. All patients are then split together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CLASS_NAMES = ["C2", "C7", *(f"T{i}" for i in range(1, 14)), *(f"L{i}" for i in range(1, 6))]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
KEYPOINT_ORDER = ["upper_left", "upper_right", "lower_right", "lower_left"]
FLIP_IDX = [1, 0, 3, 2]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FIRST_SOURCE_STATUS = "correct_same_stem"
SECOND_ID_MAP = {0: 0, 5: 1, **{old: old - 4 for old in range(6, 18)}, **{old: old - 3 for old in range(18, 23)}}


@dataclass(frozen=True)
class Sample:
    source_batch: str
    source_stem: str
    patient_id: str
    image_path: Path
    label_lines: tuple[str, ...]
    class_names: tuple[str, ...]
    review_note: str = ""
    corner_fallbacks: int = 0


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def polygon_signed_area(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    ) / 2.0


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return abs(polygon_signed_area(points))


def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def convex_hull(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        raise ValueError("Polygon has fewer than three unique points")
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3 or polygon_area(hull) <= 0:
        raise ValueError("Polygon convex hull is degenerate")
    return hull


def minimum_area_rectangle(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Return a four-corner oriented rectangle; used only for triangular hulls."""
    hull = convex_hull(points)
    best: tuple[float, list[tuple[float, float]]] | None = None
    for first, second in zip(hull, hull[1:] + hull[:1]):
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = [(x * cosine + y * sine, -x * sine + y * cosine) for x, y in hull]
        xs = [point[0] for point in rotated]
        ys = [point[1] for point in rotated]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        area = (x_max - x_min) * (y_max - y_min)
        corners_rotated = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        corners = [
            (x * cosine - y * sine, x * sine + y * cosine)
            for x, y in corners_rotated
        ]
        if best is None or area < best[0]:
            best = (area, corners)
    if best is None or best[0] <= 0:
        raise ValueError("Cannot fit a non-degenerate rectangle")
    return best[1]


def order_quadrilateral(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    if len(points) != 4:
        raise ValueError(f"Expected four points, got {len(points)}")
    converted = [(float(point[0]), float(point[1])) for point in points]
    if len(set(converted)) != 4 or polygon_area(converted) <= 0:
        raise ValueError("Degenerate quadrilateral")
    center_x = sum(point[0] for point in converted) / 4.0
    center_y = sum(point[1] for point in converted) / 4.0
    cycle = sorted(converted, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    start = min(range(4), key=lambda index: cycle[index][0] + cycle[index][1])
    ordered = cycle[start:] + cycle[:start]
    if ordered[1][0] < ordered[-1][0]:
        ordered = [ordered[0], *reversed(ordered[1:])]
    signs = [cross(ordered[i], ordered[(i + 1) % 4], ordered[(i + 2) % 4]) for i in range(4)]
    if any(abs(value) <= 1e-9 for value in signs) or not (all(value > 0 for value in signs) or all(value < 0 for value in signs)):
        raise ValueError("Four points do not form a strictly convex quadrilateral")
    return ordered


def dense_polygon_to_corners(points: Sequence[Sequence[float]]) -> tuple[list[tuple[float, float]], bool]:
    """Reduce a dense contour to four real hull vertices without triangles.

    Convex-hull vertices are removed in ascending local triangle-area loss
    until four remain. A true three-vertex hull cannot supply four unique
    contour corners, so it uses a minimum-area rectangle fallback.
    """
    hull = convex_hull(points)
    used_fallback = len(hull) == 3
    if used_fallback:
        return order_quadrilateral(minimum_area_rectangle(hull)), True
    reduced = list(hull)
    while len(reduced) > 4:
        loss_index = min(
            range(len(reduced)),
            key=lambda index: (
                abs(cross(reduced[index - 1], reduced[index], reduced[(index + 1) % len(reduced)])),
                reduced[index][1],
                reduced[index][0],
            ),
        )
        reduced.pop(loss_index)
    return order_quadrilateral(reduced), False


def validate_corners(points: Sequence[tuple[float, float]]) -> None:
    ordered = order_quadrilateral(points)
    lengths = [
        math.dist(ordered[index], ordered[(index + 1) % 4])
        for index in range(4)
    ]
    if min(lengths) <= 1e-6 or min(lengths) / max(lengths) < 0.02:
        raise ValueError(f"Near-triangular quadrilateral: edge ratio={min(lengths) / max(lengths):.6f}")


def make_pose_line(
    class_id: int,
    corners: Sequence[Sequence[float]],
    bbox_points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[str, int]:
    ordered = order_quadrilateral(corners)
    clipped_count = 0
    clipped: list[tuple[float, float]] = []
    for x, y in ordered:
        point = (min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height)))
        clipped_count += int(point != (x, y))
        clipped.append(point)
    validate_corners(clipped)
    xs = [min(max(float(point[0]), 0.0), float(width)) for point in bbox_points]
    ys = [min(max(float(point[1]), 0.0), float(height)) for point in bbox_points]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Zero-area bounding box")
    values = [
        str(class_id),
        f"{(x_min + x_max) / 2 / width:.6f}",
        f"{(y_min + y_max) / 2 / height:.6f}",
        f"{(x_max - x_min) / width:.6f}",
        f"{(y_max - y_min) / height:.6f}",
    ]
    for x, y in clipped:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}", "2"))
    return " ".join(values), clipped_count


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def unique_target_shapes(data: dict[str, Any], labels: set[str]) -> dict[str, list[list[float]]]:
    found: dict[str, list[list[float]]] = {}
    for shape in data.get("shapes", []):
        label = str(shape.get("label") or "").strip().upper()
        if label not in labels:
            continue
        if shape.get("shape_type") != "polygon" or len(shape.get("points") or []) < 3:
            raise ValueError(f"Target {label} is not a valid polygon")
        if label in found:
            raise ValueError(f"Duplicate target polygon: {label}")
        found[label] = shape["points"]
    return found


def load_first_batch(
    review_csv: Path,
    source_audit_csv: Path,
    pseudo_manifest_csv: Path,
    pseudo_root: Path,
    first_images_root: Path,
    raw_root: Path,
) -> tuple[list[Sample], list[dict[str, str]], Counter[str]]:
    reviews = {row["relative_image"]: row for row in read_csv(review_csv)}
    audits = {row["relative_image"]: row for row in read_csv(source_audit_csv)}
    if len(reviews) != 368 or set(reviews) != set(audits):
        raise ValueError(f"First-batch review/audit coverage mismatch: reviews={len(reviews)}, audits={len(audits)}")
    pseudo_by_name: dict[str, dict[str, str]] = {}
    for row in read_csv(pseudo_manifest_csv):
        filename = Path(row["relative_image"]).name
        if filename in pseudo_by_name:
            raise ValueError(f"Duplicate pseudo manifest filename: {filename}")
        pseudo_by_name[filename] = row

    samples: list[Sample] = []
    excluded: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    raw_labels = set(CLASS_NAMES) - {"C2"}
    for relative_image, audit in sorted(audits.items()):
        review = reviews[relative_image]
        verdict = review.get("verdict", "").strip().lower()
        if verdict != "accepted":
            excluded.append({"source_batch": "first", "source_stem": Path(relative_image).stem, "reason": f"human_{verdict or 'unjudged'}"})
            continue
        if audit["status"] != FIRST_SOURCE_STATUS:
            excluded.append({"source_batch": "first", "source_stem": Path(relative_image).stem, "reason": f"source_audit_{audit['status']}"})
            continue
        filename = Path(relative_image).name
        pseudo_row = pseudo_by_name.get(filename)
        if pseudo_row is None:
            raise ValueError(f"No pseudo-label manifest entry: {filename}")
        candidate_path = pseudo_root / pseudo_row["output_json"]
        candidate = read_json(candidate_path)
        pseudo_c2 = [
            shape for shape in candidate.get("shapes", [])
            if str(shape.get("label") or "").upper() == "C2"
            and shape.get("shape_type") == "polygon"
            and bool((shape.get("flags") or {}).get("pseudo_label"))
        ]
        if len(pseudo_c2) != 1 or len(pseudo_c2[0].get("points") or []) != 4:
            raise ValueError(f"Expected one reviewed four-point pseudo C2: {filename}")

        patient_id = audit["patient_id"]
        stem = Path(filename).stem
        image_path = first_images_root / relative_image
        raw_json = raw_root / patient_id / f"{stem}.json"
        if not image_path.is_file() or not raw_json.is_file():
            raise FileNotFoundError(f"Missing first-batch source for {filename}")
        width, height = read_png_size(image_path)
        raw_data = read_json(raw_json)
        raw_shapes = unique_target_shapes(raw_data, raw_labels)
        lines: list[str] = []
        present: list[str] = []
        c2_line, clipped = make_pose_line(CLASS_TO_ID["C2"], pseudo_c2[0]["points"], pseudo_c2[0]["points"], width, height)
        lines.append(c2_line)
        present.append("C2")
        stats["clipped_points"] += clipped
        fallbacks = 0
        for label in CLASS_NAMES[1:]:
            if label not in raw_shapes:
                continue
            corners, fallback = dense_polygon_to_corners(raw_shapes[label])
            line, clipped = make_pose_line(CLASS_TO_ID[label], corners, raw_shapes[label], width, height)
            lines.append(line)
            present.append(label)
            fallbacks += int(fallback)
            stats["clipped_points"] += clipped
        stats["corner_fallbacks"] += fallbacks
        samples.append(Sample("first", stem, f"first:{patient_id}", image_path, tuple(lines), tuple(present), review.get("note", ""), fallbacks))
    return samples, excluded, stats


def remap_second_line(line: str, location: str) -> tuple[str | None, str | None]:
    parts = line.split()
    if len(parts) != 17:
        raise ValueError(f"{location}: expected 17 YOLO Pose fields, got {len(parts)}")
    old_id = int(parts[0])
    if old_id not in SECOND_ID_MAP:
        if 1 <= old_id <= 4:
            return None, None
        raise ValueError(f"{location}: unknown old class id {old_id}")
    new_id = SECOND_ID_MAP[old_id]
    values = [float(value) for value in parts[1:]]
    coords = [values[index] for index in (4, 5, 7, 8, 10, 11, 13, 14)]
    if any(value < 0 or value > 1 for value in coords):
        raise ValueError(f"{location}: keypoint outside [0, 1]")
    points = [(coords[index], coords[index + 1]) for index in range(0, 8, 2)]
    validate_corners(points)
    return " ".join([str(new_id), *parts[1:]]), CLASS_NAMES[new_id]


def load_second_batch(second_root: Path) -> tuple[list[Sample], list[dict[str, str]]]:
    samples: list[Sample] = []
    for row in read_csv(second_root / "manifest.csv"):
        image_path = second_root / row["image"]
        label_path = second_root / row["label"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing second-batch pair: {row['stem']}")
        lines: list[str] = []
        labels: list[str] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            mapped, label = remap_second_line(line, f"{label_path}:{line_number}")
            if mapped is not None and label is not None:
                lines.append(mapped)
                labels.append(label)
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate remapped class in {label_path}")
        samples.append(Sample("second", row["stem"], f"second:{row['patient_id']}", image_path, tuple(lines), tuple(labels)))
    excluded = [
        {"source_batch": "second", "source_stem": row["stem"], "reason": f"upstream_{row['reason']}"}
        for row in read_csv(second_root / "excluded_samples.csv")
    ]
    return samples, excluded


def stable_group_split(
    patient_ids: Iterable[str],
    forced_train: set[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    unique = sorted(set(patient_ids))
    if not forced_train <= set(unique):
        raise ValueError("Forced-train patients are not present")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Invalid split ratios")
    available = sorted(
        set(unique) - forced_train,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    test_count = round(len(unique) * test_ratio)
    val_count = round(len(unique) * val_ratio)
    if test_count + val_count > len(available):
        raise ValueError("Too many forced-train groups for requested split")
    result = {patient: "train" for patient in unique}
    for patient in available[:test_count]:
        result[patient] = "test"
    for patient in available[test_count:test_count + val_count]:
        result[patient] = "val"
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_exact_duplicates(samples: Sequence[Sample]) -> None:
    by_size: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_size[sample.image_path.stat().st_size].append(sample)
    seen: dict[str, Sample] = {}
    for group in by_size.values():
        if len(group) < 2:
            continue
        for sample in group:
            digest = sha256(sample.image_path)
            if digest in seen:
                other = seen[digest]
                raise ValueError(f"Exact duplicate images: {other.source_batch}/{other.source_stem} and {sample.source_batch}/{sample.source_stem}")
            seen[digest] = sample


def write_data_yaml(path: Path) -> None:
    lines = [
        "# Reviewed combined lateral-spine YOLO Pose dataset",
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


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    first, first_excluded, first_stats = load_first_batch(
        args.review_csv, args.source_audit_csv, args.pseudo_manifest_csv,
        args.pseudo_root, args.first_images_root, args.raw_first_root,
    )
    second, second_excluded = load_second_batch(args.second_root)
    samples = first + second
    if len(first) != 297 or len(second) != 698 or len(samples) != 995:
        raise ValueError(f"Unexpected included counts: first={len(first)}, second={len(second)}, total={len(samples)}")
    assert_no_exact_duplicates(samples)
    t13_patients = {sample.patient_id for sample in samples if "T13" in sample.class_names}
    splits = stable_group_split((sample.patient_id for sample in samples), t13_patients, args.val_ratio, args.test_ratio, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    manifest: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    split_images: Counter[str] = Counter()
    split_objects: Counter[str] = Counter()
    split_patients: dict[str, set[str]] = defaultdict(set)
    try:
        for split in ("train", "val", "test"):
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
        for sample in sorted(samples, key=lambda item: (item.source_batch, item.source_stem)):
            split = splits[sample.patient_id]
            output_stem = f"{sample.source_batch}__{sample.source_stem}"
            destination_image = temporary / "images" / split / f"{output_stem}.png"
            destination_label = temporary / "labels" / split / f"{output_stem}.txt"
            shutil.copy2(sample.image_path, destination_image)
            destination_label.write_text("\n".join(sample.label_lines) + "\n", encoding="utf-8")
            class_counts.update(sample.class_names)
            split_images[split] += 1
            split_objects[split] += len(sample.label_lines)
            split_patients[split].add(sample.patient_id)
            manifest.append({
                "output_stem": output_stem,
                "source_batch": sample.source_batch,
                "source_stem": sample.source_stem,
                "patient_id": sample.patient_id,
                "split": split,
                "image": f"images/{split}/{output_stem}.png",
                "label": f"labels/{split}/{output_stem}.txt",
                "objects": len(sample.label_lines),
                "present_classes": ";".join(sample.class_names),
                "missing_classes": ";".join(name for name in CLASS_NAMES if name not in sample.class_names),
                "review_note": sample.review_note,
                "corner_fallbacks": sample.corner_fallbacks,
            })
        if any(split_patients[a] & split_patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise AssertionError("Patient leakage detected")
        if any(splits[patient] != "train" for patient in t13_patients):
            raise AssertionError("A T13 patient was not placed in train")

        excluded = first_excluded + second_excluded
        with (temporary / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        with (temporary / "excluded_samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_batch", "source_stem", "reason"])
            writer.writeheader()
            writer.writerows(sorted(excluded, key=lambda row: (row["source_batch"], row["source_stem"])))
        metadata = temporary / "metadata"
        metadata.mkdir()
        shutil.copy2(args.review_csv, metadata / "first_batch_human_review.csv")
        shutil.copy2(args.source_audit_csv, metadata / "first_batch_label_source_audit.csv")
        write_data_yaml(temporary / "data.yaml")
        report: dict[str, Any] = {
            "schema_version": 1,
            "classes": CLASS_NAMES,
            "metric_focus_classes": ["C2", "C7"],
            "keypoint_order": KEYPOINT_ORDER,
            "flip_idx": FLIP_IDX,
            "corner_algorithm": "convex hull followed by minimum local-area-loss reduction to four vertices; minimum-area rectangle only for triangular hull",
            "split": {"seed": args.seed, "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "t13_forced_to_train": True},
            "counts": {
                "included_images": len(samples),
                "included_by_batch": {"first": len(first), "second": len(second)},
                "included_patients": len(set(sample.patient_id for sample in samples)),
                "excluded_by_batch": {"first": len(first_excluded), "second_upstream": len(second_excluded)},
                "images_by_split": dict(sorted(split_images.items())),
                "patients_by_split": {split: len(patients) for split, patients in sorted(split_patients.items())},
                "objects_by_split": dict(sorted(split_objects.items())),
                "objects_by_class": {name: class_counts[name] for name in CLASS_NAMES},
                "first_dense_polygon_triangle_fallbacks": first_stats["corner_fallbacks"],
                "first_clipped_keypoints": first_stats["clipped_points"],
                "t13_objects": class_counts["T13"],
                "t13_patients": len(t13_patients),
            },
            "quality_guards": [
                "First batch requires human accepted and source audit correct_same_stem.",
                "C3-C6 are absent from labels and the class map.",
                "Every object has four unique strictly convex keypoints; near-triangular edge ratios are rejected.",
                "Exact image duplicates are rejected before splitting.",
                "Batch-namespaced patient groups cannot cross train/val/test.",
            ],
        }
        (temporary / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "README.md").write_text(
            "# Reviewed combined lateral-spine 20-class YOLO Pose dataset\n\n"
            "Classes: `C2`, `C7`, `T1-T13`, `L1-L5`; `C3-C6` are intentionally excluded.\n\n"
            f"- Images: {len(samples)} (first accepted/source-clean {len(first)}; second cleaned {len(second)})\n"
            f"- Patient groups: {len(set(sample.patient_id for sample in samples))}\n"
            "- Keypoints: upper-left, upper-right, lower-right, lower-left\n"
            "- Primary evaluation targets: C2 and C7\n"
            "- T13 is retained but has only two first-batch objects; both patient groups are forced into train.\n"
            "- First-batch C2 is the human-accepted reviewed candidate. Other first-batch classes are rebuilt from exact same-stem LabelMe JSON.\n\n"
            "See `build_report.json`, `manifest.csv`, `excluded_samples.csv`, and `metadata/`.\n",
            encoding="utf-8",
        )
        temporary.rename(args.output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=Path.home() / "Downloads/yolo_corner仅补C2-C6人工接受拒绝结果.csv")
    parser.add_argument("--source-audit-csv", type=Path, default=root / "analysis/yolo_corner_label_source_audit/per_image_audit.csv")
    parser.add_argument("--pseudo-manifest-csv", type=Path, default=root / "datasets/lateral_first_batch_pseudolabel_23cls/manifest.csv")
    parser.add_argument("--pseudo-root", type=Path, default=root / "datasets/lateral_first_batch_pseudolabel_23cls")
    parser.add_argument("--first-images-root", type=Path, default=root / "datasets/yolo_corner")
    parser.add_argument("--raw-first-root", type=Path, default=Path("/Volumes/E/spine_data/LAT202511"))
    parser.add_argument("--second-root", type=Path, default=root / "datasets/yolo_lateral_20260903_23cls")
    parser.add_argument("--output", type=Path, default=root / "datasets/yolo_lateral_reviewed_combined_20cls")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset(args)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"Created: {args.output.resolve()}")


if __name__ == "__main__":
    main()
