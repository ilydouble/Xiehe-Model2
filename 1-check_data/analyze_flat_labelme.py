#!/usr/bin/env python3
"""Read-only quality audit for a flat or recursively nested LabelMe dataset.

The scanner deliberately ignores macOS AppleDouble files (``._*``) and uses
only the Python standard library. PNG dimensions are read from the IHDR chunk,
so the audit does not require Pillow.

Example:
    python 1-check_data/analyze_flat_labelme.py \
        /path/to/labelme-export/P202607076308 \
        --output analysis/side_labelme_20260903/audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = list(values)
    if not numbers:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "p25": percentile(numbers, 0.25),
        "median": statistics.median(numbers),
        "p75": percentile(numbers, 0.75),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
    }


def read_png_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(29)
    if len(header) < 29 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG signature or IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


def canonical_anatomical_index(label: str) -> float | None:
    match = re.fullmatch(r"([CTLS])(\d+)", label.strip().upper())
    if not match:
        return None
    region, raw_number = match.groups()
    number = int(raw_number)
    if region == "C" and 1 <= number <= 7:
        return float(number)
    if region == "T" and 1 <= number <= 12:
        return float(7 + number)
    if region == "T" and number == 13:
        return 19.5
    if region == "L" and 1 <= number <= 5:
        return float(19 + number)
    if region == "S" and number == 1:
        return 25.0
    return None


def source_prefix(stem: str) -> str:
    match = re.match(r"([A-Za-z]+)", stem)
    return match.group(1).upper() if match else "[other]"


def audit_dataset(root: Path) -> dict[str, Any]:
    all_files = [path for path in root.rglob("*") if path.is_file()]
    appledouble = [path for path in all_files if path.name.startswith("._")]
    files = [path for path in all_files if not path.name.startswith("._")]
    images = sorted(path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS)
    json_files = sorted(path for path in files if path.suffix.lower() == ".json")

    ext_counts = Counter(path.suffix.lower() or "[no_ext]" for path in files)
    image_by_stem: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    json_by_stem: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for path in images:
        image_by_stem[(path.parent, path.stem)].append(path)
    for path in json_files:
        json_by_stem[(path.parent, path.stem)].append(path)

    image_keys = set(image_by_stem)
    json_keys = set(json_by_stem)
    image_without_json = [relative(path, root) for key in sorted(image_keys - json_keys, key=str) for path in image_by_stem[key]]
    json_without_image = [relative(path, root) for key in sorted(json_keys - image_keys, key=str) for path in json_by_stem[key]]
    ambiguous_image_stems = {
        f"{relative(parent, root)}/{stem}".lstrip("./"): [relative(path, root) for path in paths]
        for (parent, stem), paths in image_by_stem.items() if len(paths) > 1
    }

    png_metadata: dict[Path, dict[str, int]] = {}
    invalid_images: list[dict[str, str]] = []
    dimension_counts: Counter[str] = Counter()
    png_format_counts: Counter[str] = Counter()
    image_bytes: list[float] = []
    for path in images:
        image_bytes.append(float(path.stat().st_size))
        if path.suffix.lower() != ".png":
            continue
        try:
            metadata = read_png_header(path)
            png_metadata[path] = metadata
            dimension_counts[f"{metadata['width']}x{metadata['height']}"] += 1
            png_format_counts[f"bit{metadata['bit_depth']}/color{metadata['color_type']}"] += 1
        except Exception as exc:  # continue scanning after a corrupt image
            invalid_images.append({"file": relative(path, root), "error": str(exc)})

    file_sizes: dict[int, list[Path]] = defaultdict(list)
    for path in images:
        file_sizes[path.stat().st_size].append(path)
    duplicate_groups: list[list[str]] = []
    for same_size_paths in file_sizes.values():
        if len(same_size_paths) < 2:
            continue
        digest_paths: dict[str, list[Path]] = defaultdict(list)
        for path in same_size_paths:
            digest_paths[sha256(path)].append(path)
        duplicate_groups.extend(
            [relative(path, root) for path in paths]
            for paths in digest_paths.values() if len(paths) > 1
        )

    label_shape_counts: Counter[str] = Counter()
    label_image_counts: Counter[str] = Counter()
    label_shape_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    shape_type_counts: Counter[str] = Counter()
    version_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter(source_prefix(path.stem) for path in images)
    source_png_format_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path, metadata in png_metadata.items():
        source_png_format_counts[source_prefix(path.stem)][f"bit{metadata['bit_depth']}/color{metadata['color_type']}"] += 1
    annotation_profile_counts: Counter[str] = Counter()
    source_annotation_profile_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels_per_image: list[float] = []
    shapes_per_image: list[float] = []
    points_per_shape: list[float] = []
    bbox_area_ratios: list[float] = []
    polygon_area_ratios: list[float] = []
    label_point_counts: dict[str, list[int]] = defaultdict(list)
    label_area_ratios: dict[str, list[float]] = defaultdict(list)
    signature_counts: Counter[str] = Counter()
    invalid_json: list[dict[str, str]] = []
    empty_annotations: list[str] = []
    missing_or_blank_labels: list[dict[str, Any]] = []
    invalid_points: list[dict[str, Any]] = []
    insufficient_points: list[dict[str, Any]] = []
    degenerate_shapes: list[dict[str, Any]] = []
    out_of_bounds_shapes: list[dict[str, Any]] = []
    tiny_shapes: list[dict[str, Any]] = []
    duplicate_labels: list[dict[str, Any]] = []
    dimension_mismatches: list[dict[str, Any]] = []
    image_path_issues: list[dict[str, Any]] = []
    internal_level_gaps: list[dict[str, Any]] = []
    vertical_order_issues: list[dict[str, Any]] = []
    label_variants: dict[str, set[str]] = defaultdict(set)
    embedded_image_data_count = 0
    nonempty_flags_count = 0
    nonnull_group_id_count = 0
    nonempty_description_count = 0
    total_shapes = 0
    total_points = 0

    for json_path in json_files:
        json_rel = relative(json_path, root)
        try:
            with json_path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except Exception as exc:
            invalid_json.append({"file": json_rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(data, dict):
            invalid_json.append({"file": json_rel, "error": "top-level JSON is not an object"})
            continue

        version_counts[str(data.get("version", "[missing]"))] += 1
        if data.get("imageData"):
            embedded_image_data_count += 1
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            invalid_json.append({"file": json_rel, "error": "shapes is not a list"})
            continue
        if not shapes:
            empty_annotations.append(json_rel)
        shapes_per_image.append(float(len(shapes)))

        paired_images = image_by_stem.get((json_path.parent, json_path.stem), [])
        actual_image = paired_images[0] if len(paired_images) == 1 else None
        actual_meta = png_metadata.get(actual_image) if actual_image else None
        json_width = data.get("imageWidth")
        json_height = data.get("imageHeight")
        width = actual_meta["width"] if actual_meta else json_width
        height = actual_meta["height"] if actual_meta else json_height
        if actual_meta and (json_width != actual_meta["width"] or json_height != actual_meta["height"]):
            dimension_mismatches.append({
                "json": json_rel,
                "json_size": [json_width, json_height],
                "actual_size": [actual_meta["width"], actual_meta["height"]],
            })

        image_path = data.get("imagePath")
        if not isinstance(image_path, str) or not image_path.strip():
            image_path_issues.append({"json": json_rel, "issue": "missing imagePath"})
        else:
            declared_name = Path(image_path.replace("\\", "/")).name
            actual_names = {path.name for path in paired_images}
            if actual_names and declared_name not in actual_names:
                image_path_issues.append({
                    "json": json_rel,
                    "issue": "imagePath basename differs from stem-paired image",
                    "imagePath": image_path,
                    "paired_images": sorted(actual_names),
                })

        seen_labels: Counter[str] = Counter()
        centroids: list[tuple[float, float, str]] = []
        anatomical_present: set[str] = set()
        per_image_labels: set[str] = set()
        for shape_index, shape in enumerate(shapes):
            total_shapes += 1
            if not isinstance(shape, dict):
                invalid_points.append({"json": json_rel, "shape_index": shape_index, "issue": "shape is not an object"})
                continue
            raw_label = shape.get("label")
            if not isinstance(raw_label, str) or not raw_label.strip():
                missing_or_blank_labels.append({"json": json_rel, "shape_index": shape_index})
                label = "[missing]"
            else:
                label = raw_label.strip()
                label_variants[label.casefold()].add(raw_label)
            shape_type = str(shape.get("shape_type", "[missing]"))
            shape_type_counts[shape_type] += 1
            label_shape_counts[label] += 1
            label_shape_type_counts[label][shape_type] += 1
            seen_labels[label] += 1
            per_image_labels.add(label)
            if shape.get("flags"):
                nonempty_flags_count += 1
            if shape.get("group_id") is not None:
                nonnull_group_id_count += 1
            if shape.get("description"):
                nonempty_description_count += 1

            raw_points = shape.get("points")
            parsed_points: list[tuple[float, float]] = []
            if isinstance(raw_points, list):
                for point in raw_points:
                    if (
                        isinstance(point, (list, tuple)) and len(point) >= 2
                        and isinstance(point[0], (int, float)) and isinstance(point[1], (int, float))
                        and math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
                    ):
                        parsed_points.append((float(point[0]), float(point[1])))
                    else:
                        invalid_points.append({
                            "json": json_rel,
                            "shape_index": shape_index,
                            "label": label,
                            "point": point,
                        })
            else:
                invalid_points.append({"json": json_rel, "shape_index": shape_index, "label": label, "points": raw_points})

            point_count = len(parsed_points)
            total_points += point_count
            points_per_shape.append(float(point_count))
            label_point_counts[label].append(point_count)
            minimum_points = {"polygon": 3, "rectangle": 2, "circle": 2, "line": 2, "linestrip": 2, "point": 1}.get(shape_type, 1)
            if point_count < minimum_points:
                insufficient_points.append({
                    "json": json_rel,
                    "shape_index": shape_index,
                    "label": label,
                    "shape_type": shape_type,
                    "point_count": point_count,
                    "minimum": minimum_points,
                })
            if not parsed_points:
                continue

            xs = [point[0] for point in parsed_points]
            ys = [point[1] for point in parsed_points]
            bbox_width = max(xs) - min(xs)
            bbox_height = max(ys) - min(ys)
            bbox_area = bbox_width * bbox_height
            polygon_shape_area = polygon_area(parsed_points) if shape_type == "polygon" else bbox_area
            surface_shape = shape_type in {"polygon", "rectangle", "circle"}
            line_shape = shape_type in {"line", "linestrip"}
            line_length = sum(
                math.hypot(x2 - x1, y2 - y1)
                for (x1, y1), (x2, y2) in zip(parsed_points, parsed_points[1:])
            )
            if (
                (surface_shape and (bbox_width <= 0 or bbox_height <= 0 or polygon_shape_area <= 0))
                or (line_shape and line_length <= 0)
            ):
                degenerate_shapes.append({
                    "json": json_rel,
                    "shape_index": shape_index,
                    "label": label,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "polygon_area": polygon_shape_area,
                })
            centroid_y = statistics.fmean(ys)
            anatomical_index = canonical_anatomical_index(label)
            if anatomical_index is not None:
                anatomical_present.add(label.upper())
                centroids.append((anatomical_index, centroid_y, label))

            if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
                image_area = float(width) * float(height)
                bbox_ratio = bbox_area / image_area
                polygon_ratio = polygon_shape_area / image_area
                if surface_shape:
                    bbox_area_ratios.append(bbox_ratio)
                    polygon_area_ratios.append(polygon_ratio)
                    label_area_ratios[label].append(polygon_ratio)
                if surface_shape and polygon_ratio < 1e-5:
                    tiny_shapes.append({
                        "json": json_rel,
                        "shape_index": shape_index,
                        "label": label,
                        "area_ratio": polygon_ratio,
                    })
                oob = [
                    [x, y] for x, y in parsed_points
                    if x < 0 or y < 0 or x > float(width) or y > float(height)
                ]
                if oob:
                    out_of_bounds_shapes.append({
                        "json": json_rel,
                        "shape_index": shape_index,
                        "label": label,
                        "image_size": [width, height],
                        "points": oob,
                    })

        for label in per_image_labels:
            label_image_counts[label] += 1
        labels_per_image.append(float(len(per_image_labels)))
        if {"FH-1", "FH-2"}.issubset(per_image_labels):
            annotation_profile = "FH-1+FH-2"
        elif "CFH" in per_image_labels:
            annotation_profile = "CFH"
        elif {"FH-1", "FH-2"} & per_image_labels:
            annotation_profile = "incomplete_FH_pair"
        else:
            annotation_profile = "no_femoral_head_marker"
        annotation_profile_counts[annotation_profile] += 1
        source_annotation_profile_counts[source_prefix(json_path.stem)][annotation_profile] += 1
        signature_counts[", ".join(sorted(per_image_labels, key=lambda item: (canonical_anatomical_index(item) is None, canonical_anatomical_index(item) or 999, item)))] += 1
        repeated = {label: count for label, count in seen_labels.items() if count > 1}
        if repeated:
            duplicate_labels.append({"json": json_rel, "labels": repeated})

        normal_sequence = [
            *(f"C{i}" for i in range(2, 8)),
            *(f"T{i}" for i in range(1, 13)),
            *(f"L{i}" for i in range(1, 6)),
            "S1",
        ]
        normal_indices = [index for index, label in enumerate(normal_sequence) if label in anatomical_present]
        if len(normal_indices) >= 2:
            first, last = min(normal_indices), max(normal_indices)
            missing_internal = [label for label in normal_sequence[first:last + 1] if label not in anatomical_present]
            if missing_internal:
                internal_level_gaps.append({"json": json_rel, "missing_between_first_and_last": missing_internal})

        ordered_centroids = sorted(centroids)
        inversions = []
        for first_item, second_item in zip(ordered_centroids, ordered_centroids[1:]):
            if second_item[1] <= first_item[1]:
                inversions.append({
                    "first": first_item[2], "first_y": first_item[1],
                    "second": second_item[2], "second_y": second_item[1],
                })
        if inversions:
            vertical_order_issues.append({"json": json_rel, "inversions": inversions})

    variant_collisions = {
        normalized: sorted(variants)
        for normalized, variants in label_variants.items()
        if len(variants) > 1 or any(value != value.strip() for value in variants)
    }
    label_stats = {}
    for label in sorted(label_shape_counts, key=lambda item: (canonical_anatomical_index(item) is None, canonical_anatomical_index(item) or 999, item)):
        label_stats[label] = {
            "shapes": label_shape_counts[label],
            "images": label_image_counts[label],
            "shape_types": dict(label_shape_type_counts[label].most_common()),
            "points_per_shape": numeric_summary(float(value) for value in label_point_counts[label]),
            "area_ratio": numeric_summary(label_area_ratios[label]),
        }

    core_labels = [
        *(f"C{i}" for i in range(2, 8)),
        *(f"T{i}" for i in range(1, 13)),
        *(f"L{i}" for i in range(1, 6)),
        "S1",
    ]
    core_missing_counts = {
        label: len(json_files) - label_image_counts[label]
        for label in core_labels
    }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "inventory": {
            "all_files_including_appledouble": len(all_files),
            "ignored_appledouble_files": len(appledouble),
            "files_by_extension_excluding_appledouble": dict(sorted(ext_counts.items())),
            "images": len(images),
            "json_files": len(json_files),
            "paired_stems": len(image_keys & json_keys),
            "image_without_json": image_without_json,
            "json_without_image": json_without_image,
            "ambiguous_image_stems": ambiguous_image_stems,
            "source_prefix_counts": dict(source_counts.most_common()),
            "source_png_format_counts": {
                source: dict(counts.most_common())
                for source, counts in sorted(source_png_format_counts.items())
            },
        },
        "images": {
            "invalid_images": invalid_images,
            "dimension_counts": dict(dimension_counts.most_common()),
            "png_format_counts": dict(png_format_counts.most_common()),
            "file_size_bytes": numeric_summary(image_bytes),
            "exact_duplicate_groups": sorted(duplicate_groups, key=lambda group: (-len(group), group)),
        },
        "annotations": {
            "valid_json_files": len(json_files) - len(invalid_json),
            "invalid_json": invalid_json,
            "version_counts": dict(version_counts.most_common()),
            "total_shapes": total_shapes,
            "total_points": total_points,
            "shape_type_counts": dict(shape_type_counts.most_common()),
            "shapes_per_image": numeric_summary(shapes_per_image),
            "unique_labels_per_image": numeric_summary(labels_per_image),
            "points_per_shape": numeric_summary(points_per_shape),
            "bbox_area_ratio": numeric_summary(bbox_area_ratios),
            "polygon_area_ratio": numeric_summary(polygon_area_ratios),
            "label_stats": label_stats,
            "core_missing_counts": core_missing_counts,
            "annotation_profile_counts": dict(annotation_profile_counts.most_common()),
            "source_annotation_profile_counts": {
                source: dict(counts.most_common())
                for source, counts in sorted(source_annotation_profile_counts.items())
            },
            "label_set_signatures": dict(signature_counts.most_common()),
            "label_variant_collisions": variant_collisions,
            "embedded_image_data_count": embedded_image_data_count,
            "nonempty_flags_count": nonempty_flags_count,
            "nonnull_group_id_count": nonnull_group_id_count,
            "nonempty_description_count": nonempty_description_count,
        },
        "quality_issues": {
            "empty_annotations": empty_annotations,
            "missing_or_blank_labels": missing_or_blank_labels,
            "invalid_points": invalid_points,
            "insufficient_points": insufficient_points,
            "degenerate_shapes": degenerate_shapes,
            "out_of_bounds_shapes": out_of_bounds_shapes,
            "tiny_shapes_area_ratio_below_1e-5": tiny_shapes,
            "duplicate_labels_within_image": duplicate_labels,
            "dimension_mismatches": dimension_mismatches,
            "image_path_issues": image_path_issues,
            "internal_level_gaps": internal_level_gaps,
            "vertical_order_issues": vertical_order_issues,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a flat/recursive LabelMe dataset without modifying it")
    parser.add_argument("dataset", type=Path, help="Dataset directory to scan recursively")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"dataset directory does not exist: {dataset}")
    report = audit_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "images": report["inventory"]["images"],
        "json_files": report["inventory"]["json_files"],
        "paired_stems": report["inventory"]["paired_stems"],
        "total_shapes": report["annotations"]["total_shapes"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
