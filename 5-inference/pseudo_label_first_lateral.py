#!/usr/bin/env python3
"""Create review-only LabelMe pseudo-labels for the first lateral batch.

The source directory is always treated as read-only.  For every exactly paired
PNG/JSON sample, the model is run on up to three full-context views:

1. the original image;
2. a conservative crop of continuous black edge bands;
3. a full-height crop matching the newer batch's median aspect ratio, centred
   on the existing vertebral polygons.

Predictions are mapped back to original-image coordinates and reconciled across
views.  Only missing vertebral polygons are appended; all existing LabelMe
shapes are preserved byte-for-byte at the JSON-object level.  Every appended
shape is explicitly marked as a pseudo-label requiring human review.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CLASS_NAMES = [
    *(f"C{i}" for i in range(2, 8)),
    *(f"T{i}" for i in range(1, 13)),
    *(f"L{i}" for i in range(1, 6)),
]
CLASS_SET = set(CLASS_NAMES)
DEFAULT_MODEL = Path(
    "3-model_training/runs/pose/yolo11l_lateral_23cls_best/weights/best.pt"
)
DEFAULT_SOURCE = Path("/Volumes/E/spine_data/LAT202511")
DEFAULT_OUTPUT = Path("datasets/lateral_first_batch_pseudolabel_23cls")


@dataclass(frozen=True)
class View:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Candidate:
    label: str
    box_conf: float
    keypoint_conf: float
    points: tuple[tuple[float, float], ...]
    box: tuple[float, float, float, float]
    view: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_samples(source: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    """Return exact PNG/JSON pairs and unmatched files, ignoring AppleDouble."""
    images: dict[tuple[Path, str], Path] = {}
    annotations: dict[tuple[Path, str], Path] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        key = (path.parent, path.stem)
        suffix = path.suffix.lower()
        if suffix == ".png":
            if key in images:
                raise ValueError(f"Duplicate PNG stem: {path}")
            images[key] = path
        elif suffix == ".json":
            if key in annotations:
                raise ValueError(f"Duplicate JSON stem: {path}")
            annotations[key] = path
    paired_keys = sorted(set(images) & set(annotations), key=lambda item: str(images[item]))
    pairs = [(images[key], annotations[key]) for key in paired_keys]
    image_only = sorted(images[key] for key in set(images) - set(annotations))
    json_only = sorted(annotations[key] for key in set(annotations) - set(images))
    return pairs, image_only, json_only


def load_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"Selection file is empty: {path}")
    return values


def load_annotation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data.get("shapes"), list):
        raise ValueError(f"LabelMe shapes is not a list: {path}")
    return data


def target_polygon_shapes(annotation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label") or "").strip().upper()
        if label in CLASS_SET and shape.get("shape_type") == "polygon":
            points = shape.get("points")
            if not isinstance(points, list) or len(points) < 4:
                continue
            by_label[label].append(shape)
    return dict(by_label)


def points_box(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def annotation_points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for shapes in target_polygon_shapes(annotation).values():
        for shape in shapes:
            result.extend((float(point[0]), float(point[1])) for point in shape["points"])
    return result


def plan_views(
    width: int,
    height: int,
    black_bands: dict[str, int],
    existing_points: Sequence[tuple[float, float]],
    target_aspect: float = 0.353,
    guard_pixels: int = 2,
) -> list[View]:
    """Plan unique, full-context inference views in original pixel coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    views = [View("original", 0, 0, width, height)]

    left = max(0, int(black_bands.get("left", 0)) - guard_pixels)
    right = max(0, int(black_bands.get("right", 0)) - guard_pixels)
    top = max(0, int(black_bands.get("top", 0)) - guard_pixels)
    bottom = max(0, int(black_bands.get("bottom", 0)) - guard_pixels)
    if left + right < width and top + bottom < height and any((left, right, top, bottom)):
        views.append(View("black_trim", left, top, width - right, height - bottom))

    desired_width = min(width, max(1, round(height * target_aspect)))
    if existing_points and desired_width < width:
        xs = [point[0] for point in existing_points]
        annotated_width = max(xs) - min(xs)
        desired_width = min(width, max(desired_width, math.ceil(annotated_width * 1.10)))
        center_x = (min(xs) + max(xs)) / 2.0
        x0 = max(0, min(width - desired_width, round(center_x - desired_width / 2.0)))
        views.append(View("training_aspect", x0, 0, x0 + desired_width, height))

    unique: list[View] = []
    seen: set[tuple[int, int, int, int]] = set()
    for view in views:
        geometry = (view.x0, view.y0, view.x1, view.y1)
        if geometry not in seen:
            unique.append(view)
            seen.add(geometry)
    return unique


def detect_black_bands(
    image: Any,
    sample_width: int = 256,
    near_black_threshold: int = 5,
    required_fraction: float = 0.98,
    notable_ratio: float = 0.01,
) -> dict[str, int]:
    """Detect continuous, nearly-black bands and return source-pixel thicknesses."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    sample_height = max(1, round(height * sample_width / width))
    gray = cv2.cvtColor(
        cv2.resize(image, (sample_width, sample_height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )

    def count_lines(axis: int, reverse: bool) -> int:
        length = gray.shape[axis]
        indices = range(length - 1, -1, -1) if reverse else range(length)
        count = 0
        for index in indices:
            line = gray[index, :] if axis == 0 else gray[:, index]
            if float(np.mean(line <= near_black_threshold)) < required_fraction:
                break
            count += 1
        return count

    sample_bands = {
        "top": count_lines(0, False),
        "bottom": count_lines(0, True),
        "left": count_lines(1, False),
        "right": count_lines(1, True),
    }
    result: dict[str, int] = {}
    for side, count in sample_bands.items():
        denominator = sample_height if side in {"top", "bottom"} else sample_width
        source_length = height if side in {"top", "bottom"} else width
        ratio = count / denominator
        result[side] = round(ratio * source_length) if ratio >= notable_ratio else 0
    return result


def box_iou(
    first: Sequence[float], second: Sequence[float]
) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def candidate_distance(first: Candidate, second: Candidate) -> float:
    distances = [
        math.hypot(a[0] - b[0], a[1] - b[1])
        for a, b in zip(first.points, second.points)
    ]
    diagonals = [
        math.hypot(candidate.box[2] - candidate.box[0], candidate.box[3] - candidate.box[1])
        for candidate in (first, second)
    ]
    scale = max(1.0, sum(diagonals) / len(diagonals))
    return sum(distances) / len(distances) / scale


def candidates_agree(first: Candidate, second: Candidate) -> bool:
    return box_iou(first.box, second.box) >= 0.15 or candidate_distance(first, second) <= 0.60


def cluster_candidates(candidates: Sequence[Candidate]) -> list[list[Candidate]]:
    """Greedily cluster same-label predictions, with at most one item per view."""
    clusters: list[list[Candidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.box_conf, reverse=True):
        matching = [
            cluster
            for cluster in clusters
            if candidate.view not in {item.view for item in cluster}
            and any(candidates_agree(candidate, item) for item in cluster)
        ]
        if matching:
            best = max(
                matching,
                key=lambda cluster: max(box_iou(candidate.box, item.box) for item in cluster),
            )
            best.append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def merge_cluster(cluster: Sequence[Candidate]) -> Candidate:
    if not cluster:
        raise ValueError("Cannot merge an empty cluster")
    label = cluster[0].label
    if any(candidate.label != label for candidate in cluster):
        raise ValueError("Cannot merge different labels")
    count = len(cluster)
    points = tuple(
        (
            sum(candidate.points[index][0] for candidate in cluster) / count,
            sum(candidate.points[index][1] for candidate in cluster) / count,
        )
        for index in range(4)
    )
    return Candidate(
        label=label,
        box_conf=sum(item.box_conf for item in cluster) / count,
        keypoint_conf=sum(item.keypoint_conf for item in cluster) / count,
        points=points,
        box=points_box(points),
        view="+".join(sorted(item.view for item in cluster)),
    )


def choose_candidate(
    candidates: Sequence[Candidate],
    anchor_scores: dict[str, float],
) -> tuple[Candidate, list[Candidate]] | None:
    if not candidates:
        return None
    clusters = cluster_candidates(candidates)

    def score(cluster: list[Candidate]) -> tuple[int, float, float]:
        support = len({item.view for item in cluster})
        confidence = sum(item.box_conf for item in cluster) / len(cluster)
        anchor = sum(anchor_scores.get(item.view, 0.0) for item in cluster) / len(cluster)
        return support, confidence + 0.15 * anchor, confidence

    selected = max(clusters, key=score)
    return merge_cluster(selected), selected


def prediction_candidates(result: Any, names: dict[int, str], view: View) -> list[Candidate]:
    if result.boxes is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    keypoints = result.keypoints.xy.cpu().tolist()
    keypoint_confidences = (
        result.keypoints.conf.cpu().tolist()
        if result.keypoints.conf is not None
        else [[1.0] * 4 for _ in keypoints]
    )
    output: list[Candidate] = []
    for box, class_id, confidence, points, point_conf in zip(
        boxes, classes, confidences, keypoints, keypoint_confidences
    ):
        label = names[int(class_id)]
        if label not in CLASS_SET or len(points) != 4:
            continue
        mapped = tuple(
            (float(point[0]) + view.x0, float(point[1]) + view.y0)
            for point in points
        )
        output.append(
            Candidate(
                label=label,
                box_conf=float(confidence),
                keypoint_conf=sum(float(value) for value in point_conf) / len(point_conf),
                points=mapped,
                box=(
                    float(box[0]) + view.x0,
                    float(box[1]) + view.y0,
                    float(box[2]) + view.x0,
                    float(box[3]) + view.y0,
                ),
                view=view.name,
            )
        )
    return output


def anchor_scores(
    annotation: dict[str, Any], candidates: Sequence[Candidate], views: Sequence[View]
) -> dict[str, float]:
    existing = target_polygon_shapes(annotation)
    gt = {
        label: points_box(shapes[0]["points"])
        for label, shapes in existing.items()
        if len(shapes) == 1
    }
    scores: dict[str, float] = {}
    for view in views:
        overlaps: list[float] = []
        for label, box in gt.items():
            overlaps.append(
                max(
                    [
                        box_iou(box, item.box)
                        for item in candidates
                        if item.view == view.name and item.label == label
                    ]
                    or [0.0]
                )
            )
        if not overlaps:
            scores[view.name] = 0.0
        else:
            hit_rate = sum(value >= 0.10 for value in overlaps) / len(overlaps)
            scores[view.name] = 0.7 * hit_rate + 0.3 * (sum(overlaps) / len(overlaps))
    return scores


def center_y(points: Sequence[Sequence[float]]) -> float:
    return sum(float(point[1]) for point in points) / len(points)


def anatomy_validity(
    annotation: dict[str, Any], selected: dict[str, Candidate]
) -> dict[str, bool]:
    ordered_y: dict[str, float] = {}
    for label, shapes in target_polygon_shapes(annotation).items():
        if len(shapes) == 1:
            ordered_y[label] = center_y(shapes[0]["points"])
    ordered_y.update({label: center_y(item.points) for label, item in selected.items()})
    validity = {label: True for label in selected}
    for index in range(len(CLASS_NAMES) - 1):
        upper, lower = CLASS_NAMES[index], CLASS_NAMES[index + 1]
        if upper in ordered_y and lower in ordered_y and ordered_y[upper] >= ordered_y[lower]:
            if upper in validity:
                validity[upper] = False
            if lower in validity:
                validity[lower] = False
    return validity


def quality_level(
    candidate: Candidate,
    cluster: Sequence[Candidate],
    anatomy_ok: bool,
    anchor: dict[str, float],
) -> str:
    support = len({item.view for item in cluster})
    mean_anchor = sum(anchor.get(item.view, 0.0) for item in cluster) / len(cluster)
    if anatomy_ok and support >= 2 and candidate.box_conf >= 0.25 and mean_anchor >= 0.25:
        return "high"
    if anatomy_ok and support >= 2 and candidate.box_conf >= 0.08:
        return "medium"
    return "low"


def make_pseudo_shape(
    candidate: Candidate,
    cluster: Sequence[Candidate],
    quality: str,
    anatomy_ok: bool,
    model_sha256: str,
) -> dict[str, Any]:
    views = sorted({item.view for item in cluster})
    return {
        "label": candidate.label,
        "points": [[round(x, 3), round(y, 3)] for x, y in candidate.points],
        "group_id": None,
        "description": (
            "PSEUDO_LABEL_REVIEW_REQUIRED; "
            f"quality={quality}; support={len(views)}; "
            f"box_conf={candidate.box_conf:.4f}; kpt_conf={candidate.keypoint_conf:.4f}; "
            f"anatomy_ok={str(anatomy_ok).lower()}; views={'+'.join(views)}; "
            f"model_sha256={model_sha256[:12]}"
        ),
        "shape_type": "polygon",
        "flags": {"pseudo_label": True, "needs_review": True, f"quality_{quality}": True},
        "mask": None,
    }


def append_missing_shapes(
    annotation: dict[str, Any],
    selected_with_clusters: dict[str, tuple[Candidate, list[Candidate]]],
    anchor: dict[str, float],
    model_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Return a copied LabelMe annotation with only missing polygons appended."""
    output = copy.deepcopy(annotation)
    existing = target_polygon_shapes(annotation)
    selected = {
        label: value[0]
        for label, value in selected_with_clusters.items()
        if label not in existing
    }
    validity = anatomy_validity(annotation, selected)
    added: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for label in CLASS_NAMES:
        if label in existing:
            continue
        value = selected_with_clusters.get(label)
        if value is None:
            unresolved.append(label)
            continue
        candidate, cluster = value
        quality = quality_level(candidate, cluster, validity.get(label, False), anchor)
        shape = make_pseudo_shape(
            candidate, cluster, quality, validity.get(label, False), model_sha256
        )
        output["shapes"].append(shape)
        added.append(
            {
                "label": label,
                "quality": quality,
                "box_conf": candidate.box_conf,
                "keypoint_conf": candidate.keypoint_conf,
                "support": len({item.view for item in cluster}),
                "views": "+".join(sorted({item.view for item in cluster})),
                "anatomy_ok": validity.get(label, False),
            }
        )
    return output, added, unresolved


def clip_candidates(candidates: Iterable[Candidate], width: int, height: int) -> list[Candidate]:
    output: list[Candidate] = []
    for item in candidates:
        points = tuple(
            (min(max(point[0], 0.0), float(width)), min(max(point[1], 0.0), float(height)))
            for point in item.points
        )
        if len(set(points)) != 4:
            continue
        output.append(
            Candidate(
                label=item.label,
                box_conf=item.box_conf,
                keypoint_conf=item.keypoint_conf,
                points=points,
                box=points_box(points),
                view=item.view,
            )
        )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_preview(image: Any, annotation: dict[str, Any], path: Path) -> None:
    import cv2
    import numpy as np

    canvas = image.copy()
    for shape in annotation.get("shapes", []):
        flags = shape.get("flags") or {}
        if not flags.get("pseudo_label"):
            continue
        points = np.asarray(shape["points"], dtype=np.int32)
        quality = next(
            (name.removeprefix("quality_") for name, value in flags.items() if name.startswith("quality_") and value),
            "low",
        )
        color = {"high": (0, 220, 0), "medium": (0, 180, 255), "low": (0, 0, 255)}[quality]
        cv2.polylines(canvas, [points], True, color, max(2, round(canvas.shape[1] / 800)))
        x, y = points[0]
        cv2.putText(
            canvas,
            f"{shape['label']} {quality}",
            (int(x), max(20, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, canvas.shape[1] / 3000),
            color,
            max(1, round(canvas.shape[1] / 1200)),
            cv2.LINE_AA,
        )
    max_height = 1800
    if canvas.shape[0] > max_height:
        scale = max_height / canvas.shape[0]
        canvas = cv2.resize(
            canvas,
            (max(1, round(canvas.shape[1] * scale)), max_height),
            interpolation=cv2.INTER_AREA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to write preview: {path}")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_model(model: Any) -> dict[int, str]:
    names = {int(index): str(name) for index, name in model.names.items()}
    if model.task != "pose":
        raise ValueError(f"Expected a pose model, got task={model.task!r}")
    if [names[index] for index in sorted(names)] != CLASS_NAMES:
        raise ValueError(f"Unexpected model classes: {names}")
    kpt_shape = getattr(model.model, "kpt_shape", None)
    if list(kpt_shape or []) != [4, 3]:
        raise ValueError(f"Expected kpt_shape [4, 3], got {kpt_shape}")
    return names


def prepare_labelme_review(source: Path, output: Path) -> dict[str, Any]:
    """Create one flat directory that LabelMe can browse continuously."""
    source = source.resolve()
    output = output.resolve()
    manifest_path = output / "manifest.csv"
    destination = output / "labelme_review"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if destination.exists():
        raise FileExistsError(f"Review directory already exists; refusing overwrite: {destination}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    basenames = [Path(row["relative_image"]).name for row in manifest]
    if len(basenames) != len(set(basenames)):
        duplicates = sorted(name for name, count in Counter(basenames).items() if count > 1)
        raise ValueError(f"Cannot flatten duplicate image basenames: {duplicates[:5]}")

    staging = Path(tempfile.mkdtemp(prefix=".labelme_review.", dir=output))
    try:
        for row in manifest:
            relative_image = Path(row["relative_image"])
            source_image = source / relative_image
            candidate_json = output / row["output_json"]
            if not source_image.is_file() or not candidate_json.is_file():
                raise FileNotFoundError(f"Missing source/candidate for {relative_image}")
            annotation = load_annotation(candidate_json)
            annotation["imagePath"] = source_image.name
            annotation["imageData"] = None
            (staging / source_image.with_suffix(".json").name).write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.symlink(source_image, staging / source_image.name)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = {
        "status": "ready",
        "directory": str(destination),
        "samples": len(manifest),
        "images_are_source_symlinks": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def audit_output(source: Path, output: Path) -> dict[str, Any]:
    """Audit a generated review set without importing inference dependencies."""
    source = source.resolve()
    output = output.resolve()
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.csv"
    review_path = output / "review_queue.csv"
    for path in (summary_path, manifest_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required output file not found: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with review_path.open(encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))

    errors: list[str] = []
    added_counts: Counter[str] = Counter()
    unresolved_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    flat_review = output / "labelme_review"
    for row in manifest:
        relative_image = Path(row["relative_image"])
        source_image = source / relative_image
        source_json = source_image.with_suffix(".json")
        output_image = output / "candidates" / relative_image
        output_json = output / row["output_json"]
        prefix = str(relative_image)
        if not source_image.is_file() or not source_json.is_file():
            errors.append(f"{prefix}: source pair missing")
            continue
        if not output_image.is_symlink() or output_image.resolve() != source_image.resolve():
            errors.append(f"{prefix}: image symlink does not resolve to source")
        if not output_json.is_file():
            errors.append(f"{prefix}: output JSON missing")
            continue
        try:
            original = load_annotation(source_json)
            candidate = load_annotation(output_json)
        except Exception as exc:
            errors.append(f"{prefix}: JSON parse failed: {exc}")
            continue
        original_shapes = original["shapes"]
        candidate_shapes = candidate["shapes"]
        if candidate_shapes[: len(original_shapes)] != original_shapes:
            errors.append(f"{prefix}: existing shape prefix changed")
        appended = candidate_shapes[len(original_shapes):]
        existing_labels = set(target_polygon_shapes(original))
        appended_labels: list[str] = []
        for shape in appended:
            label = str(shape.get("label") or "").strip().upper()
            appended_labels.append(label)
            added_counts[label] += 1
            flags = shape.get("flags") or {}
            quality = next(
                (
                    key.removeprefix("quality_")
                    for key, value in flags.items()
                    if key.startswith("quality_") and value
                ),
                "missing",
            )
            quality_counts[quality] += 1
            points = shape.get("points")
            if label not in CLASS_SET or label in existing_labels:
                errors.append(f"{prefix}: invalid or non-missing appended label {label}")
            if shape.get("shape_type") != "polygon" or not isinstance(points, list) or len(points) != 4:
                errors.append(f"{prefix}: appended {label} is not a four-point polygon")
                continue
            if not flags.get("pseudo_label") or not flags.get("needs_review"):
                errors.append(f"{prefix}: appended {label} lacks review flags")
            width = float(candidate.get("imageWidth") or 0)
            height = float(candidate.get("imageHeight") or 0)
            converted = [(float(point[0]), float(point[1])) for point in points]
            if len(set(converted)) != 4:
                errors.append(f"{prefix}: appended {label} has duplicate points")
            if any(not 0 <= x <= width or not 0 <= y <= height for x, y in converted):
                errors.append(f"{prefix}: appended {label} is out of bounds")
        recorded_added = [value for value in row["added_labels"].split(";") if value]
        if appended_labels != recorded_added:
            errors.append(f"{prefix}: manifest added labels differ from JSON")
        unresolved = [value for value in row["unresolved_labels"].split(";") if value]
        for label in unresolved:
            unresolved_counts[label] += 1
        coverage = existing_labels | set(appended_labels) | set(unresolved)
        if coverage != CLASS_SET:
            errors.append(f"{prefix}: class coverage accounting is incomplete")
        if flat_review.is_dir():
            flat_image = flat_review / relative_image.name
            flat_json = flat_review / relative_image.with_suffix(".json").name
            if not flat_image.is_symlink() or flat_image.resolve() != source_image.resolve():
                errors.append(f"{prefix}: flat review image symlink is invalid")
            if not flat_json.is_file():
                errors.append(f"{prefix}: flat review JSON missing")
            else:
                flat_annotation = load_annotation(flat_json)
                if flat_annotation.get("shapes") != candidate_shapes:
                    errors.append(f"{prefix}: flat review shapes differ from candidate JSON")

    expected_counts = summary.get("counts", {})
    if len(manifest) != expected_counts.get("processed_samples"):
        errors.append("Manifest row count differs from summary")
    if len(review) != sum(added_counts.values()) + sum(unresolved_counts.values()):
        errors.append("Review queue row count differs from added+unresolved counts")
    if sum(added_counts.values()) != expected_counts.get("added_shapes"):
        errors.append("Added shape count differs from summary")
    if sum(unresolved_counts.values()) != expected_counts.get("unresolved_shapes"):
        errors.append("Unresolved shape count differs from summary")
    if dict(sorted(quality_counts.items())) != expected_counts.get("quality"):
        errors.append("Quality counts differ from summary")

    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "source": str(source),
        "output": str(output),
        "checks": {
            "manifest_rows": len(manifest),
            "review_rows": len(review),
            "added_shapes": sum(added_counts.values()),
            "unresolved_shapes": sum(unresolved_counts.values()),
            "quality": dict(sorted(quality_counts.items())),
            "existing_shape_prefix_preserved": not any(
                "existing shape prefix changed" in error for error in errors
            ),
            "source_image_symlinks_valid": not any("symlink" in error for error in errors),
            "flat_labelme_review_present": flat_review.is_dir(),
        },
        "errors": errors,
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError(f"Pseudo-label output audit failed with {len(errors)} errors")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if args.prepare_review_only:
        if not output.is_dir():
            raise FileNotFoundError(f"Output directory not found: {output}")
        return prepare_labelme_review(source, output)
    if args.audit_only:
        if not output.is_dir():
            raise FileNotFoundError(f"Output directory not found for audit: {output}")
        return audit_output(source, output)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    pairs, image_only, json_only = discover_samples(source)
    selection = load_selection(args.selection_file)
    if selection is not None:
        available = {str(image.relative_to(source)) for image, _ in pairs}
        missing = sorted(selection - available)
        if missing:
            raise ValueError(f"Selection contains unknown/unpaired images: {missing[:5]}")
        pairs = [pair for pair in pairs if str(pair[0].relative_to(source)) in selection]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No paired samples selected")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = validate_model(model)
    device = resolve_device(args.device)
    model_hash = sha256_file(model_path)
    dry_report = {
        "source": str(source),
        "model": str(model_path),
        "model_sha256": model_hash,
        "selected_pairs": len(pairs),
        "all_pairs": len(discover_samples(source)[0]),
        "image_only": len(image_only),
        "json_only": len(json_only),
        "device": device,
        "imgsz": args.imgsz,
        "predict_conf": args.predict_conf,
        "candidate_conf": args.candidate_conf,
    }
    print(json.dumps(dry_report, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run complete; no inference or files were written. Add --apply to proceed.")
        return dry_report

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    raw_predictions: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()
    added_counts: Counter[str] = Counter()
    unresolved_counts: Counter[str] = Counter()
    preview_count = 0
    try:
        for index, (image_path, json_path) in enumerate(pairs, 1):
            import cv2

            relative_image = image_path.relative_to(source)
            annotation = load_annotation(json_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not decode image: {image_path}")
            height, width = image.shape[:2]
            json_width = int(annotation.get("imageWidth") or 0)
            json_height = int(annotation.get("imageHeight") or 0)
            if (json_width, json_height) != (width, height):
                raise ValueError(
                    f"Image/JSON dimensions differ for {relative_image}: "
                    f"PNG={width}x{height}, JSON={json_width}x{json_height}"
                )

            black_bands = detect_black_bands(image)
            views = plan_views(width, height, black_bands, annotation_points(annotation))
            sources = [image[view.y0:view.y1, view.x0:view.x1] for view in views]
            results = model.predict(
                sources,
                conf=args.predict_conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                max_det=args.max_det,
                verbose=False,
            )
            candidates = clip_candidates(
                (
                    candidate
                    for result, view in zip(results, views)
                    for candidate in prediction_candidates(result, names, view)
                    if candidate.box_conf >= args.candidate_conf
                ),
                width,
                height,
            )
            anchors = anchor_scores(annotation, candidates, views)
            grouped: dict[str, list[Candidate]] = defaultdict(list)
            for candidate in candidates:
                grouped[candidate.label].append(candidate)
            selected_with_clusters: dict[str, tuple[Candidate, list[Candidate]]] = {}
            for label in CLASS_NAMES:
                choice = choose_candidate(grouped.get(label, []), anchors)
                if choice is not None:
                    selected_with_clusters[label] = choice

            output_annotation, added, unresolved = append_missing_shapes(
                annotation, selected_with_clusters, anchors, model_hash
            )
            output_annotation["imagePath"] = image_path.name
            output_annotation["imageData"] = None
            relative_json = relative_image.with_suffix(".json")
            destination_json = staging / "candidates" / relative_json
            destination_image = staging / "candidates" / relative_image
            destination_json.parent.mkdir(parents=True, exist_ok=True)
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_json.write_text(
                json.dumps(output_annotation, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.symlink(image_path, destination_image)

            existing = sorted(target_polygon_shapes(annotation), key=CLASS_NAMES.index)
            for item in added:
                quality_counts[item["quality"]] += 1
                added_counts[item["label"]] += 1
                review_rows.append(
                    {
                        "relative_image": str(relative_image),
                        "label": item["label"],
                        "status": "review_required",
                        **item,
                    }
                )
            for label in unresolved:
                unresolved_counts[label] += 1
                review_rows.append(
                    {
                        "relative_image": str(relative_image),
                        "label": label,
                        "status": "no_candidate",
                        "quality": "unresolved",
                        "box_conf": "",
                        "keypoint_conf": "",
                        "support": 0,
                        "views": "",
                        "anatomy_ok": False,
                    }
                )
            manifest_rows.append(
                {
                    "relative_image": str(relative_image),
                    "source_json": str(json_path),
                    "output_json": str(Path("candidates") / relative_json),
                    "existing_polygon_labels": ";".join(existing),
                    "added_labels": ";".join(item["label"] for item in added),
                    "unresolved_labels": ";".join(unresolved),
                    "high": sum(item["quality"] == "high" for item in added),
                    "medium": sum(item["quality"] == "medium" for item in added),
                    "low": sum(item["quality"] == "low" for item in added),
                    "views": ";".join(view.name for view in views),
                    "black_left_px": black_bands["left"],
                    "black_right_px": black_bands["right"],
                    "black_top_px": black_bands["top"],
                    "black_bottom_px": black_bands["bottom"],
                    "anchor_scores": json.dumps(anchors, sort_keys=True),
                }
            )
            raw_predictions.append(
                {
                    "relative_image": str(relative_image),
                    "views": [view.__dict__ for view in views],
                    "anchor_scores": anchors,
                    "candidates": [
                        {
                            "label": item.label,
                            "box_conf": item.box_conf,
                            "keypoint_conf": item.keypoint_conf,
                            "points": item.points,
                            "box": item.box,
                            "view": item.view,
                        }
                        for item in candidates
                    ],
                }
            )
            if preview_count < args.preview_limit:
                write_preview(
                    image,
                    output_annotation,
                    staging / "previews" / relative_image.with_suffix(".jpg"),
                )
                preview_count += 1
            print(
                f"[{index}/{len(pairs)}] {relative_image}: "
                f"added={len(added)} unresolved={len(unresolved)} views={len(views)}"
            )

        write_csv(
            staging / "manifest.csv",
            manifest_rows,
            [
                "relative_image", "source_json", "output_json",
                "existing_polygon_labels", "added_labels", "unresolved_labels",
                "high", "medium", "low", "views",
                "black_left_px", "black_right_px", "black_top_px", "black_bottom_px",
                "anchor_scores",
            ],
        )
        review_priority = {"no_candidate": 0, "review_required": 1}
        quality_priority = {"unresolved": 0, "low": 1, "medium": 2, "high": 3}
        review_rows.sort(
            key=lambda row: (
                review_priority[row["status"]],
                quality_priority[row["quality"]],
                row["relative_image"],
                CLASS_NAMES.index(row["label"]),
            )
        )
        write_csv(
            staging / "review_queue.csv",
            review_rows,
            [
                "relative_image", "label", "status", "quality", "box_conf",
                "keypoint_conf", "support", "views", "anatomy_ok",
            ],
        )
        write_csv(
            staging / "unpaired_images.csv",
            [{"relative_image": str(path.relative_to(source))} for path in image_only],
            ["relative_image"],
        )
        (staging / "raw_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_predictions),
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "status": "review_required",
            "source": str(source),
            "source_was_modified": False,
            "model": str(model_path),
            "model_sha256": model_hash,
            "classes": CLASS_NAMES,
            "keypoint_order": ["upper_left", "upper_right", "lower_right", "lower_left"],
            "settings": {
                "imgsz": args.imgsz,
                "predict_conf": args.predict_conf,
                "candidate_conf": args.candidate_conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "device": device,
                "views": ["original", "black_trim", "training_aspect"],
                "training_aspect": 0.353,
            },
            "counts": {
                "all_paired_samples": len(discover_samples(source)[0]),
                "processed_samples": len(pairs),
                "source_image_only": len(image_only),
                "source_json_only": len(json_only),
                "added_shapes": sum(added_counts.values()),
                "unresolved_shapes": sum(unresolved_counts.values()),
                "quality": dict(sorted(quality_counts.items())),
                "added_by_label": {label: added_counts[label] for label in CLASS_NAMES},
                "unresolved_by_label": {label: unresolved_counts[label] for label in CLASS_NAMES},
                "previews": preview_count,
            },
            "safety": {
                "existing_shapes_preserved": True,
                "only_missing_target_polygons_appended": True,
                "all_appended_shapes_require_review": True,
                "output_images_are_symlinks_to_read_only_source": True,
                "unpaired_images_excluded": True,
                "output_refuses_overwrite": True,
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# First-batch lateral pseudo-label review set\n\n"
            "Open `candidates/` with LabelMe. Existing shapes are preserved. "
            "Every appended vertebral polygon has `pseudo_label=true` and "
            "`needs_review=true`; inspect, correct, and clear these flags manually.\n\n"
            "Use `review_queue.csv` from low/unresolved to high. Images are symlinks "
            "to the read-only source on the external volume.\n",
            encoding="utf-8",
        )
        staging.rename(output)
        prepare_labelme_review(source, output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=project_root / DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=project_root / DEFAULT_OUTPUT)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--predict-conf", type=float, default=0.01)
    parser.add_argument("--candidate-conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit an existing output without loading the model or running inference",
    )
    parser.add_argument(
        "--prepare-review-only",
        action="store_true",
        help="Build a flat LabelMe browsing directory from an existing output",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.imgsz <= 0 or args.max_det <= 0 or args.preview_limit < 0:
        parser.error("imgsz/max-det must be positive and preview-limit non-negative")
    if not 0 <= args.predict_conf <= args.candidate_conf <= 1:
        parser.error("Require 0 <= predict-conf <= candidate-conf <= 1")
    modes = sum((args.apply, args.audit_only, args.prepare_review_only))
    if modes > 1:
        parser.error("--apply, --audit-only and --prepare-review-only are mutually exclusive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
