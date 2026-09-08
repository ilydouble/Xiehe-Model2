#!/usr/bin/env python3
"""Audit which raw LabelMe JSON produced each yolo_corner label file.

The legacy converter selected one JSON per patient directory and applied it to
every PNG in that directory.  This audit recreates every possible conversion
and identifies whether each YOLO label matches its same-stem JSON or another
image's JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/yolo_corner"
DEFAULT_RAW = Path("/Volumes/E/spine_data/LAT202511")
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/yolo_corner_label_source_audit"
SPLITS = ("train", "val", "test")
LABELS = ["C7", "L1", "L2", "L3", "L4", "L5", *(f"T{i}" for i in range(1, 13))]
LABEL_MAP = {label: index for index, label in enumerate(LABELS)}


def load_converter() -> Any:
    path = PROJECT_ROOT / "2-build_dataset/convert_to_yolo_pose.py"
    spec = importlib.util.spec_from_file_location("legacy_yolo_pose_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONVERTER = load_converter()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lines_by_class(lines: Sequence[str], source: Path | str) -> dict[int, str]:
    result: dict[int, str] = {}
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 17:
            raise ValueError(f"{source}:{line_number}: expected 17 fields, got {len(fields)}")
        class_id = int(fields[0])
        if class_id in result:
            raise ValueError(f"{source}:{line_number}: duplicate class {class_id}")
        result[class_id] = " ".join(fields)
    return result


def converted_signature(json_path: Path, width: int, height: int) -> dict[int, str]:
    annotations = CONVERTER.parse_json_annotation(json_path)
    lines = CONVERTER.convert_to_yolo_format(annotations, width, height, LABEL_MAP)
    return lines_by_class(lines, json_path)


def signature_distance(actual: dict[int, str], candidate: dict[int, str]) -> tuple[int, float]:
    class_difference = len(set(actual) ^ set(candidate))
    max_delta = 0.0
    for class_id in set(actual) & set(candidate):
        actual_fields = actual[class_id].split()
        candidate_fields = candidate[class_id].split()
        max_delta = max(
            max_delta,
            *(abs(float(a) - float(b)) for a, b in zip(actual_fields[1:], candidate_fields[1:])),
        )
    return class_difference, max_delta


def discover_dataset(dataset: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        for image_path in sorted(path for path in image_dir.glob("*.png") if not path.name.startswith("._")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            records.append({"split": split, "image": image_path, "label": label_path})
    return records


def raw_image_index(raw_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in raw_root.rglob("*.png"):
        if not path.name.startswith("._"):
            result[path.name].append(path)
    return dict(result)


def audit(dataset: Path, raw_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_by_name = raw_image_index(raw_root)
    rows: list[dict[str, Any]] = []
    for record in discover_dataset(dataset):
        image_path, label_path = record["image"], record["label"]
        raw_matches = raw_by_name.get(image_path.name, [])
        row: dict[str, Any] = {
            "split": record["split"],
            "filename": image_path.name,
            "relative_image": str(image_path.relative_to(dataset)),
            "status": "",
            "patient_id": "",
            "same_stem_json_exists": False,
            "matched_jsons": "",
            "best_json": "",
            "best_class_set_difference": "",
            "best_max_coordinate_delta_normalized": "",
            "dataset_raw_image_sha256_equal": False,
            "requires_full_23cls_reannotation": False,
        }
        if len(raw_matches) != 1:
            row["status"] = "raw_image_missing" if not raw_matches else "raw_image_ambiguous"
            rows.append(row)
            continue
        raw_image = raw_matches[0]
        patient_dir = raw_image.parent
        row["patient_id"] = patient_dir.name
        row["dataset_raw_image_sha256_equal"] = sha256_file(image_path) == sha256_file(raw_image)
        same_stem_json = raw_image.with_suffix(".json")
        row["same_stem_json_exists"] = same_stem_json.is_file()
        json_paths = sorted(
            path for path in patient_dir.glob("*.json") if not path.name.startswith("._")
        )
        if not json_paths:
            row["status"] = "no_json_in_patient_directory"
            row["requires_full_23cls_reannotation"] = True
            rows.append(row)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        actual = lines_by_class(label_path.read_text(encoding="utf-8").splitlines(), label_path)
        comparisons = []
        exact_matches = []
        for json_path in json_paths:
            candidate = converted_signature(json_path, width, height)
            class_difference, max_delta = signature_distance(actual, candidate)
            exact = actual == candidate
            comparisons.append((class_difference, max_delta, json_path))
            if exact:
                exact_matches.append(json_path)
        comparisons.sort(key=lambda item: (item[0], item[1], item[2].name))
        best_class_difference, best_delta, best_json = comparisons[0]
        row["best_json"] = best_json.name
        row["best_class_set_difference"] = best_class_difference
        row["best_max_coordinate_delta_normalized"] = f"{best_delta:.9f}"
        row["matched_jsons"] = ";".join(path.name for path in exact_matches)
        if same_stem_json in exact_matches:
            row["status"] = "correct_same_stem"
        elif exact_matches:
            row["status"] = "definite_mismatch"
            row["requires_full_23cls_reannotation"] = True
        else:
            row["status"] = "unresolved_no_exact_source"
            row["requires_full_23cls_reannotation"] = True
        rows.append(row)

    status = Counter(str(row["status"]) for row in rows)
    summary = {
        "schema_version": 1,
        "dataset": str(dataset.resolve()),
        "raw_root": str(raw_root.resolve()),
        "sample_count": len(rows),
        "status": dict(sorted(status.items())),
        "requires_full_23cls_reannotation_count": sum(
            bool(row["requires_full_23cls_reannotation"]) for row in rows
        ),
        "dataset_raw_image_sha256_mismatch_count": sum(
            not bool(row["dataset_raw_image_sha256_equal"]) for row in rows
        ),
        "definite_mismatch_by_split": dict(sorted(Counter(
            str(row["split"]) for row in rows if row["status"] == "definite_mismatch"
        ).items())),
    }
    return summary, rows


def write_outputs(output: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0]) if rows else []
    with (output / "per_image_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [row for row in rows if row["requires_full_23cls_reannotation"]]
    lines = [
        "# yolo_corner旧18类标注来源审计",
        "",
        f"- 样本总数：{summary['sample_count']}",
        f"- 需要补全23类：{len(mismatches)}",
        f"- 状态计数：`{json.dumps(summary['status'], ensure_ascii=False)}`",
        "",
        "## 需要补全23类的图像",
        "",
    ]
    for row in mismatches:
        lines.append(
            f"- `{row['relative_image']}`：{row['status']}；现有标签来源 `{row['matched_jsons'] or row['best_json']}`"
        )
    (output / "full_23cls_reannotation_list.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, rows = audit(args.dataset, args.raw_root)
    write_outputs(args.output, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
