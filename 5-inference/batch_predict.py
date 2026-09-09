#!/usr/bin/env python3
"""Run the final 20-class lateral-spine and three-keypoint pelvis models together."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPINE_MODEL = PROJECT_ROOT / "3-model_training/runs/pose/yolo11l_lateral_20cls_scratch_best/weights/best.pt"
DEFAULT_PELVIS_MODEL = PROJECT_ROOT / "4-model_training_CFH/runs/yolo11l_pelvis_3kpt_roi_mixed_best/weights/best.pt"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets/yolo_lateral_reviewed_combined_20cls/images/test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets/lateral_combined_predictions"

SPINE_NAMES = [
    "C2", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13",
    "L1", "L2", "L3", "L4", "L5",
]
PELVIS_KEYPOINT_NAMES = ["CFH", "S1_left", "S1_right"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PELVIS_POINT_COLORS = [(255, 80, 210), (40, 210, 255), (50, 150, 255)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_images(image_dir: Path, recursive: bool = True) -> list[Path]:
    iterator = image_dir.rglob("*") if recursive else image_dir.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def tensor_rows(result: Any, expected_keypoints: int) -> list[dict[str, Any]]:
    if result.boxes is None or result.keypoints is None:
        return []
    classes = result.boxes.cls.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    keypoints = result.keypoints.data.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for class_id, score, bbox, points in zip(classes, scores, boxes, keypoints):
        if len(points) != expected_keypoints:
            raise ValueError(f"Expected {expected_keypoints} keypoints, found {len(points)}")
        rows.append({
            "class_id": int(class_id),
            "confidence": float(score),
            "bbox_xyxy": [float(value) for value in bbox],
            "keypoints": [[float(value) for value in point] for point in points],
        })
    return rows


def select_spine_predictions(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep exactly the highest-confidence prediction for each vertebra class."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        if not 0 <= class_id < len(SPINE_NAMES):
            raise ValueError(f"Invalid spine class id: {class_id}")
        grouped[class_id].append(prediction)
    return [
        max(grouped[class_id], key=lambda item: float(item["confidence"]))
        for class_id in sorted(grouped)
    ]


def select_pelvis_prediction(predictions: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Keep the highest-confidence pelvis instance."""
    if not predictions:
        return None
    return max(predictions, key=lambda item: float(item["confidence"]))


def model_names(model: Any) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in range(len(names))]
    return [str(name) for name in names]


def validate_model_contract(model: Any, expected_names: Sequence[str], expected_keypoints: int, label: str) -> None:
    names = model_names(model)
    keypoint_shape = list(model.model.yaml.get("kpt_shape", []))
    if model.task != "pose" or names != list(expected_names) or keypoint_shape != [expected_keypoints, 3]:
        raise ValueError(
            f"Unexpected {label} model contract: task={model.task}, names={names}, kpt_shape={keypoint_shape}"
        )


def _point_visible(point: Sequence[float], threshold: float) -> bool:
    return len(point) < 3 or float(point[2]) >= threshold


def _draw_label(image: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = anchor
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    y = max(height + 8, y)
    cv2.rectangle(image, (x, y - height - 7), (x + width + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def draw_spine(image: np.ndarray, predictions: Sequence[dict[str, Any]], keypoint_confidence: float) -> np.ndarray:
    output = image.copy()
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        color = (55, 210, 105)
        visible = [point for point in prediction["keypoints"] if _point_visible(point, keypoint_confidence)]
        if len(visible) == 4:
            polygon = np.asarray([[round(point[0]), round(point[1])] for point in visible], dtype=np.int32)
            cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
        for point in visible:
            cv2.circle(output, (round(point[0]), round(point[1])), 4, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = [round(value) for value in prediction["bbox_xyxy"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        _draw_label(output, f"{SPINE_NAMES[class_id]} {prediction['confidence']:.2f}", (x1, y1), color)
    return output


def draw_pelvis(image: np.ndarray, prediction: dict[str, Any] | None, keypoint_confidence: float) -> np.ndarray:
    output = image.copy()
    if prediction is None:
        return output
    x1, y1, x2, y2 = [round(value) for value in prediction["bbox_xyxy"]]
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 210), 2, cv2.LINE_AA)
    _draw_label(output, f"pelvis {prediction['confidence']:.2f}", (x1, y1), (255, 100, 210))
    for index, point in enumerate(prediction["keypoints"]):
        if not _point_visible(point, keypoint_confidence):
            continue
        x, y = round(point[0]), round(point[1])
        color = PELVIS_POINT_COLORS[index]
        cv2.circle(output, (x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(output, (x, y), 5, color, -1, cv2.LINE_AA)
        _draw_label(output, PELVIS_KEYPOINT_NAMES[index], (x + 8, y), color)
    return output


def serialize_record(
    source: Path,
    relative: Path,
    width: int,
    height: int,
    spine: Sequence[dict[str, Any]],
    pelvis: dict[str, Any] | None,
) -> dict[str, Any]:
    spine_rows = []
    for prediction in spine:
        row = dict(prediction)
        row["class_name"] = SPINE_NAMES[int(row["class_id"])]
        spine_rows.append(row)
    pelvis_row = None
    if pelvis is not None:
        pelvis_row = dict(pelvis)
        pelvis_row["class_name"] = "pelvis"
        pelvis_row["named_keypoints"] = {
            name: point for name, point in zip(PELVIS_KEYPOINT_NAMES, pelvis["keypoints"])
        }
    return {
        "source_image": str(source.resolve()),
        "relative_image": relative.as_posix(),
        "width": width,
        "height": height,
        "spine_predictions": spine_rows,
        "pelvis_prediction": pelvis_row,
    }


def build_gallery(records: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards = []
    for record in records:
        relative = html.escape(record["relative_image"], quote=True)
        preview = html.escape(record["combined_preview"], quote=True)
        spine_count = len(record["spine_predictions"])
        pelvis = record["pelvis_prediction"]
        pelvis_text = "漏检" if pelvis is None else f"{float(pelvis['confidence']):.3f}"
        cards.append(
            f'<article><a href="{preview}"><img loading="lazy" src="{preview}"></a>'
            f'<div><b>{relative}</b><span>脊柱 {spine_count}/20 · pelvis {pelvis_text}</span></div></article>'
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>20类脊柱 + 三点 pelvis 联合推理</title><style>
body{{margin:0;background:#0d1118;color:#edf2f8;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}header{{padding:18px 22px;background:#111824;position:sticky;top:0}}h1{{margin:0 0 6px;font-size:23px}}header p{{margin:0;color:#aab8ca}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;padding:14px}}article{{background:#151d29;border:1px solid #2c394d;border-radius:9px;overflow:hidden}}img{{display:block;width:100%;height:auto}}article div{{display:grid;gap:5px;padding:10px 12px}}b{{font-size:13px;word-break:break-all}}span{{font-size:13px;color:#aab8ca}}
</style></head><body><header><h1>20类整脊柱 + pelvis 三关键点联合推理</h1><p>{summary['processed_images']}张 · 每个脊柱类别只保留最高置信度 · pelvis只保留最高置信度实例</p></header><main>{''.join(cards)}</main></body></html>"""


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    for path in (args.spine_model, args.pelvis_model, args.image_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    images = discover_images(args.image_dir, recursive=not args.no_recursive)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"No supported images found in {args.image_dir}")

    spine_model = YOLO(str(args.spine_model))
    pelvis_model = YOLO(str(args.pelvis_model))
    validate_model_contract(spine_model, SPINE_NAMES, 4, "spine")
    validate_model_contract(pelvis_model, ["pelvis"], 3, "pelvis")

    output_dirs = {
        "spine": args.output_dir / "spine_only",
        "pelvis": args.output_dir / "pelvis_only",
        "combined": args.output_dir / "combined",
        "json": args.output_dir / "json",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, 1):
        relative = image_path.relative_to(args.image_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        height, width = image.shape[:2]
        common = {
            "source": str(image_path),
            "imgsz": args.imgsz,
            "conf": args.confidence,
            "iou": args.iou,
            "device": args.device,
            "verbose": False,
        }
        spine_raw = tensor_rows(spine_model.predict(max_det=40, **common)[0], expected_keypoints=4)
        pelvis_raw = tensor_rows(pelvis_model.predict(max_det=5, **common)[0], expected_keypoints=3)
        spine = select_spine_predictions(spine_raw)
        pelvis = select_pelvis_prediction(pelvis_raw)

        output_relative = relative.with_suffix(".jpg")
        json_relative = relative.with_suffix(".json")
        destinations = {name: directory / output_relative for name, directory in output_dirs.items() if name != "json"}
        for destination in [*destinations.values(), output_dirs["json"] / json_relative]:
            destination.parent.mkdir(parents=True, exist_ok=True)

        spine_image = draw_spine(image, spine, args.keypoint_confidence)
        pelvis_image = draw_pelvis(image, pelvis, args.keypoint_confidence)
        combined_image = draw_pelvis(spine_image, pelvis, args.keypoint_confidence)
        for destination, rendered in (
            (destinations["spine"], spine_image),
            (destinations["pelvis"], pelvis_image),
            (destinations["combined"], combined_image),
        ):
            if not cv2.imwrite(str(destination), rendered):
                raise OSError(f"Unable to write image: {destination}")

        record = serialize_record(image_path, relative, width, height, spine, pelvis)
        record["combined_preview"] = (Path("combined") / output_relative).as_posix()
        (output_dirs["json"] / json_relative).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
        print(f"Processed {index}/{len(images)}: {relative}", flush=True)

    summary = {
        "processed_images": len(records),
        "images_with_complete_20_class_spine": sum(len(row["spine_predictions"]) == 20 for row in records),
        "images_with_pelvis": sum(row["pelvis_prediction"] is not None for row in records),
        "spine_model": str(args.spine_model.resolve()),
        "spine_model_sha256": sha256_file(args.spine_model),
        "pelvis_model": str(args.pelvis_model.resolve()),
        "pelvis_model_sha256": sha256_file(args.pelvis_model),
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "iou": args.iou,
            "keypoint_confidence": args.keypoint_confidence,
            "device": args.device,
            "selection": "highest-confidence prediction per spine class and highest-confidence pelvis instance",
        },
    }
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "prediction_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(build_gallery(records, summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine-model", "--corner-model", dest="spine_model", type=Path, default=DEFAULT_SPINE_MODEL)
    parser.add_argument("--pelvis-model", "--cfh-model", dest="pelvis_model", type=Path, default=DEFAULT_PELVIS_MODEL)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", "--num-samples", dest="limit", type=int, default=0, help="0 means all images")
    parser.add_argument("--confidence", "--conf", dest="confidence", type=float, default=0.10)
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    for name in ("confidence", "keypoint_confidence", "iou"):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    summary = run_inference(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
