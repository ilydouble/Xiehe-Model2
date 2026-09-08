#!/usr/bin/env python3
"""Render 23-class spine and three-keypoint pelvis predictions on one test split.

The two YOLO Pose models are run independently on the original pelvis test images.
Spine predictions are shown as vertebral quadrilaterals.  Pelvis predictions are
shown together with the three-keypoint ground truth so that CFH and S1 endpoint
errors can be inspected directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SPINE_NAMES = [
    "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
]
PELVIS_KEYPOINT_NAMES = ["CFH", "S1_left", "S1_right"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pelvis_gt(label_path: Path, width: int, height: int) -> dict[str, Any]:
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label_path}: expected one pelvis object, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 14:
        raise ValueError(f"{label_path}: expected 14 fields, found {len(fields)}")
    values = [float(value) for value in fields]
    if int(values[0]) != 0:
        raise ValueError(f"{label_path}: expected pelvis class 0")
    if any(value < 0 or value > 1 for value in values[1:5]):
        raise ValueError(f"{label_path}: bbox is outside [0,1]")
    cx, cy, bw, bh = values[1:5]
    bbox = [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ]
    keypoints = []
    for offset in range(5, 14, 3):
        x, y, visible = values[offset:offset + 3]
        if x < 0 or x > 1 or y < 0 or y > 1 or visible <= 0:
            raise ValueError(f"{label_path}: invalid visible keypoint")
        keypoints.append([x * width, y * height, visible])
    return {"bbox": bbox, "keypoints": keypoints}


def select_spine_predictions(
    predictions: Sequence[dict[str, Any]], class_count: int = 23
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"invalid spine class id: {class_id}")
        grouped[class_id].append(prediction)
    selected = []
    duplicate_counts: dict[str, int] = {}
    for class_id in sorted(grouped):
        ranked = sorted(grouped[class_id], key=lambda item: float(item["score"]), reverse=True)
        selected.append(ranked[0])
        if len(ranked) > 1:
            duplicate_counts[SPINE_NAMES[class_id]] = len(ranked) - 1
    missing = [SPINE_NAMES[class_id] for class_id in range(class_count) if class_id not in grouped]
    return selected, missing, duplicate_counts


def select_pelvis_prediction(
    predictions: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    if not predictions:
        return None, 0
    ranked = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    return ranked[0], len(ranked) - 1


def point_errors(
    predicted: dict[str, Any] | None,
    ground_truth: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[float | None], list[float | None]]:
    if predicted is None:
        return [None, None, None], [None, None, None]
    pred_points = predicted["keypoints"]
    if len(pred_points) != 3:
        raise ValueError(f"expected 3 pelvis keypoints, found {len(pred_points)}")
    diagonal = math.hypot(width, height)
    pixels: list[float | None] = []
    relative: list[float | None] = []
    for pred, truth in zip(pred_points, ground_truth["keypoints"]):
        error = math.hypot(float(pred[0]) - float(truth[0]), float(pred[1]) - float(truth[1]))
        pixels.append(error)
        relative.append(error / diagonal * 100)
    return pixels, relative


def risk_tags(
    missing_spine: Sequence[str],
    duplicate_spine: dict[str, int],
    pelvis_prediction: dict[str, Any] | None,
    pelvis_extra: int,
    relative_errors: Sequence[float | None],
) -> tuple[list[str], float]:
    tags: list[str] = []
    score = 0.0
    if missing_spine:
        tags.append(f"脊柱缺{len(missing_spine)}类")
        score += 20 * len(missing_spine)
    duplicate_total = sum(duplicate_spine.values())
    if duplicate_total:
        tags.append(f"脊柱重复候选{duplicate_total}")
        score += 8 * duplicate_total
    if pelvis_prediction is None:
        tags.append("三点模型漏检")
        score += 200
    else:
        pelvis_score = float(pelvis_prediction["score"])
        if pelvis_score < 0.25:
            tags.append("三点低置信")
            score += (0.25 - pelvis_score) * 200
    if pelvis_extra:
        tags.append(f"pelvis额外候选{pelvis_extra}")
        score += 10 * pelvis_extra
    numeric_errors = [float(value) for value in relative_errors if value is not None]
    if numeric_errors:
        max_error = max(numeric_errors)
        score += max_error * 10
        if max_error >= 2.0:
            tags.append("三点误差≥2%对角线")
        elif max_error >= 1.0:
            tags.append("三点误差≥1%对角线")
    if not tags:
        tags.append("常规")
    return tags, score


def _tensor_rows(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or result.keypoints is None:
        return []
    classes = result.boxes.cls.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    keypoints = result.keypoints.data.detach().cpu().tolist()
    rows = []
    for class_id, score, bbox, points in zip(classes, scores, boxes, keypoints):
        rows.append({
            "class_id": int(class_id),
            "score": float(score),
            "bbox": [[float(value) for value in bbox]][0],
            "keypoints": [[float(value) for value in point] for point in points],
        })
    return rows


def _font(size: int):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _expand_crop(
    points: Iterable[Sequence[float]], width: int, height: int, fallback: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    pts = [(float(point[0]), float(point[1])) for point in points]
    if not pts:
        return fallback
    xs = [point[0] for point in pts]
    ys = [point[1] for point in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span_x = max(x1 - x0, width * 0.10)
    span_y = max(y1 - y0, height * 0.08)
    margin_x = max(span_x * 0.65, width * 0.03)
    margin_y = max(span_y * 0.55, height * 0.025)
    left = max(0, int(math.floor(x0 - margin_x)))
    top = max(0, int(math.floor(y0 - margin_y)))
    right = min(width, int(math.ceil(x1 + margin_x)))
    bottom = min(height, int(math.ceil(y1 + margin_y)))
    return left, top, max(left + 2, right), max(top + 2, bottom)


def _draw_text(draw: Any, xy: tuple[float, float], text: str, font: Any, fill: tuple[int, int, int], background=(0, 0, 0)) -> None:
    box = draw.textbbox(xy, text, font=font, stroke_width=0)
    padded = (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2)
    draw.rectangle(padded, fill=background)
    draw.text(xy, text, font=font, fill=fill)


def _render_panel(
    canvas: Any,
    source: Any,
    panel_box: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
    spine: Sequence[dict[str, Any]],
    pelvis: dict[str, Any] | None,
    gt: dict[str, Any],
    title: str,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = panel_box
    panel_width, panel_height = x1 - x0, y1 - y0
    left, top, right, bottom = crop
    cropped = source.crop(crop)
    scale = min(panel_width / cropped.width, panel_height / cropped.height)
    resized = cropped.resize((max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))))
    offset_x = x0 + (panel_width - resized.width) // 2
    offset_y = y0 + (panel_height - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    draw.rectangle(panel_box, outline=(90, 105, 125), width=2)
    _draw_text(draw, (x0 + 8, y0 + 7), title, _font(22), (235, 240, 247), (20, 26, 36))

    def mapped(point: Sequence[float]) -> tuple[float, float]:
        return offset_x + (float(point[0]) - left) * scale, offset_y + (float(point[1]) - top) * scale

    spine_color = (35, 235, 125)
    label_font = _font(18 if panel_width > 800 else 15)
    line_width = 3 if panel_width > 800 else 2
    for prediction in spine:
        points = [mapped(point) for point in prediction["keypoints"]]
        if all(x0 - 20 <= x <= x1 + 20 and y0 - 20 <= y <= y1 + 20 for x, y in points):
            draw.line(points + [points[0]], fill=spine_color, width=line_width, joint="curve")
            name = SPINE_NAMES[int(prediction["class_id"])]
            _draw_text(draw, points[0], f"{name} {float(prediction['score']):.2f}", label_font, spine_color)

    gt_points = [mapped(point) for point in gt["keypoints"]]
    if len(gt_points) == 3:
        draw.line([gt_points[1], gt_points[2]], fill=(245, 245, 245), width=3)
        for index, (px, py) in enumerate(gt_points):
            radius = 8
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(255, 255, 255), width=3)
            draw.line((px - radius, py, px + radius, py), fill=(255, 255, 255), width=2)
            draw.line((px, py - radius, px, py + radius), fill=(255, 255, 255), width=2)
            _draw_text(draw, (px + 10, py - 20), f"GT {PELVIS_KEYPOINT_NAMES[index]}", label_font, (255, 255, 255))

    if pelvis is not None:
        colors = [(255, 220, 40), (30, 220, 255), (255, 80, 210)]
        pred_points = [mapped(point) for point in pelvis["keypoints"]]
        if len(pred_points) == 3:
            draw.line([pred_points[1], pred_points[2]], fill=(255, 150, 35), width=4)
            for index, (px, py) in enumerate(pred_points):
                radius = 7
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colors[index], outline=(0, 0, 0), width=2)
                _draw_text(draw, (px + 9, py + 4), f"P {PELVIS_KEYPOINT_NAMES[index]}", label_font, colors[index])


def render_preview(
    image_path: Path,
    output_path: Path,
    spine: Sequence[dict[str, Any]],
    pelvis: dict[str, Any] | None,
    gt: dict[str, Any],
    record: dict[str, Any],
) -> None:
    from PIL import Image, ImageDraw

    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    canvas = Image.new("RGB", (1800, 1320), (15, 19, 27))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    detail_font = _font(20)
    filename = record["file_name"]
    draw.text((24, 18), filename, font=title_font, fill=(238, 243, 250))
    missing = ",".join(record["missing_spine"]) if record["missing_spine"] else "无"
    pelvis_score = "漏检" if pelvis is None else f"{float(pelvis['score']):.3f}"
    error_text = ", ".join(
        f"{name}={value:.1f}px" if value is not None else f"{name}=NA"
        for name, value in zip(PELVIS_KEYPOINT_NAMES, record["pelvis_error_px"])
    )
    draw.text(
        (24, 61),
        f"批次 {record['batch']} | 23类已检 {len(spine)}/23 | 缺类 {missing} | pelvis置信 {pelvis_score}",
        font=detail_font,
        fill=(90, 235, 160) if not record["missing_spine"] and pelvis is not None else (255, 185, 75),
    )
    draw.text((24, 94), f"三点误差: {error_text} | 风险: {' / '.join(record['risk_tags'])}", font=detail_font, fill=(215, 220, 230))
    draw.text((24, 124), "绿色=23类预测；彩色实心点=P三点预测；白圈/十字=三点GT", font=detail_font, fill=(185, 195, 210))

    cervical_points = [point for item in spine if int(item["class_id"]) <= 5 for point in item["keypoints"]]
    pelvis_points = [point for point in gt["keypoints"]]
    if pelvis is not None:
        pelvis_points.extend(pelvis["keypoints"])
    cervical_crop = _expand_crop(cervical_points, width, height, (0, 0, width, max(2, height // 3)))
    pelvis_crop = _expand_crop(pelvis_points, width, height, (0, height * 2 // 3, width, height))
    _render_panel(canvas, source, (20, 165, 720, 1300), (0, 0, width, height), spine, pelvis, gt, "全图联合预测")
    _render_panel(canvas, source, (740, 165, 1780, 720), cervical_crop, spine, pelvis, gt, "颈椎放大 C2-C7")
    _render_panel(canvas, source, (740, 740, 1780, 1300), pelvis_crop, spine, pelvis, gt, "骨盆放大：预测 vs GT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=91, optimize=True)


def build_html(records: Sequence[dict[str, Any]], report: dict[str, Any]) -> str:
    cards = []
    for index, record in enumerate(records):
        tags = " ".join(record["risk_tags"])
        searchable = html.escape((record["file_name"] + " " + tags).lower(), quote=True)
        preview = html.escape(record["preview"], quote=True)
        cards.append(
            f'<article class="card" data-batch="{html.escape(record["batch"])}" '
            f'data-risk="{html.escape("risk" if record["risk_tags"] != ["常规"] else "normal")}" '
            f'data-search="{searchable}">'
            f'<a href="{preview}" target="_blank"><img loading="lazy" src="{preview}"></a>'
            f'<div class="meta"><b>{index + 1}/{len(records)}　{html.escape(record["file_name"])}</b>'
            f'<span>批次 {html.escape(record["batch"])} · 23类 {record["spine_selected_count"]}/23 · '
            f'pelvis {html.escape(record["pelvis_score_text"])} · {html.escape(tags)}</span></div></article>'
        )
    summary = report["summary"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>23类脊柱 + 三点骨盆 test 联合预测</title>
<style>
body{{margin:0;background:#0d1118;color:#e8edf5;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#111824ee;padding:18px 24px;border-bottom:1px solid #283244}}
h1{{margin:0 0 8px;font-size:24px}} .summary{{color:#9fb0c5;margin-bottom:12px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap}}button,input{{background:#182235;color:#e8edf5;border:1px solid #35425a;border-radius:7px;padding:8px 12px}}
button.active{{background:#285c48;border-color:#58c695}}input{{min-width:300px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px;padding:16px}}
.card{{background:#151c28;border:1px solid #273246;border-radius:9px;overflow:hidden}}.card img{{display:block;width:100%;height:auto}}
.meta{{padding:10px 12px;display:grid;gap:5px}}.meta b{{font-size:13px;word-break:break-all}}.meta span{{color:#a8b7ca;font-size:13px}}
.hidden{{display:none}}
</style></head><body>
<header><h1>23类脊柱 + 三点骨盆：共同 test 联合预测</h1>
<div class="summary">{summary['images']}张（旧批{summary['old']} / 新批{summary['new']}） · 风险优先排序 · 三点漏检{summary['pelvis_missing']} · 23类不全{summary['spine_incomplete']} · 置信阈值{report['settings']['confidence']}</div>
<div class="controls"><button class="active" data-filter="all">全部</button><button data-filter="risk">仅风险</button><button data-filter="normal">仅常规</button><button data-filter="old">旧批</button><button data-filter="new">新批</button><input id="search" placeholder="搜索文件名或风险"></div></header>
<main>{''.join(cards)}</main>
<script>
const cards=[...document.querySelectorAll('.card')],buttons=[...document.querySelectorAll('button')],search=document.querySelector('#search');let filter='all';
function apply(){{const q=search.value.trim().toLowerCase();cards.forEach(c=>{{let ok=!q||c.dataset.search.includes(q);if(filter==='risk'||filter==='normal')ok=ok&&c.dataset.risk===filter;else if(filter==='old'||filter==='new')ok=ok&&c.dataset.batch===filter;c.classList.toggle('hidden',!ok)}})}}
buttons.forEach(b=>b.onclick=()=>{{buttons.forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.filter;apply()}});search.oninput=apply;
</script></body></html>"""


def audit_package(output_dir: Path, expected: int | None = None) -> dict[str, Any]:
    from PIL import Image

    errors: list[str] = []
    index_path = output_dir / "index.csv"
    if not index_path.exists():
        return {"status": "failed", "errors": ["missing index.csv"]}
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if expected is not None and len(rows) != expected:
        errors.append(f"expected {expected} index rows, found {len(rows)}")
    previews = set()
    for row in rows:
        relative = row.get("preview", "")
        if not relative or relative in previews:
            errors.append(f"missing/duplicate preview field: {relative}")
            continue
        previews.add(relative)
        path = output_dir / relative
        if not path.exists():
            errors.append(f"missing preview: {relative}")
            continue
        try:
            with Image.open(path) as image:
                if image.size != (1800, 1320):
                    errors.append(f"wrong preview size {image.size}: {relative}")
                image.verify()
        except Exception as exc:
            errors.append(f"unreadable preview {relative}: {exc}")
    jsonl_path = output_dir / "predictions.jsonl"
    if not jsonl_path.exists():
        errors.append("missing predictions.jsonl")
        prediction_rows = 0
    else:
        prediction_rows = sum(1 for line in jsonl_path.open() if line.strip())
        if prediction_rows != len(rows):
            errors.append(f"prediction/index mismatch: {prediction_rows} != {len(rows)}")
    if not (output_dir / "index.html").exists():
        errors.append("missing index.html")
    apple_double = [path for path in output_dir.rglob("._*")]
    if apple_double:
        errors.append(f"AppleDouble files: {len(apple_double)}")
    result = {
        "status": "passed" if not errors else "failed",
        "counts": {"index_rows": len(rows), "previews": len(previews), "prediction_rows": prediction_rows},
        "errors": errors,
    }
    return result


def _write_index(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "rank", "file_name", "batch", "patient_id", "cfh_source", "preview",
        "spine_selected_count", "missing_spine", "duplicate_spine", "pelvis_score",
        "pelvis_extra_candidates", "cfh_error_px", "s1_left_error_px", "s1_right_error_px",
        "max_error_percent_diagonal", "risk_tags", "risk_score",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            errors = record["pelvis_error_px"]
            relative = [value for value in record["pelvis_error_percent_diagonal"] if value is not None]
            writer.writerow({
                "rank": rank,
                "file_name": record["file_name"],
                "batch": record["batch"],
                "patient_id": record["patient_id"],
                "cfh_source": record["cfh_source"],
                "preview": record["preview"],
                "spine_selected_count": record["spine_selected_count"],
                "missing_spine": ";".join(record["missing_spine"]),
                "duplicate_spine": json.dumps(record["duplicate_spine"], ensure_ascii=False, sort_keys=True),
                "pelvis_score": "" if record["pelvis_prediction"] is None else f"{float(record['pelvis_prediction']['score']):.8f}",
                "pelvis_extra_candidates": record["pelvis_extra_candidates"],
                "cfh_error_px": "" if errors[0] is None else f"{errors[0]:.4f}",
                "s1_left_error_px": "" if errors[1] is None else f"{errors[1]:.4f}",
                "s1_right_error_px": "" if errors[2] is None else f"{errors[2]:.4f}",
                "max_error_percent_diagonal": "" if not relative else f"{max(relative):.6f}",
                "risk_tags": ";".join(record["risk_tags"]),
                "risk_score": f"{record['risk_score']:.6f}",
            })


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image
    from ultralytics import YOLO

    for path in (args.dataset, args.spine_model, args.pelvis_model):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {args.output}")
    manifest_path = args.dataset / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    manifest.sort(key=lambda row: row["image"])
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise ValueError("No test images found in manifest")

    spine_model = YOLO(str(args.spine_model))
    pelvis_model = YOLO(str(args.pelvis_model))
    spine_names = [spine_model.names[index] for index in range(len(spine_model.names))]
    if spine_model.task != "pose" or spine_names != SPINE_NAMES or spine_model.model.yaml.get("kpt_shape") != [4, 3]:
        raise ValueError(f"Unexpected spine model contract: {spine_names}, {spine_model.model.yaml.get('kpt_shape')}")
    pelvis_names = [pelvis_model.names[index] for index in range(len(pelvis_model.names))]
    if pelvis_model.task != "pose" or pelvis_names != ["pelvis"] or pelvis_model.model.yaml.get("kpt_shape") != [3, 3]:
        raise ValueError(f"Unexpected pelvis model contract: {pelvis_names}, {pelvis_model.model.yaml.get('kpt_shape')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    records: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(manifest, 1):
            image_path = args.dataset / row["image"]
            label_path = args.dataset / row["label"]
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(f"missing image/label pair: {image_path}, {label_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            gt = parse_pelvis_gt(label_path, width, height)
            common = dict(source=str(image_path), imgsz=args.imgsz, conf=args.confidence, iou=args.iou, device=args.device, verbose=False)
            spine_result = spine_model.predict(max_det=40, **common)[0]
            pelvis_result = pelvis_model.predict(max_det=5, **common)[0]
            spine_raw = _tensor_rows(spine_result)
            pelvis_raw = _tensor_rows(pelvis_result)
            spine_selected, missing, duplicates = select_spine_predictions(spine_raw)
            pelvis_selected, pelvis_extra = select_pelvis_prediction(pelvis_raw)
            errors_px, errors_relative = point_errors(pelvis_selected, gt, width, height)
            tags, risk_score = risk_tags(missing, duplicates, pelvis_selected, pelvis_extra, errors_relative)
            preview_name = f"{index:03d}__{image_path.stem}.jpg"
            record = {
                "file_name": image_path.name,
                "source_image": row["image"],
                "source_label": row["label"],
                "batch": row["batch"],
                "patient_id": row["patient_id"],
                "group_id": row["group_id"],
                "cfh_source": row["cfh_source"],
                "width": width,
                "height": height,
                "preview": f"previews/{preview_name}",
                "spine_raw_count": len(spine_raw),
                "spine_selected_count": len(spine_selected),
                "spine_predictions": spine_selected,
                "missing_spine": missing,
                "duplicate_spine": duplicates,
                "pelvis_raw_count": len(pelvis_raw),
                "pelvis_prediction": pelvis_selected,
                "pelvis_extra_candidates": pelvis_extra,
                "pelvis_ground_truth": gt,
                "pelvis_error_px": errors_px,
                "pelvis_error_percent_diagonal": errors_relative,
                "risk_tags": tags,
                "risk_score": risk_score,
                "source_sha256": sha256_file(image_path),
            }
            record["pelvis_score_text"] = "漏检" if pelvis_selected is None else f"{float(pelvis_selected['score']):.3f}"
            render_preview(image_path, staging / record["preview"], spine_selected, pelvis_selected, gt, record)
            records.append(record)
            print(f"Rendered {index}/{len(manifest)}: {image_path.name}", flush=True)

        records.sort(key=lambda item: (-float(item["risk_score"]), item["file_name"]))
        summary = {
            "images": len(records),
            "old": sum(item["batch"] == "old" for item in records),
            "new": sum(item["batch"] == "new" for item in records),
            "pelvis_missing": sum(item["pelvis_prediction"] is None for item in records),
            "pelvis_extra_candidate_images": sum(item["pelvis_extra_candidates"] > 0 for item in records),
            "spine_incomplete": sum(bool(item["missing_spine"]) for item in records),
            "spine_complete": sum(not item["missing_spine"] for item in records),
            "risk_images": sum(item["risk_tags"] != ["常规"] for item in records),
        }
        report = {
            "schema_version": 1,
            "task": "independent 23-class spine + 3-keypoint pelvis prediction overlay",
            "dataset": str(args.dataset.resolve()),
            "split": "test",
            "models": {
                "spine": {"path": str(args.spine_model.resolve()), "sha256": sha256_file(args.spine_model), "names": SPINE_NAMES, "kpt_shape": [4, 3]},
                "pelvis": {"path": str(args.pelvis_model.resolve()), "sha256": sha256_file(args.pelvis_model), "names": ["pelvis"], "kpt_shape": [3, 3]},
            },
            "settings": {"imgsz": args.imgsz, "confidence": args.confidence, "iou": args.iou, "device": args.device, "selection": "highest-confidence instance per spine class and highest-confidence pelvis instance"},
            "summary": summary,
        }
        with (staging / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _write_index(staging / "index.csv", records)
        (staging / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "index.html").write_text(build_html(records, report), encoding="utf-8")
        (staging / "README.md").write_text(
            "# 23类脊柱 + 三点骨盆 test 联合预测\n\n"
            "两个模型在三点联合数据集的原始test图上独立推理；绿色四边形为23类脊柱预测，"
            "彩色实心点为三点模型预测，白色圈/十字为三点GT。打开`index.html`浏览。\n",
            encoding="utf-8",
        )
        audit = audit_package(staging, expected=len(records))
        (staging / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if audit["status"] != "passed":
            raise RuntimeError(f"package audit failed: {audit['errors']}")
        os.rename(staging, args.output)
        return {"report": report, "audit": audit, "output": str(args.output)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "datasets/yolo_pelvis_3kpt_all")
    parser.add_argument("--spine-model", type=Path, default=root / "3-model_training/runs/pose/yolo11l_lateral_23cls_best/weights/best.pt")
    parser.add_argument("--pelvis-model", type=Path, default=root / "4-model_training_CFH/runs/yolo11l_pelvis_3kpt_roi_mixed_best/weights/best.pt")
    parser.add_argument("--output", type=Path, default=root / "analysis/lateral_dual_model_test_102_20260908")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N test images")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        result = audit_package(args.output)
    else:
        result = build_package(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
