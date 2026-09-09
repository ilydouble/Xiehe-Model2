#!/usr/bin/env python3
"""Build an offline human-review package for the 20-class lateral Pose test set."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


CLASS_NAMES = ["C2", "C7", *(f"T{i}" for i in range(1, 14)), *(f"L{i}" for i in range(1, 6))]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CANVAS_SIZE = (1900, 1640)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_yolo_pose(label_path: Path, width: int, height: int) -> dict[int, dict[str, Any]]:
    objects: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"{label_path}:{line_number}: expected 17 fields, found {len(fields)}")
        class_id = int(fields[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
        if class_id in objects:
            raise ValueError(f"{label_path}:{line_number}: duplicate class {class_id}")
        values = [float(value) for value in fields[1:]]
        cx, cy, bw, bh = values[:4]
        points = []
        for offset in range(4, 16, 3):
            x, y, visible = values[offset:offset + 3]
            if not 0 <= x <= 1 or not 0 <= y <= 1 or visible <= 0:
                raise ValueError(f"{label_path}:{line_number}: invalid visible keypoint")
            points.append([x * width, y * height, visible])
        objects[class_id] = {
            "class_id": class_id,
            "bbox": [(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height],
            "keypoints": points,
        }
    if not objects:
        raise ValueError(f"{label_path}: empty label")
    return objects


def tensor_rows(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or result.keypoints is None:
        return []
    classes = result.boxes.cls.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    keypoints = result.keypoints.data.detach().cpu().tolist()
    return [
        {
            "class_id": int(class_id),
            "score": float(score),
            "bbox": [float(value) for value in bbox],
            "keypoints": [[float(value) for value in point] for point in points],
        }
        for class_id, score, bbox, points in zip(classes, scores, boxes, keypoints)
    ]


def select_predictions(predictions: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"invalid prediction class: {class_id}")
        grouped[class_id].append(prediction)
    selected: dict[int, dict[str, Any]] = {}
    for class_id, items in grouped.items():
        ranked = sorted(items, key=lambda item: float(item["score"]), reverse=True)
        selected[class_id] = ranked[0]
    return selected


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def compare_predictions(
    ground_truth: dict[int, dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    width: int,
    height: int,
    low_confidence: float = 0.25,
) -> dict[str, Any]:
    diagonal = math.hypot(width, height)
    missing_ids = sorted(set(ground_truth) - set(selected))
    false_positive_ids = sorted(set(selected) - set(ground_truth))
    low_confidence_ids = sorted(class_id for class_id in ground_truth.keys() & selected.keys() if float(selected[class_id]["score"]) < low_confidence)
    matches = []
    for class_id in sorted(ground_truth.keys() & selected.keys()):
        truth, prediction = ground_truth[class_id], selected[class_id]
        errors = [
            math.hypot(float(pred[0]) - float(gt[0]), float(pred[1]) - float(gt[1]))
            for pred, gt in zip(prediction["keypoints"], truth["keypoints"])
        ]
        matches.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "score": float(prediction["score"]),
            "bbox_iou": bbox_iou(prediction["bbox"], truth["bbox"]),
            "keypoint_mean_error_px": sum(errors) / len(errors),
            "keypoint_max_error_px": max(errors),
            "keypoint_mean_error_percent_diagonal": sum(errors) / len(errors) / diagonal * 100,
            "keypoint_max_error_percent_diagonal": max(errors) / diagonal * 100,
        })
    max_error = max((row["keypoint_max_error_percent_diagonal"] for row in matches), default=None)
    min_iou = min((row["bbox_iou"] for row in matches), default=None)
    tags: list[str] = []
    risk_score = 0.0
    if missing_ids:
        tags.append(f"漏检{len(missing_ids)}类")
        risk_score += 100 * len(missing_ids)
    if false_positive_ids:
        tags.append(f"无GT误检{len(false_positive_ids)}类")
        risk_score += 50 * len(false_positive_ids)
    if low_confidence_ids:
        tags.append(f"低置信{len(low_confidence_ids)}类")
        risk_score += 15 * len(low_confidence_ids)
    if max_error is not None:
        risk_score += max_error * 10
        if max_error >= 2.0:
            tags.append("关键点误差≥2%对角线")
        elif max_error >= 1.0:
            tags.append("关键点误差≥1%对角线")
    if min_iou is not None and min_iou < 0.5:
        tags.append("框IoU<0.5")
        risk_score += 25
    if not tags:
        tags.append("常规")
    return {
        "matches": matches,
        "missing_classes": [CLASS_NAMES[class_id] for class_id in missing_ids],
        "false_positive_classes": [CLASS_NAMES[class_id] for class_id in false_positive_ids],
        "low_confidence_classes": [CLASS_NAMES[class_id] for class_id in low_confidence_ids],
        "max_keypoint_error_percent_diagonal": max_error,
        "min_bbox_iou": min_iou,
        "risk_tags": tags,
        "risk_score": risk_score,
    }


def font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def expanded_crop(points: Sequence[Sequence[float]], width: int, height: int, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not points:
        return fallback
    xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
    span_x, span_y = max(max(xs) - min(xs), width * 0.12), max(max(ys) - min(ys), height * 0.10)
    margin_x, margin_y = max(span_x * 0.45, width * 0.025), max(span_y * 0.25, height * 0.018)
    left, top = max(0, int(min(xs) - margin_x)), max(0, int(min(ys) - margin_y))
    right, bottom = min(width, int(max(xs) + margin_x)), min(height, int(max(ys) + margin_y))
    return left, top, max(left + 2, right), max(top + 2, bottom)


def draw_text(draw: Any, xy: tuple[float, float], value: str, text_font: Any, fill: tuple[int, int, int], background=(10, 14, 20)) -> None:
    box = draw.textbbox(xy, value, font=text_font)
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=background)
    draw.text(xy, value, font=text_font, fill=fill)


def draw_panel(
    canvas: Any,
    source: Any,
    panel: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
    ground_truth: dict[int, dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    title: str,
    class_ids: set[int] | None = None,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = panel
    left, top, right, bottom = crop
    cropped = source.crop(crop)
    scale = min((x1 - x0) / cropped.width, (y1 - y0) / cropped.height)
    resized = cropped.resize((max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))))
    offset_x, offset_y = x0 + (x1 - x0 - resized.width) // 2, y0 + (y1 - y0 - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    draw.rectangle(panel, outline=(85, 100, 120), width=2)
    draw_text(draw, (x0 + 8, y0 + 7), title, font(21), (240, 244, 250))

    def mapped(point: Sequence[float]) -> tuple[float, float]:
        return offset_x + (float(point[0]) - left) * scale, offset_y + (float(point[1]) - top) * scale

    ids = sorted((set(ground_truth) | set(selected)) if class_ids is None else class_ids)
    label_font = font(15 if x1 - x0 < 900 else 18)
    for class_id in ids:
        truth, prediction = ground_truth.get(class_id), selected.get(class_id)
        if truth is not None:
            points = [mapped(point) for point in truth["keypoints"]]
            gt_color = (255, 75, 85) if prediction is None else (30, 225, 240)
            draw.line(points + [points[0]], fill=gt_color, width=3, joint="curve")
            for px, py in points:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline=gt_color, width=2)
            draw_text(draw, points[0], f"GT {CLASS_NAMES[class_id]}", label_font, gt_color)
        if prediction is not None:
            points = [mapped(point) for point in prediction["keypoints"]]
            pred_color = (255, 170, 45) if truth is None else (65, 235, 115)
            draw.line(points + [points[0]], fill=pred_color, width=3, joint="curve")
            for px, py in points:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=pred_color)
            anchor = points[1] if len(points) > 1 else points[0]
            draw_text(draw, anchor, f"P {CLASS_NAMES[class_id]} {float(prediction['score']):.2f}", label_font, pred_color)


def render_preview(image_path: Path, output_path: Path, ground_truth: dict[int, dict[str, Any]], selected: dict[int, dict[str, Any]], record: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    canvas = Image.new("RGB", CANVAS_SIZE, (14, 18, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), record["file_name"], font=font(27), fill=(238, 243, 250))
    missing = ",".join(record["missing_classes"]) or "无"
    false_positive = ",".join(record["false_positive_classes"]) or "无"
    worst = record["max_keypoint_error_percent_diagonal"]
    worst_text = "NA" if worst is None else f"{worst:.3f}%图像对角线"
    draw.text((24, 58), f"来源 {record['source_batch']} | GT {len(ground_truth)}类 | 命中 {len(record['matches'])}类 | 漏检 {missing} | 无GT误检 {false_positive}", font=font(19), fill=(225, 230, 238))
    draw.text((24, 88), f"最大关键点误差 {worst_text} | 风险 {' / '.join(record['risk_tags'])}", font=font(19), fill=(255, 190, 80) if record["risk_tags"] != ["常规"] else (100, 235, 155))
    draw.text((24, 118), "青=GT；绿=预测；红=漏检GT；橙=无GT预测。左为全图，右为颈胸段/中胸段/胸腰段放大。", font=font(17), fill=(175, 190, 210))

    groups = [
        ({0, 1, 2, 3, 4, 5, 6}, "颈胸段 C2/C7/T1-T5"),
        (set(range(7, 14)), "中胸段 T6-T12"),
        (set(range(14, 20)), "胸腰段 T13/L1-L5"),
    ]
    draw_panel(canvas, source, (20, 160, 810, 1615), (0, 0, width, height), ground_truth, selected, "全图 GT vs 预测")
    right_panels = [(830, 160, 1880, 625), (830, 650, 1880, 1115), (830, 1140, 1880, 1615)]
    for (class_ids, title), panel in zip(groups, right_panels):
        points = [point for class_id in class_ids for obj in (ground_truth.get(class_id), selected.get(class_id)) if obj for point in obj["keypoints"]]
        crop = expanded_crop(points, width, height, (0, 0, width, height))
        draw_panel(canvas, source, panel, crop, ground_truth, selected, title, class_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=91, optimize=True)


def build_html(report: dict[str, Any]) -> str:
    summary, metrics = report["summary"], report.get("official_metrics", {})
    pose_map = metrics.get("metrics/mAP50-95(P)")
    pose_text = "未提供" if pose_map is None else f"{pose_map:.4f}"
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>20类侧面Pose test人工核验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0d1118;color:#edf2f8;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}header{{position:sticky;top:0;z-index:4;background:#111824f2;padding:14px 20px;border-bottom:1px solid #2b3749}}h1{{font-size:23px;margin:0 0 7px}}.summary,.legend,.position{{color:#aab8ca;font-size:14px;margin:5px 0}}.controls{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}button,input,select,textarea{{background:#182235;color:#edf2f8;border:1px solid #3b4960;border-radius:7px;padding:8px 10px}}button{{cursor:pointer}}.viewer{{max-width:1920px;margin:14px auto;padding:0 12px}}.card{{background:#151d29;border:1px solid #2c394d;border-radius:10px;padding:12px}}.card img{{display:block;width:100%;max-height:calc(100vh - 315px);object-fit:contain;background:#080b0f}}.meta{{display:flex;gap:7px;flex-wrap:wrap;margin:9px 0}}.tag{{background:#303c4d;border-radius:13px;padding:4px 9px;font-size:13px}}.danger{{background:#81313b}}.ok{{background:#225b3a}}.review{{display:grid;gap:8px}}.decision{{display:grid;grid-template-columns:1fr 1fr auto auto;gap:8px}}.decision button{{min-height:45px;font-weight:700}}.accept{{background:#17633a}}.reject{{background:#7a2530}}button.active{{outline:3px solid #fff}}@media(max-width:800px){{.decision{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>20类侧面脊柱 Pose：test 人工核验</h1><div class=\"summary\">{summary['images']}张（第一批{summary['first']} / 第二批{summary['second']}） · 风险{summary['risk_images']}张 · 漏检{summary['missing_images']}张 · 无GT误检{summary['false_positive_images']}张 · 官方Pose mAP50-95 {pose_text}</div><div class=\"legend\">青=GT，绿=预测，红=漏检GT，橙=无GT预测；每类只保留最高置信度预测。</div><div class=\"controls\"><input id=\"q\" placeholder=\"搜索文件名或类别\"><select id=\"filter\"><option value=\"all\">全部</option><option value=\"risk\">仅风险</option><option value=\"missing\">有漏检</option><option value=\"false\">有无GT误检</option><option value=\"low\">有低置信</option><option value=\"error\">关键点误差≥1%</option><option value=\"first\">第一批</option><option value=\"second\">第二批</option><option value=\"todo\">未判断</option></select><button id=\"prev\">← 上一张</button><button id=\"next\">下一张 →</button><input id=\"jump\" type=\"number\" min=\"1\" style=\"width:82px\"><button id=\"export\">导出人工核验CSV</button></div></header><main class=\"viewer\"><section class=\"card\"><a id=\"link\" target=\"_blank\"><img id=\"preview\"></a><div class=\"meta\" id=\"meta\"></div><div class=\"review\"><div class=\"decision\"><button class=\"accept\" data-verdict=\"accepted\">✓ 接受并下一张 A</button><button class=\"reject\" data-verdict=\"rejected\">✕ 拒绝并下一张 R</button><button data-verdict=\"accept_after_edit\">修改后接受</button><button data-verdict=\"uncertain\">不确定</button></div><span id=\"status\" class=\"position\">当前：未判断</span><textarea id=\"note\" rows=\"2\" placeholder=\"人工备注\"></textarea></div><div id=\"position\" class=\"position\"></div></section></main><script src=\"review_data.js\"></script><script>
const $=id=>document.getElementById(id),d=window.LATERAL20_TEST_REVIEW,s=d.samples,key='lateral20-test-'+d.summary.model_sha256.slice(0,16),vt={{accepted:'接受',rejected:'拒绝',accept_after_edit:'修改后接受',uncertain:'不确定'}};let saved={{}},ids=[],at=0;try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(e){{}}
function match(x){{const q=$('q').value.trim().toLowerCase(),f=$('filter').value,r=saved[x.source_image]||{{}};if(q&&!x.search_text.includes(q))return false;return f==='all'||(f==='risk'&&x.risk_tags[0]!=='常规')||(f==='missing'&&x.missing_classes.length)||(f==='false'&&x.false_positive_classes.length)||(f==='low'&&x.low_confidence_classes.length)||(f==='error'&&(x.max_keypoint_error_percent_diagonal||0)>=1)||f===x.source_batch||(f==='todo'&&!r.verdict)}}
function refresh(){{ids=s.map((x,i)=>i).filter(i=>match(s[i])).sort((a,b)=>s[b].risk_score-s[a].risk_score||s[a].file_name.localeCompare(s[b].file_name));at=Math.min(at,Math.max(0,ids.length-1));render()}}
function render(){{const x=s[ids[at]],done=s.filter(x=>saved[x.source_image]?.verdict).length;$('summary').textContent=`已判断 ${{done}}/${{s.length}}；当前筛选 ${{ids.length}} 张；风险 ${{d.summary.risk_images}}，漏检 ${{d.summary.missing_images}}，无GT误检 ${{d.summary.false_positive_images}}`;if(!x){{$('preview').removeAttribute('src');$('meta').textContent='没有匹配样本';$('position').textContent='0/0';return}}$('preview').src=x.preview;$('link').href=x.preview;const bad=x.risk_tags[0]!=='常规',err=x.max_keypoint_error_percent_diagonal==null?'NA':x.max_keypoint_error_percent_diagonal.toFixed(3)+'%';$('meta').innerHTML=`<span class=\"tag ${{bad?'danger':'ok'}}\">${{x.source_batch}}</span><span class=tag>风险 ${{x.risk_tags.join(' / ')}}</span><span class=tag>漏检 ${{x.missing_classes.join(',')||'无'}}</span><span class=tag>无GT误检 ${{x.false_positive_classes.join(',')||'无'}}</span><span class=tag>低置信 ${{x.low_confidence_classes.join(',')||'无'}}</span><span class=tag>最大关键点误差 ${{err}}</span>`;const r=saved[x.source_image]||{{}};$('note').value=r.note||'';$('status').textContent='当前：'+(vt[r.verdict]||'未判断');document.querySelectorAll('[data-verdict]').forEach(b=>b.classList.toggle('active',b.dataset.verdict===(r.verdict||'')));$('jump').value=at+1;$('position').textContent=`${{at+1}}/${{ids.length}}　${{x.source_image}}`}}
function move(n){{at=Math.max(0,Math.min(ids.length-1,at+n));render()}}function save(v){{const x=s[ids[at]];if(!x)return;saved[x.source_image]={{verdict:v,note:$('note').value,updated_at:new Date().toISOString()}};try{{localStorage.setItem(key,JSON.stringify(saved))}}catch(e){{}}if($('filter').value==='todo')refresh();else move(1)}}
function exportCsv(){{const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=[['source_image','verdict','note','updated_at']];s.forEach(x=>{{const r=saved[x.source_image]||{{}};rows.push([x.source_image,r.verdict||'',r.note||'',r.updated_at||''])}});const blob=new Blob(['\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='20类侧面test人工核验结果.csv';a.click();URL.revokeObjectURL(a.href)}}
$('filter').onchange=()=>{{at=0;refresh()}};$('q').oninput=()=>{{at=0;refresh()}};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>{{at=Math.max(0,Math.min(ids.length-1,Number($('jump').value)-1));render()}};document.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>save(b.dataset.verdict));document.onkeydown=e=>{{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key.toLowerCase()==='a')save('accepted');if(e.key.toLowerCase()==='r')save('rejected')}};$('export').onclick=exportCsv;refresh();</script></body></html>"""


def validate_html_runtime(html_text: str) -> None:
    broken = ".join('" + chr(10) + "')"
    expected = ".join('" + chr(92) + "n')"
    if broken in html_text or expected not in html_text:
        raise ValueError("Generated HTML contains an invalid JavaScript newline literal")


def write_index(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = ["rank", "file_name", "source_batch", "patient_id", "source_image", "preview", "gt_classes", "matched_classes", "missing_classes", "false_positive_classes", "low_confidence_classes", "max_keypoint_error_percent_diagonal", "min_bbox_iou", "risk_tags", "risk_score"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            writer.writerow({
                "rank": rank,
                "file_name": record["file_name"],
                "source_batch": record["source_batch"],
                "patient_id": record["patient_id"],
                "source_image": record["source_image"],
                "preview": record["preview"],
                "gt_classes": ";".join(record["gt_classes"]),
                "matched_classes": ";".join(row["class_name"] for row in record["matches"]),
                "missing_classes": ";".join(record["missing_classes"]),
                "false_positive_classes": ";".join(record["false_positive_classes"]),
                "low_confidence_classes": ";".join(record["low_confidence_classes"]),
                "max_keypoint_error_percent_diagonal": "" if record["max_keypoint_error_percent_diagonal"] is None else f"{record['max_keypoint_error_percent_diagonal']:.6f}",
                "min_bbox_iou": "" if record["min_bbox_iou"] is None else f"{record['min_bbox_iou']:.6f}",
                "risk_tags": ";".join(record["risk_tags"]),
                "risk_score": f"{record['risk_score']:.6f}",
            })


def write_file_manifest(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "file_manifest.csv" and not path.name.startswith("._"))
    with (root / "file_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in paths:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path)])


def remove_apple_double(root: Path) -> int:
    paths = sorted(path for path in root.rglob("._*") if path.is_file())
    for path in paths:
        path.unlink()
    return len(paths)


def audit_package(output_dir: Path, expected: int | None = None) -> dict[str, Any]:
    from PIL import Image

    errors: list[str] = []
    try:
        with (output_dir / "index.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        return {"status": "failed", "errors": ["missing index.csv"]}
    if expected is not None and len(rows) != expected:
        errors.append(f"expected {expected} rows, found {len(rows)}")
    previews: set[str] = set()
    for row in rows:
        relative = row.get("preview", "")
        if not relative or relative in previews:
            errors.append(f"missing/duplicate preview path: {relative}")
            continue
        previews.add(relative)
        try:
            with Image.open(output_dir / relative) as image:
                if image.size != CANVAS_SIZE:
                    errors.append(f"wrong preview size {image.size}: {relative}")
                image.verify()
        except Exception as exc:
            errors.append(f"unreadable preview {relative}: {exc}")
    prediction_rows = sum(1 for line in (output_dir / "predictions.jsonl").open(encoding="utf-8") if line.strip()) if (output_dir / "predictions.jsonl").exists() else 0
    if prediction_rows != len(rows):
        errors.append(f"prediction/index mismatch: {prediction_rows} != {len(rows)}")
    for required in ("index.html", "review_data.js", "build_report.json", "README.md", "file_manifest.csv"):
        if not (output_dir / required).exists():
            errors.append(f"missing {required}")
    apple_double = list(output_dir.rglob("._*"))
    if apple_double:
        errors.append(f"AppleDouble files: {len(apple_double)}")
    return {"status": "passed" if not errors else "failed", "counts": {"index_rows": len(rows), "previews": len(previews), "prediction_rows": prediction_rows}, "errors": errors}


def read_metrics(evaluation_dir: Path | None) -> dict[str, float]:
    if evaluation_dir is None:
        return {}
    path = evaluation_dir / "metrics_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.get("overall", {}).items()}


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image
    from ultralytics import YOLO

    for required in (args.dataset, args.model):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.output.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {args.output}")
    with (args.dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    manifest.sort(key=lambda row: row["image"])
    if args.limit:
        manifest = manifest[:args.limit]
    if not manifest:
        raise ValueError("No test rows in manifest")

    model = YOLO(str(args.model))
    names = [model.names[index] for index in range(len(model.names))]
    kpt_shape = getattr(model.model, "kpt_shape", model.model.yaml.get("kpt_shape"))
    if model.task != "pose" or names != CLASS_NAMES or list(kpt_shape) != [4, 3]:
        raise ValueError(f"Unexpected model contract: task={model.task}, names={names}, kpt_shape={kpt_shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    records: list[dict[str, Any]] = []
    model_hash = sha256_file(args.model)
    try:
        for index, row in enumerate(manifest, 1):
            image_path, label_path = args.dataset / row["image"], args.dataset / row["label"]
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(f"missing pair: {image_path}, {label_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            ground_truth = parse_yolo_pose(label_path, width, height)
            result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.confidence, iou=args.iou, max_det=args.max_det, device=args.device, verbose=False)[0]
            raw = tensor_rows(result)
            selected = select_predictions(raw)
            comparison = compare_predictions(ground_truth, selected, width, height, args.low_confidence)
            preview = f"previews/{index:03d}__{image_path.stem}.jpg"
            record = {
                "file_name": image_path.name,
                "source_image": row["image"],
                "source_label": row["label"],
                "source_batch": row["source_batch"],
                "patient_id": row["patient_id"],
                "width": width,
                "height": height,
                "preview": preview,
                "gt_classes": [CLASS_NAMES[class_id] for class_id in sorted(ground_truth)],
                "ground_truth": [ground_truth[class_id] for class_id in sorted(ground_truth)],
                "raw_prediction_count": len(raw),
                "selected_predictions": [selected[class_id] for class_id in sorted(selected)],
                "source_sha256": sha256_file(image_path),
                **comparison,
            }
            record["search_text"] = " ".join([record["file_name"], *record["missing_classes"], *record["false_positive_classes"], *record["risk_tags"]]).lower()
            render_preview(image_path, staging / preview, ground_truth, selected, record)
            records.append(record)
            print(f"Rendered {index}/{len(manifest)}: {image_path.name}", flush=True)

        records.sort(key=lambda item: (-float(item["risk_score"]), item["file_name"]))
        summary = {
            "images": len(records),
            "first": sum(row["source_batch"] == "first" for row in records),
            "second": sum(row["source_batch"] == "second" for row in records),
            "risk_images": sum(row["risk_tags"] != ["常规"] for row in records),
            "missing_images": sum(bool(row["missing_classes"]) for row in records),
            "false_positive_images": sum(bool(row["false_positive_classes"]) for row in records),
            "low_confidence_images": sum(bool(row["low_confidence_classes"]) for row in records),
            "keypoint_error_ge_1pct_images": sum((row["max_keypoint_error_percent_diagonal"] or 0) >= 1 for row in records),
            "model_sha256": model_hash,
        }
        report = {
            "schema_version": 1,
            "task": "20-class lateral-spine pose test prediction and human review",
            "dataset": str(args.dataset.resolve()),
            "split": "test",
            "model": {"path": str(args.model.resolve()), "sha256": model_hash, "names": CLASS_NAMES, "kpt_shape": [4, 3]},
            "settings": {"imgsz": args.imgsz, "confidence": args.confidence, "iou": args.iou, "max_det": args.max_det, "low_confidence": args.low_confidence, "device": args.device, "selection": "highest-confidence prediction per class"},
            "official_metrics": read_metrics(args.evaluation_dir),
            "summary": summary,
        }
        with (staging / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_index(staging / "index.csv", records)
        (staging / "review_data.js").write_text("window.LATERAL20_TEST_REVIEW = " + json.dumps({"summary": summary, "samples": records}, ensure_ascii=False) + ";\n", encoding="utf-8")
        (staging / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        html_text = build_html(report)
        validate_html_runtime(html_text)
        (staging / "index.html").write_text(html_text, encoding="utf-8")
        (staging / "README.md").write_text(
            "# 20类侧面脊柱Pose test人工核验包\n\n"
            "双击`index.html`逐张核验。青色为GT，绿色为预测，红色为漏检GT，橙色为无GT预测。"
            "页面判断保存在浏览器本地，请完成后点击“导出人工核验CSV”备份。\n\n"
            f"模型SHA-256：`{model_hash}`\n",
            encoding="utf-8",
        )
        if args.evaluation_dir is not None:
            if not args.evaluation_dir.is_dir():
                raise FileNotFoundError(args.evaluation_dir)
            shutil.copytree(args.evaluation_dir, staging / "evaluation")
        removed_apple_double = remove_apple_double(staging)
        report["removed_apple_double_files"] = removed_apple_double
        (staging / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_file_manifest(staging)
        remove_apple_double(staging)
        audit = audit_package(staging, len(records))
        if audit["status"] != "passed":
            raise RuntimeError(f"Package audit failed: {audit['errors']}")
        (staging / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        remove_apple_double(staging)
        audit = audit_package(staging, len(records))
        if audit["status"] != "passed":
            raise RuntimeError(f"Final package audit failed: {audit['errors']}")
        os.rename(staging, args.output)
        return {"output": str(args.output), "report": report, "audit": audit}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def refresh_html(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    html_text = build_html(report)
    validate_html_runtime(html_text)
    temporary = output_dir / ".index.html.tmp"
    temporary.write_text(html_text, encoding="utf-8")
    os.replace(temporary, output_dir / "index.html")
    remove_apple_double(output_dir)
    audit = audit_package(output_dir)
    if audit["status"] != "passed":
        raise RuntimeError(f"Package audit failed after HTML refresh: {audit['errors']}")
    return {"output": str(output_dir), "html": "refreshed", "audit": audit}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "datasets/yolo_lateral_reviewed_combined_20cls")
    parser.add_argument("--model", type=Path, default=root / "3-model_training/runs/pose/yolo11l_lateral_20cls_scratch_best/weights/best.pt")
    parser.add_argument("--evaluation-dir", type=Path, default=root / "analysis/lateral_20cls_test_eval_100_20260909")
    parser.add_argument("--output", type=Path, default=Path("/Volumes/E/spine_data/lateral_20cls_test_review_100_20260909"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--low-confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh-html", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        result = audit_package(args.output)
    elif args.refresh_html:
        result = refresh_html(args.output)
    else:
        result = build_package(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
