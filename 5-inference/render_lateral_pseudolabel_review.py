#!/usr/bin/env python3
"""Render a complete offline review package for first-batch lateral pseudo-labels."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_NAMES = [
    *(f"C{i}" for i in range(2, 8)),
    *(f"T{i}" for i in range(1, 13)),
    *(f"L{i}" for i in range(1, 6)),
]
CERVICAL_LABELS = {f"C{i}" for i in range(2, 8)}
C2_C6_LABELS = {f"C{i}" for i in range(2, 7)}
BASELINE_LABELS = ["C7", *(f"T{i}" for i in range(1, 13)), *(f"L{i}" for i in range(1, 6))]
DEFAULT_SOURCE = Path("/Volumes/E/spine_data/LAT202511")
DEFAULT_PSEUDOLABEL_ROOT = Path("datasets/lateral_first_batch_pseudolabel_23cls")
DEFAULT_OUTPUT = Path("/Volumes/E/spine_data/LAT202511_侧面补标人工复核可视化_384份_20260908")
DEFAULT_C2C6_OUTPUT = Path("/Volumes/E/spine_data/LAT202511_仅补C2-C6_人工接受拒绝复核包_20260908")

BACKGROUND = (19, 22, 28)
PANEL_BACKGROUND = (5, 7, 10)
WHITE = (245, 247, 250)
MUTED = (183, 193, 207)
MANUAL_COLOR = (0, 220, 245)
QUALITY_COLORS = {
    "high": (48, 218, 105),
    "medium": (255, 174, 48),
    "low": (255, 72, 72),
}


def choose_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data.get("shapes"), list):
        raise ValueError(f"LabelMe shapes is not a list: {path}")
    return data


def is_target_polygon(shape: dict[str, Any]) -> bool:
    return (
        str(shape.get("label") or "").strip().upper() in CLASS_NAMES
        and shape.get("shape_type") == "polygon"
        and isinstance(shape.get("points"), list)
        and len(shape["points"]) >= 4
    )


def is_pseudo(shape: dict[str, Any]) -> bool:
    return bool((shape.get("flags") or {}).get("pseudo_label"))


def shape_quality(shape: dict[str, Any]) -> str:
    flags = shape.get("flags") or {}
    for quality in ("low", "medium", "high"):
        if flags.get(f"quality_{quality}"):
            return quality
    description = str(shape.get("description") or "")
    for quality in ("low", "medium", "high"):
        if f"quality={quality}" in description:
            return quality
    return "low"


def compose_c2c6_annotation(
    source_annotation: dict[str, Any], candidate_annotation: dict[str, Any]
) -> dict[str, Any]:
    """Keep every source shape and append only C2-C6 model pseudo-labels."""
    output = copy.deepcopy(source_annotation)
    for shape in candidate_annotation.get("shapes", []):
        label = str(shape.get("label") or "").strip().upper()
        if is_pseudo(shape) and label in C2_C6_LABELS and is_target_polygon(shape):
            output["shapes"].append(copy.deepcopy(shape))
    return output


def missing_baseline_labels(annotation: dict[str, Any]) -> list[str]:
    labels = [
        str(shape.get("label") or "").strip().upper()
        for shape in annotation.get("shapes", [])
        if is_target_polygon(shape) and not is_pseudo(shape)
    ]
    return [label for label in BASELINE_LABELS if labels.count(label) != 1]


def c2_boundary_info(
    annotation: dict[str, Any], width: int, height: int, near_fraction: float = 0.005
) -> dict[str, Any]:
    candidates = [
        shape for shape in annotation.get("shapes", [])
        if is_target_polygon(shape)
        and is_pseudo(shape)
        and str(shape.get("label") or "").strip().upper() == "C2"
    ]
    if not candidates:
        return {"state": "missing", "margin_px": None, "margin_fraction": None}
    points = candidates[0]["points"]
    margin = min(
        min(float(x), float(width) - float(x), float(y), float(height) - float(y))
        for x, y in points
    )
    fraction = margin / max(1, min(width, height))
    state = "touch" if margin <= 1.01 else ("near" if fraction <= near_fraction else "safe")
    return {
        "state": state,
        "margin_px": round(margin, 3),
        "margin_fraction": round(fraction, 6),
    }


def polygon_box(shapes: Iterable[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    points = [
        (float(point[0]), float(point[1]))
        for shape in shapes
        for point in shape.get("points", [])
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def cervical_crop(
    annotation: dict[str, Any], width: int, height: int, target_aspect: float = 0.82
) -> tuple[int, int, int, int]:
    """Return a padded C2-C7 crop, falling back to an upper-spine crop around C7."""
    target_shapes = [
        shape
        for shape in annotation["shapes"]
        if is_target_polygon(shape)
        and str(shape.get("label") or "").strip().upper() in CERVICAL_LABELS
    ]
    box = polygon_box(target_shapes)
    if box is None:
        return 0, 0, width, max(1, round(height * 0.32))

    x0, y0, x1, y1 = box
    raw_w = max(1.0, x1 - x0)
    raw_h = max(1.0, y1 - y0)
    pad_x = max(raw_w * 0.85, width * 0.055)
    pad_y = max(raw_h * 0.20, height * 0.018)
    x0, x1 = x0 - pad_x, x1 + pad_x
    y0, y1 = y0 - pad_y, y1 + pad_y

    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w / crop_h < target_aspect:
        wanted_w = crop_h * target_aspect
        delta = (wanted_w - crop_w) / 2
        x0, x1 = x0 - delta, x1 + delta
    else:
        wanted_h = crop_w / target_aspect
        delta = (wanted_h - crop_h) / 2
        y0, y1 = y0 - delta, y1 + delta

    def shift_inside(low: float, high: float, limit: int) -> tuple[float, float]:
        span = min(float(limit), high - low)
        low = max(0.0, min(float(limit) - span, low))
        return low, low + span

    x0, x1 = shift_inside(x0, x1, width)
    y0, y1 = shift_inside(y0, y1, height)
    return round(x0), round(y0), max(round(x0) + 1, round(x1)), max(round(y0) + 1, round(y1))


def fit_crop(
    source: Image.Image,
    crop: tuple[int, int, int, int],
    panel_size: tuple[int, int],
) -> tuple[Image.Image, float, int, int]:
    cropped = source.crop(crop)
    panel_w, panel_h = panel_size
    scale = min(panel_w / cropped.width, panel_h / cropped.height)
    rendered = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", panel_size, PANEL_BACKGROUND)
    offset_x = (panel_w - rendered.width) // 2
    offset_y = (panel_h - rendered.height) // 2
    panel.paste(rendered, (offset_x, offset_y))
    return panel, scale, offset_x, offset_y


def draw_annotations(
    panel: Image.Image,
    annotation: dict[str, Any],
    crop: tuple[int, int, int, int],
    scale: float,
    offset_x: int,
    offset_y: int,
    *,
    zoom: bool,
) -> None:
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = choose_font(25 if zoom else 16)
    point_font = choose_font(16)
    manual_width = 3 if zoom else 2
    pseudo_width = 7 if zoom else 4
    crop_x0, crop_y0, _, _ = crop

    for shape in annotation["shapes"]:
        if not is_target_polygon(shape):
            continue
        label = str(shape["label"]).strip().upper()
        points = [
            (
                offset_x + (float(point[0]) - crop_x0) * scale,
                offset_y + (float(point[1]) - crop_y0) * scale,
            )
            for point in shape["points"]
        ]
        pseudo = is_pseudo(shape)
        quality = shape_quality(shape) if pseudo else "manual"
        color = QUALITY_COLORS.get(quality, MANUAL_COLOR)
        if pseudo:
            draw.polygon(points, fill=(*color, 32))
        draw.line(points + [points[0]], fill=(*color, 255), width=pseudo_width if pseudo else manual_width, joint="curve")

        anchor = min(points, key=lambda point: (point[1], point[0]))
        suffix = f" {quality[0].upper()}" if pseudo else (" 人工" if zoom else "")
        text = label + suffix
        tx, ty = anchor[0] + 5, anchor[1] - (29 if zoom else 20)
        draw.text(
            (tx, ty), text, font=font, fill=(*color, 255),
            stroke_width=3 if zoom else 2, stroke_fill=(0, 0, 0, 235),
        )
        if zoom and pseudo:
            radius = 7
            for number, (px, py) in enumerate(points[:4], 1):
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=(*color, 255), outline=(255, 255, 255, 255), width=2,
                )
                draw.text(
                    (px + 8, py - 10), str(number), font=point_font,
                    fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255),
                )
    panel.paste(Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB"))


def summarize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    manual_labels: list[str] = []
    added: dict[str, list[str]] = defaultdict(list)
    for shape in annotation["shapes"]:
        if not is_target_polygon(shape):
            continue
        label = str(shape["label"]).strip().upper()
        if is_pseudo(shape):
            added[shape_quality(shape)].append(label)
        else:
            manual_labels.append(label)
    return {
        "manual_labels": manual_labels,
        "added_labels": [label for quality in ("high", "medium", "low") for label in added[quality]],
        "high_labels": added["high"],
        "medium_labels": added["medium"],
        "low_labels": added["low"],
    }


def priority_for(unresolved: Sequence[str], summary: dict[str, Any]) -> str:
    if unresolved:
        return "缺候选"
    if summary["low_labels"]:
        return "含low"
    if summary["medium_labels"]:
        return "含medium"
    return "仅high"


def render_preview(
    source_path: Path,
    annotation: dict[str, Any],
    output_path: Path,
    index: int,
    unresolved: Sequence[str],
    *,
    only_c2c6: bool = False,
    original_missing: Sequence[str] = (),
    c2_boundary: dict[str, Any] | None = None,
    baseline_name: str = "原始",
    sample_warning: str = "",
) -> dict[str, Any]:
    with Image.open(source_path) as opened:
        opened.load()
        source = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = source.size
    full_crop = (0, 0, width, height)
    neck_crop = cervical_crop(annotation, width, height)
    full_panel, full_scale, full_x, full_y = fit_crop(source, full_crop, (720, 1460))
    neck_panel, neck_scale, neck_x, neck_y = fit_crop(source, neck_crop, (1000, 1120))
    draw_annotations(full_panel, annotation, full_crop, full_scale, full_x, full_y, zoom=False)
    draw_annotations(neck_panel, annotation, neck_crop, neck_scale, neck_x, neck_y, zoom=True)

    stats = summarize_annotation(annotation)
    priority = priority_for(unresolved, stats)
    boundary_state = (c2_boundary or {}).get("state", "safe")
    risk_color = (255, 82, 82) if unresolved or stats["low_labels"] or boundary_state == "touch" else ((255, 190, 65) if stats["medium_labels"] or boundary_state == "near" else (100, 235, 150))
    canvas = Image.new("RGB", (1780, 1750), BACKGROUND)
    canvas.paste(full_panel, (20, 170))
    canvas.paste(neck_panel, (760, 170))
    draw = ImageDraw.Draw(canvas)
    title_font = choose_font(26)
    body_font = choose_font(20)
    small_font = choose_font(18)
    name = source_path.name
    title = f"{index:04d}  {name}"
    if len(title) > 118:
        title = title[:115] + "..."
    draw.text((20, 14), title, font=title_font, fill=WHITE)
    prefix = "仅补C2-C6" if only_c2c6 else f"优先级：{priority}"
    draw.text((20, 57), f"{prefix}  |  补标 {len(stats['added_labels'])} 个  (high {len(stats['high_labels'])} / medium {len(stats['medium_labels'])} / low {len(stats['low_labels'])})", font=body_font, fill=risk_color)
    unresolved_text = ", ".join(unresolved) if unresolved else "无"
    if only_c2c6:
        boundary_names = {"touch": "C2触边", "near": "C2近边", "safe": "C2边界安全", "missing": "C2无候选"}
        boundary_text = boundary_names.get(boundary_state, boundary_state)
        if (c2_boundary or {}).get("margin_px") is not None:
            boundary_text += f" ({c2_boundary['margin_px']}px)"
        draw.text((20, 94), f"C2-C6没有候选：{unresolved_text}  |  {boundary_text}  |  请人工接受或拒绝", font=body_font, fill=risk_color)
        missing_text = ", ".join(original_missing) if original_missing else "无"
        if len(missing_text) > 78:
            missing_text = missing_text[:75] + "..."
        baseline_line = f"{baseline_name}C7-L5缺级提示：{missing_text}（只提示，不自动排除，也不由模型补）"
        if sample_warning:
            baseline_line += f"  |  {sample_warning}"
        if len(baseline_line) > 128:
            baseline_line = baseline_line[:125] + "..."
        draw.text(
            (20, 128), baseline_line, font=small_font,
            fill=(255, 190, 65) if original_missing or sample_warning else MUTED,
        )
    else:
        draw.text((20, 94), f"没有生成候选：{unresolved_text}  |  所有模型补标都必须人工复核", font=body_font, fill=risk_color)
        draw.text((20, 128), "青=第一批原始人工polygon；绿=high；橙=medium；红=low。右图数字1→4为Pose四角点顺序。", font=small_font, fill=MUTED)
    draw.text((20, 1630), "左：完整侧位片（用于检查全脊柱编号和上下顺序）", font=body_font, fill=WHITE)
    draw.text((760, 1310), "右：C2–C7局部放大（重点核对补标框是否贴合椎体）", font=body_font, fill=WHITE)
    draw.text((760, 1355), f"颈椎裁剪框：x={neck_crop[0]}–{neck_crop[2]}, y={neck_crop[1]}–{neck_crop[3]}", font=small_font, fill=MUTED)
    draw.text((760, 1390), f"原始尺寸：{width}×{height}", font=small_font, fill=MUTED)
    draw.text((760, 1430), "人工复核建议：先看右图 C2→C7 顺序与边界，再回左图核对整体层级。", font=small_font, fill=WHITE)
    edit_text = "此图只用于接受/拒绝判断；待确认后再生成严格C2-C6 LabelMe数据。" if only_c2c6 else "此图只用于核验；实际修改请在 LabelMe 候选 JSON 中完成。"
    draw.text((760, 1470), edit_text, font=small_font, fill=(255, 190, 65))
    if only_c2c6:
        draw.text((760, 1510), f"青={baseline_name}；绿/橙/红=仅C2-C6模型候选；没有显示任何其它类别伪标签。", font=small_font, fill=MUTED)
    draw.text((20, 1688), "REVIEW REQUIRED — 图上 high 不等于已经确认。", font=body_font, fill=(255, 190, 65))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=90, optimize=True)
    return {
        **stats,
        "priority": priority,
        "width": width,
        "height": height,
        "neck_crop": list(neck_crop),
        "original_missing_labels": list(original_missing),
        "c2_boundary": boundary_state,
        "c2_edge_px": (c2_boundary or {}).get("margin_px"),
        "c2_edge_fraction": (c2_boundary or {}).get("margin_fraction"),
    }


def snapshot_source(rows: Sequence[dict[str, str]], source: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for row in rows:
        image = source / row["relative_image"]
        annotation = Path(row["source_json"])
        for path in (image, annotation):
            stat = path.stat()
            snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一批侧面补标人工复核</title><style>
*{box-sizing:border-box}body{margin:0;background:#11151b;color:#eef2f7;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{position:sticky;top:0;z-index:3;background:#1b222cf4;padding:14px 20px;box-shadow:0 2px 12px #0009}h1{font-size:22px;margin:0 0 8px}.summary{color:#b9c4d2;margin:7px 0}.controls{display:flex;flex-wrap:wrap;gap:8px}button,input,select,textarea{background:#10161e;color:#fff;border:1px solid #445267;border-radius:7px;padding:8px 10px}.viewer{max-width:1840px;margin:16px auto;padding:0 14px}.card{background:#1a222d;border:1px solid #344151;border-radius:10px;padding:13px}.card img{display:block;width:100%;max-height:calc(100vh - 255px);object-fit:contain;background:#080b0f}.meta{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.tag{background:#303b49;border-radius:13px;padding:4px 9px}.danger{background:#822f38}.warn{background:#76521d}.ok{background:#245f3c}.review{display:grid;grid-template-columns:minmax(190px,280px) 1fr auto;gap:8px}.position{color:#b9c4d2;margin-top:8px}.legend{font-size:13px;color:#b9c4d2}.green{color:#49dc78}.orange{color:#ffb342}.red{color:#ff6262}.cyan{color:#24dcef}</style></head><body><header><h1 id="heading">第一批侧面补标人工复核</h1><div class="summary" id="summary"></div><div class="legend"><span class="cyan">青=原始人工polygon</span>；<span class="green">绿=high</span>；<span class="orange">橙=medium</span>；<span class="red">红=low/缺候选/触边</span>。最终由人工接受或拒绝。</div><div class="controls"><input id="q" placeholder="搜索文件名"><select id="filter"><option value="all">全部</option><option value="missing">C2-C6缺候选</option><option value="touch">C2触边</option><option value="near">C2近边</option><option value="originalmissing">原始C7-L5缺级提示</option><option value="low">含low</option><option value="medium">含medium</option><option value="high">仅high</option><option value="todo">未判断</option></select><select id="sort"><option value="priority">风险优先</option><option value="name">文件名</option></select><button id="prev">← 上一张</button><button id="next">下一张 →</button><input id="jump" type="number" min="1" style="width:88px"><button id="export">导出人工复核CSV</button></div></header><main class="viewer"><section class="card"><a id="link" target="_blank"><img id="preview"></a><div class="meta" id="meta"></div><div class="review"><select id="verdict"><option value="">未判断</option><option value="accepted">接受</option><option value="rejected">拒绝</option><option value="accept_after_edit">修改后接受</option><option value="uncertain">不确定</option></select><textarea id="note" rows="2" placeholder="人工备注"></textarea><button id="save">保存并下一张</button></div><div class="position" id="position"></div></section></main><script src="review_data.js"></script><script>
const $=id=>document.getElementById(id),d=window.LATERAL_REVIEW,s=d.samples,key=d.summary.review_key||'lateral-pseudolabel-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{}');let ids=[],at=0;const rank={"缺候选":0,"含low":1,"含medium":2,"仅high":3};function matches(x){const q=$('q').value.trim().toLowerCase(),f=$('filter').value;if(q&&!x.filename.toLowerCase().includes(q))return false;return f==='all'||(f==='missing'&&x.unresolved_labels.length)||(f==='touch'&&x.c2_boundary==='touch')||(f==='near'&&x.c2_boundary==='near')||(f==='originalmissing'&&(x.original_missing_labels||[]).length)||(f==='low'&&x.low_labels.length)||(f==='medium'&&x.medium_labels.length)||(f==='high'&&x.priority==='仅high')||(f==='todo'&&!saved[x.relative_image]?.verdict)}function refresh(){ids=s.map((x,i)=>i).filter(i=>matches(s[i]));ids.sort((a,b)=>$('sort').value==='name'?s[a].filename.localeCompare(s[b].filename):(rank[s[a].priority]-rank[s[b].priority]||s[a].filename.localeCompare(s[b].filename)));at=Math.min(at,Math.max(0,ids.length-1));render()}function render(){const x=s[ids[at]],b=d.summary.c2_boundary||{};$('heading').textContent=d.summary.title||'第一批侧面补标人工复核';$('summary').textContent=`显示 ${ids.length}/${d.summary.sample_count} 张；C2-C6缺候选 ${d.summary.priority['缺候选']||0}，C2触边 ${b.touch||0}，C2近边 ${b.near||0}，原始缺级提示 ${d.summary.original_missing_count||0}`;if(!x){$('preview').removeAttribute('src');$('link').removeAttribute('href');$('meta').textContent='没有匹配样本';$('position').textContent='0/0';return}$('preview').src=x.preview;$('link').href=x.preview;const cls=x.c2_boundary==='touch'||x.priority==='缺候选'||x.priority==='含low'?'danger':x.c2_boundary==='near'||x.priority==='含medium'?'warn':'ok';$('meta').innerHTML=`<span class="tag ${cls}">${x.priority}</span><span class=tag>C2边界 ${x.c2_boundary||'未统计'} ${x.c2_edge_px??'—'}px</span><span class=tag>补标 ${x.added_labels.join(',')||'无'}</span><span class=tag>未候选 ${x.unresolved_labels.join(',')||'无'}</span><span class=tag>原始缺级提示 ${(x.original_missing_labels||[]).join(',')||'无'}</span><span class=tag>黑边 L/R/T/B ${x.black_bands.join('/')}</span>`;const r=saved[x.relative_image]||{};$('verdict').value=r.verdict||'';$('note').value=r.note||'';$('jump').value=at+1;$('position').textContent=`${at+1}/${ids.length}　${x.relative_image}`;}function move(n){at=Math.max(0,Math.min(ids.length-1,at+n));render()}function save(){const x=s[ids[at]];if(!x)return;saved[x.relative_image]={verdict:$('verdict').value,note:$('note').value,updated_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(saved));if($('filter').value==='todo')refresh();else move(1)}function exportCsv(){const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=[['relative_image','verdict','note','updated_at']];s.forEach(x=>{const r=saved[x.relative_image]||{};rows.push([x.relative_image,r.verdict||'',r.note||'',r.updated_at||''])});const blob=new Blob(['\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\n')],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='仅补C2-C6人工接受拒绝结果.csv';a.click();URL.revokeObjectURL(a.href)}['filter','sort'].forEach(id=>$(id).onchange=()=>{at=0;refresh()});$('q').oninput=()=>{at=0;refresh()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>{at=Math.max(0,Math.min(ids.length-1,Number($('jump').value)-1));render()};$('save').onclick=save;$('export').onclick=exportCsv;document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)};refresh();
</script></body></html>'''


def build_package(
    source: Path,
    pseudolabel_root: Path,
    output: Path,
    limit: int | None = None,
    only_c2c6: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    manifest_path = pseudolabel_root / "manifest.csv"
    queue_path = pseudolabel_root / "review_queue.csv"
    summary_path = pseudolabel_root / "summary.json"
    for path in (source, manifest_path, queue_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest_rows = read_csv(manifest_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        manifest_rows = manifest_rows[:limit]
    queue_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(queue_path):
        queue_by_image[row["relative_image"]].append(row)
    before = snapshot_source(manifest_rows, source)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    samples: list[dict[str, Any]] = []
    try:
        previews = temp / "previews"
        previews.mkdir()
        for index, row in enumerate(manifest_rows, 1):
            relative_image = row["relative_image"]
            source_image = source / relative_image
            candidate_json = pseudolabel_root / row["output_json"]
            if not source_image.is_file() or not candidate_json.is_file():
                raise FileNotFoundError(f"Missing source/candidate for {relative_image}")
            candidate_annotation = load_json(candidate_json)
            source_annotation = load_json(Path(row["source_json"]))
            annotation = (
                compose_c2c6_annotation(source_annotation, candidate_annotation)
                if only_c2c6 else candidate_annotation
            )
            queue = queue_by_image[relative_image]
            unresolved = [
                item["label"] for item in queue
                if item["status"] == "no_candidate"
                and (not only_c2c6 or item["label"] in C2_C6_LABELS)
            ]
            original_missing = missing_baseline_labels(source_annotation) if only_c2c6 else []
            with Image.open(source_image) as dimension_image:
                image_width, image_height = dimension_image.size
            boundary = (
                c2_boundary_info(annotation, image_width, image_height)
                if only_c2c6 else {"state": "not_checked", "margin_px": None, "margin_fraction": None}
            )
            safe_stem = source_image.stem.replace(os.sep, "_")
            preview_name = f"{index:04d}_{safe_stem}.jpg"
            preview_path = previews / preview_name
            rendered = render_preview(
                source_image, annotation, preview_path, index, unresolved,
                only_c2c6=only_c2c6,
                original_missing=original_missing,
                c2_boundary=boundary,
            )
            samples.append({
                "index": index,
                "relative_image": relative_image,
                "filename": source_image.name,
                "source_image": str(source_image),
                "candidate_json": str(candidate_json),
                "preview": f"previews/{preview_name}",
                "preview_sha256": sha256_file(preview_path),
                "unresolved_labels": unresolved,
                "black_bands": [
                    int(row["black_left_px"]), int(row["black_right_px"]),
                    int(row["black_top_px"]), int(row["black_bottom_px"]),
                ],
                **rendered,
            })

        after = snapshot_source(manifest_rows, source)
        if before != after:
            raise RuntimeError("Source PNG/JSON size or timestamp changed during rendering")
        priority_counts = Counter(sample["priority"] for sample in samples)
        boundary_counts = Counter(sample["c2_boundary"] for sample in samples)
        original_missing_count = sum(bool(sample["original_missing_labels"]) for sample in samples)
        review_payload = {
            "schema_version": 1,
            "summary": {
                "sample_count": len(samples),
                "priority": dict(priority_counts),
                "c2_boundary": dict(boundary_counts),
                "original_missing_count": original_missing_count,
                "title": "第一批侧面仅补C2-C6：人工接受/拒绝" if only_c2c6 else "第一批侧面23类补标：完整人工复核",
                "review_key": "lateral-c2c6-only-review-v1" if only_c2c6 else "lateral-pseudolabel-review-v1",
                "source": str(source),
                "pseudolabel_root": str(pseudolabel_root.resolve()),
            },
            "samples": samples,
        }
        (temp / "review_data.js").write_text(
            "window.LATERAL_REVIEW = " + json.dumps(review_payload, ensure_ascii=False) + ";\n",
            encoding="utf-8",
        )
        (temp / "打开此文件逐张人工复核.html").write_text(HTML, encoding="utf-8")
        csv_fields = [
            "index", "relative_image", "filename", "source_image", "candidate_json", "preview",
            "priority", "added_labels", "high_labels", "medium_labels", "low_labels",
            "unresolved_labels", "manual_labels", "width", "height", "neck_crop",
            "original_missing_labels", "c2_boundary", "c2_edge_px", "c2_edge_fraction",
            "black_bands", "preview_sha256", "human_result", "notes",
        ]
        csv_rows = []
        for sample in samples:
            row = dict(sample)
            for field in ("added_labels", "high_labels", "medium_labels", "low_labels", "unresolved_labels", "manual_labels", "original_missing_labels"):
                row[field] = ";".join(row[field])
            row["neck_crop"] = ";".join(map(str, row["neck_crop"]))
            row["black_bands"] = ";".join(map(str, row["black_bands"]))
            row["human_result"] = ""
            row["notes"] = ""
            csv_rows.append(row)
        write_csv(temp / "人工复核索引.csv", csv_rows, csv_fields)
        package_manifest = {
            "schema_version": 1,
            "purpose": "first_batch_lateral_c2c6_only_accept_reject_review" if only_c2c6 else "first_batch_lateral_pseudolabel_human_review",
            "sample_count": len(samples),
            "source": str(source),
            "pseudolabel_root": str(pseudolabel_root.resolve()),
            "source_was_modified": False,
            "pseudolabel_summary_sha256": sha256_file(summary_path),
            "priority": dict(priority_counts),
            "c2_boundary": dict(boundary_counts),
            "original_missing_count": original_missing_count,
            "allowed_pseudo_labels": sorted(C2_C6_LABELS) if only_c2c6 else CLASS_NAMES,
            "files": {sample["preview"]: sample["preview_sha256"] for sample in samples},
        }
        (temp / "manifest.json").write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        title = "第一批侧面仅补C2-C6人工接受/拒绝复核包" if only_c2c6 else "第一批侧面23类补标人工复核包"
        scope_text = (
            "本包只显示模型新增的C2-C6。C7-L5只显示原始人工polygon；即使原始缺级，也不会显示模型补齐结果，也不会自动排除。"
            if only_c2c6 else "本包显示全部缺失层级的模型候选。"
        )
        edit_text = (
            "请先在HTML页面选择接受、拒绝、修改后接受或不确定，并导出CSV。当前旧候选JSON含其它类别伪标签，不应直接作为严格C2-C6数据使用；收到人工结论后再生成最终LabelMe数据。"
            if only_c2c6 else f"实际候选JSON位于 `{(pseudolabel_root / 'labelme_review').resolve()}`，可用LabelMe逐张修改。"
        )
        readme = f"""# {title}

本目录包含 {len(samples)} 张第一批侧面复核图。原始数据没有被修改。

{scope_text}

## 怎么看

1. 双击 `打开此文件逐张人工复核.html`，可按C2-C6缺候选、C2触边、C2近边、原始缺级提示、low、medium、仅high筛选。
2. 每张图左边是完整侧位片，右边是 C2-C7 放大图。
3. 青色是第一批原始人工 polygon；绿色、橙色、红色依次是模型生成的 high、medium、low 候选。
4. high 也不代表真值，所有新框都必须人工确认。红色“没有生成候选”需要人工补画。
5. 页面中的判断保存在当前浏览器本地，可选择接受/拒绝并点击“导出人工复核CSV”备份结果。

## 实际改标位置

{edit_text}

## 优先级

{json.dumps(dict(priority_counts), ensure_ascii=False)}

C2边界提示：{json.dumps(dict(boundary_counts), ensure_ascii=False)}；原始C7-L5缺级提示：{original_missing_count}张。所有提示都只用于筛选，最终由人工决定。

`人工复核索引.csv` 可手工填写最后两列；`manifest.json` 记录全部预览图SHA-256，便于完整性核验。
"""
        (temp / "复核说明.md").write_text(readme, encoding="utf-8")
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return audit_package(pseudolabel_root, output, expected_count=len(manifest_rows))


def audit_package(pseudolabel_root: Path, output: Path, expected_count: int | None = None) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    index_path = output / "人工复核索引.csv"
    js_path = output / "review_data.js"
    html_path = output / "打开此文件逐张人工复核.html"
    readme_path = output / "复核说明.md"
    required = (manifest_path, index_path, js_path, html_path, readme_path, output / "previews")
    errors = [f"missing:{path}" for path in required if not path.exists()]
    if errors:
        return {"status": "failed", "errors": errors}
    package_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_csv(index_path)
    preview_paths = sorted(
        path for path in (output / "previews").glob("*.jpg")
        if not path.name.startswith("._")
    )
    expected = expected_count if expected_count is not None else package_manifest["sample_count"]
    if len(rows) != expected:
        errors.append(f"index_count:{len(rows)}!={expected}")
    if len(preview_paths) != expected:
        errors.append(f"preview_count:{len(preview_paths)}!={expected}")
    listed = {row["preview"] for row in rows}
    actual = {str(path.relative_to(output)) for path in preview_paths}
    if listed != actual:
        errors.append("index_preview_set_mismatch")
    allowed_pseudo_labels = set(package_manifest.get("allowed_pseudo_labels", CLASS_NAMES))
    for row in rows:
        added = {label for label in row.get("added_labels", "").split(";") if label}
        unexpected = sorted(added - allowed_pseudo_labels)
        if unexpected:
            errors.append(f"unexpected_pseudo_labels:{row.get('relative_image')}:{','.join(unexpected)}")
    hash_map = package_manifest.get("files", {})
    for path in preview_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            relative = str(path.relative_to(output))
            if hash_map.get(relative) != sha256_file(path):
                errors.append(f"hash_mismatch:{relative}")
        except Exception as exc:
            errors.append(f"invalid_preview:{path.name}:{exc}")
    pseudo_summary = pseudolabel_root / "summary.json"
    if pseudo_summary.is_file() and package_manifest.get("pseudolabel_summary_sha256") != sha256_file(pseudo_summary):
        errors.append("pseudolabel_summary_hash_mismatch")
    report = {
        "status": "passed" if not errors else "failed",
        "expected_count": expected,
        "index_rows": len(rows),
        "preview_count": len(preview_paths),
        "priority": dict(Counter(row["priority"] for row in rows)),
        "c2_boundary": dict(Counter(row.get("c2_boundary", "") for row in rows if row.get("c2_boundary"))),
        "original_missing_count": sum(bool(row.get("original_missing_labels")) for row in rows),
        "allowed_pseudo_labels": sorted(allowed_pseudo_labels),
        "all_previews_decodable": not any(error.startswith("invalid_preview:") for error in errors),
        "all_hashes_match": not any(error.startswith("hash_mismatch:") for error in errors),
        "errors": errors,
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第一批侧面补标的完整离线人工复核可视化包")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pseudolabel-root", type=Path, default=DEFAULT_PSEUDOLABEL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="仅用于开发抽样；正式复核不要设置")
    parser.add_argument("--only-c2-c6", action="store_true", help="只显示模型新增C2-C6；其它层级仅显示原始人工标注")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (DEFAULT_C2C6_OUTPUT if args.only_c2_c6 else DEFAULT_OUTPUT)
    report = (
        audit_package(args.pseudolabel_root, output)
        if args.audit_only
        else build_package(
            args.source, args.pseudolabel_root, output, args.limit,
            only_c2c6=args.only_c2_c6,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
