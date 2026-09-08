#!/usr/bin/env python3
"""Build an accept/reject review package for adding C2-C6 to yolo_corner.

The existing 18-class YOLO Pose labels are the immutable baseline.  Existing
first-batch pseudo-labels are reused by filename when available; only uncovered
images are sent through the returned 23-class lateral model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = PROJECT_ROOT / "datasets/yolo_corner"
DEFAULT_RAW_SOURCE = Path("/Volumes/E/spine_data/LAT202511")
DEFAULT_PSEUDOLABEL_ROOT = PROJECT_ROOT / "datasets/lateral_first_batch_pseudolabel_23cls"
DEFAULT_MODEL = PROJECT_ROOT / "3-model_training/runs/pose/yolo11l_lateral_23cls_best/weights/best.pt"
DEFAULT_OUTPUT = Path("/Volumes/E/spine_data/yolo_corner_仅补C2-C6_人工接受拒绝复核包_368张_20260908")
SPLITS = ("train", "val", "test")
C2_C6 = tuple(f"C{i}" for i in range(2, 7))
BASELINE_NAMES = {
    0: "C7", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "T1", 7: "T2", 8: "T3", 9: "T4", 10: "T5", 11: "T6",
    12: "T7", 13: "T8", 14: "T9", 15: "T10", 16: "T11", 17: "T12",
}
BASELINE_ORDER = ["C7", *(f"T{i}" for i in range(1, 13)), *(f"L{i}" for i in range(1, 6))]


def load_sibling(module_name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RENDER = load_sibling("lateral_review_renderer_shared", "render_lateral_pseudolabel_review.py")
PSEUDO = load_sibling("lateral_pseudolabel_shared", "pseudo_label_first_lateral.py")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_annotation(label_path: Path, image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
    shapes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"{label_path}:{line_number}: expected 17 fields, got {len(fields)}")
        class_id = int(fields[0])
        if class_id not in BASELINE_NAMES:
            raise ValueError(f"{label_path}:{line_number}: unexpected class {class_id}")
        if class_id in seen:
            raise ValueError(f"{label_path}:{line_number}: duplicate class {class_id}")
        seen.add(class_id)
        points = []
        for offset in range(5, 17, 3):
            x, y, visibility = float(fields[offset]), float(fields[offset + 1]), int(fields[offset + 2])
            if visibility <= 0:
                raise ValueError(f"{label_path}:{line_number}: invisible baseline keypoint")
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"{label_path}:{line_number}: keypoint out of range")
            points.append([x * width, y * height])
        shapes.append({
            "label": BASELINE_NAMES[class_id],
            "points": points,
            "group_id": None,
            "description": "YOLO_CORNER_EXISTING_18CLS_BASELINE",
            "shape_type": "polygon",
            "flags": {"yolo_corner_baseline": True},
            "mask": None,
        })
    return {
        "version": "yolo_corner_review",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def discover_dataset(dataset: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for split in SPLITS:
        image_dir, label_dir = dataset / "images" / split, dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split}")
        images = sorted(path for path in image_dir.glob("*.png") if not path.name.startswith("._"))
        labels = sorted(path for path in label_dir.glob("*.txt") if not path.name.startswith("._"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"Image/label stem mismatch in {split}")
        for image_path in images:
            if image_path.name in seen_names:
                raise ValueError(f"Duplicate basename across splits: {image_path.name}")
            seen_names.add(image_path.name)
            records.append({
                "split": split,
                "image": image_path,
                "label": label_dir / f"{image_path.stem}.txt",
                "relative_image": f"images/{split}/{image_path.name}",
            })
    return records


def source_index(raw_source: Path) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = defaultdict(list)
    if raw_source.is_dir():
        for path in raw_source.rglob("*.png"):
            if not path.name.startswith("._"):
                output[path.name].append(path)
    return dict(output)


def duplicate_groups(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for record in records:
        by_size[record["image"].stat().st_size].append(record["image"])
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for paths in by_size.values():
        if len(paths) > 1:
            for path in paths:
                by_hash[sha256_file(path)].append(path)
    result: dict[str, str] = {}
    group_number = 0
    for digest, paths in sorted(by_hash.items()):
        if len(paths) < 2:
            continue
        group_number += 1
        group_id = f"duplicate_{group_number:02d}_{digest[:10]}"
        for path in paths:
            result[path.name] = group_id
    return result


def existing_candidate_index(pseudolabel_root: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(pseudolabel_root / "manifest.csv")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = Path(row["relative_image"]).name
        if name in output:
            raise ValueError(f"Duplicate pseudo-label basename: {name}")
        output[name] = {
            "json": pseudolabel_root / row["output_json"],
            "black_bands": [
                int(row["black_left_px"]), int(row["black_right_px"]),
                int(row["black_top_px"]), int(row["black_bottom_px"]),
            ],
        }
    return output


def infer_annotation(
    image_path: Path,
    baseline: dict[str, Any],
    model_state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[int]]:
    import cv2
    from ultralytics import YOLO

    if "model" not in model_state:
        if not args.model.is_file():
            raise FileNotFoundError(args.model)
        model_state["model"] = YOLO(str(args.model))
        model_state["names"] = PSEUDO.validate_model(model_state["model"])
        model_state["device"] = PSEUDO.resolve_device(args.device)
        model_state["model_hash"] = sha256_file(args.model)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {image_path}")
    height, width = image.shape[:2]
    bands = PSEUDO.detect_black_bands(image)
    views = PSEUDO.plan_views(width, height, bands, PSEUDO.annotation_points(baseline))
    sources = [image[view.y0:view.y1, view.x0:view.x1] for view in views]
    results = model_state["model"].predict(
        # Values below candidate_conf are discarded immediately and only make
        # CPU NMS slower, so threshold before NMS for the ten uncovered images.
        sources, conf=args.candidate_conf, iou=args.iou, imgsz=args.imgsz,
        device=model_state["device"], max_det=args.max_det, verbose=False,
    )
    candidates = PSEUDO.clip_candidates(
        (
            candidate
            for result, view in zip(results, views)
            for candidate in PSEUDO.prediction_candidates(result, model_state["names"], view)
            if candidate.box_conf >= args.candidate_conf
        ),
        width,
        height,
    )
    anchors = PSEUDO.anchor_scores(baseline, candidates, views)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.label].append(candidate)
    selected = {}
    for label in PSEUDO.CLASS_NAMES:
        choice = PSEUDO.choose_candidate(grouped.get(label, []), anchors)
        if choice is not None:
            selected[label] = choice
    augmented, _, _ = PSEUDO.append_missing_shapes(
        baseline, selected, anchors, model_state["model_hash"]
    )
    combined = RENDER.compose_c2c6_annotation(baseline, augmented)
    return combined, [bands["left"], bands["right"], bands["top"], bands["bottom"]]


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>yolo_corner仅补C2-C6人工复核</title><style>
*{box-sizing:border-box}body{margin:0;background:#11151b;color:#eef2f7;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{position:sticky;top:0;z-index:3;background:#1b222cf4;padding:14px 20px;box-shadow:0 2px 12px #0009}h1{font-size:22px;margin:0 0 8px}.summary{color:#b9c4d2;margin:7px 0}.controls{display:flex;flex-wrap:wrap;gap:8px}button,input,select,textarea{background:#10161e;color:#fff;border:1px solid #445267;border-radius:7px;padding:8px 10px}.viewer{max-width:1840px;margin:16px auto;padding:0 14px}.card{background:#1a222d;border:1px solid #344151;border-radius:10px;padding:13px}.card img{display:block;width:100%;max-height:calc(100vh - 315px);object-fit:contain;background:#080b0f}.meta{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.tag{background:#303b49;border-radius:13px;padding:4px 9px}.danger{background:#822f38}.warn{background:#76521d}.ok{background:#245f3c}.review{display:grid;grid-template-columns:1fr;gap:8px}.decision-buttons{display:grid;grid-template-columns:1fr 1fr auto auto;gap:9px}.decision-buttons button{font-size:17px;font-weight:700;min-height:48px;cursor:pointer}.decision-buttons .accept{background:#17633a;border-color:#39c875}.decision-buttons .reject{background:#7a2530;border-color:#ef5867}.decision-buttons .secondary{font-size:14px;background:#303b49}.decision-buttons button.active{outline:3px solid #fff;box-shadow:0 0 0 2px #111}.decision-status{color:#b9c4d2}.position{color:#b9c4d2;margin-top:8px}.legend{font-size:13px;color:#b9c4d2}.green{color:#49dc78}.orange{color:#ffb342}.red{color:#ff6262}.cyan{color:#24dcef}kbd{font-size:12px;background:#1118;border:1px solid #ffffff66;border-radius:4px;padding:1px 5px}@media(max-width:800px){.decision-buttons{grid-template-columns:1fr 1fr}.decision-buttons .secondary{font-size:13px}}</style></head><body><header><h1>yolo_corner仅补C2-C6：人工接受/拒绝</h1><div class="summary" id="summary"></div><div class="legend"><span class="cyan">青=yolo_corner现有18类四角点</span>；<span class="green">绿=high</span>；<span class="orange">橙=medium</span>；<span class="red">红=low/缺候选/触边</span>。所有风险只提示，由人工决定。</div><div class="controls"><input id="q" placeholder="搜索文件名"><select id="filter"><option value="all">全部</option><option value="source">旧标注错配风险</option><option value="leak">患者跨split</option><option value="duplicate">精确重复图</option><option value="baseline">基线缺级</option><option value="missing">C2-C6缺候选</option><option value="touch">C2触边</option><option value="near">C2近边</option><option value="low">含low</option><option value="medium">含medium</option><option value="high">仅high</option><option value="train">train</option><option value="val">val</option><option value="test">test</option><option value="todo">未判断</option></select><button id="prev">← 上一张</button><button id="next">下一张 →</button><input id="jump" type="number" min="1" style="width:88px"><button id="export">导出人工复核CSV</button></div></header><main class="viewer"><section class="card"><a id="link" target="_blank"><img id="preview"></a><div class="meta" id="meta"></div><div class="review"><div class="decision-buttons"><button id="accept" class="accept" data-verdict="accepted">✓ 接受并下一张 <kbd>A</kbd></button><button id="reject" class="reject" data-verdict="rejected">✕ 拒绝并下一张 <kbd>R</kbd></button><button class="secondary" data-verdict="accept_after_edit">修改后接受</button><button class="secondary" data-verdict="uncertain">不确定</button></div><span class="decision-status" id="decisionStatus">当前：未判断</span><textarea id="note" rows="2" placeholder="人工备注（点击任一判断按钮时一并保存）"></textarea><select id="verdict" hidden><option value="">未判断</option><option value="accepted">接受</option><option value="rejected">拒绝</option><option value="accept_after_edit">修改后接受</option><option value="uncertain">不确定</option></select><button id="save" hidden>保存并下一张</button></div><div class="position" id="position"></div></section></main><script src="review_data.js"></script><script>
const $=id=>document.getElementById(id),d=window.YOLO_CORNER_REVIEW,s=d.samples,key='yolo-corner-c2c6-review-v1',saved=JSON.parse(localStorage.getItem(key)||'{}'),verdictText={accepted:'接受',rejected:'拒绝',accept_after_edit:'修改后接受',uncertain:'不确定'};let ids=[],at=0;const rank={source_mismatch:0,patient_leak:1,duplicate:2,baseline_missing:3,candidate_risk:4,safe:5};function matches(x){const q=$('q').value.trim().toLowerCase(),f=$('filter').value;if(q&&!x.filename.toLowerCase().includes(q))return false;return f==='all'||(f==='source'&&x.source_mismatch)||(f==='leak'&&x.patient_leak)||(f==='duplicate'&&x.duplicate_group)||(f==='baseline'&&x.baseline_missing_labels.length)||(f==='missing'&&x.unresolved_labels.length)||(f==='touch'&&x.c2_boundary==='touch')||(f==='near'&&x.c2_boundary==='near')||(f==='low'&&x.low_labels.length)||(f==='medium'&&x.medium_labels.length)||(f==='high'&&x.priority==='仅high')||f===x.split||(f==='todo'&&!saved[x.relative_image]?.verdict)}function risk(x){if(x.source_mismatch)return 0;if(x.patient_leak)return 1;if(x.duplicate_group)return 2;if(x.baseline_missing_labels.length)return 3;if(x.unresolved_labels.length||x.low_labels.length||x.c2_boundary==='touch')return 4;return 5}function refresh(){ids=s.map((x,i)=>i).filter(i=>matches(s[i])).sort((a,b)=>risk(s[a])-risk(s[b])||s[a].filename.localeCompare(s[b].filename));at=Math.min(at,Math.max(0,ids.length-1));render()}function render(){const x=s[ids[at]],z=d.summary,done=s.filter(x=>saved[x.relative_image]?.verdict).length;$('summary').textContent=`已判断 ${done}/${z.sample_count}；当前筛选 ${ids.length} 张；错配风险 ${z.source_mismatch_count}，跨split ${z.patient_leak_count}，重复 ${z.duplicate_count}，C2-C6缺候选 ${z.missing_candidate_count}`;if(!x){$('preview').removeAttribute('src');$('meta').textContent='没有匹配样本';$('position').textContent='0/0';return}$('preview').src=x.preview;$('link').href=x.preview;const bad=x.source_mismatch||x.patient_leak||x.duplicate_group||x.unresolved_labels.length||x.low_labels.length||x.c2_boundary==='touch';$('meta').innerHTML=`<span class="tag ${bad?'danger':'ok'}">${x.split}</span><span class=tag>来源 ${x.candidate_origin}</span><span class=tag>C2边界 ${x.c2_boundary}</span><span class=tag>补标 ${x.added_labels.join(',')||'无'}</span><span class=tag>无候选 ${x.unresolved_labels.join(',')||'无'}</span><span class=tag>基线缺级 ${x.baseline_missing_labels.join(',')||'无'}</span><span class=tag>错配风险 ${x.source_mismatch?'是':'否'}</span><span class=tag>跨split ${x.patient_leak?'是':'否'}</span><span class=tag>重复 ${x.duplicate_group||'否'}</span>`;const r=saved[x.relative_image]||{};$('verdict').value=r.verdict||'';$('note').value=r.note||'';$('decisionStatus').textContent=`当前：${verdictText[r.verdict]||'未判断'}`;document.querySelectorAll('[data-verdict]').forEach(b=>b.classList.toggle('active',b.dataset.verdict===(r.verdict||'')));$('jump').value=at+1;$('position').textContent=`${at+1}/${ids.length}　${x.relative_image}`;}function move(n){at=Math.max(0,Math.min(ids.length-1,at+n));render()}function save(){const x=s[ids[at]];if(!x)return;saved[x.relative_image]={verdict:$('verdict').value,note:$('note').value,updated_at:new Date().toISOString()};localStorage.setItem(key,JSON.stringify(saved));if($('filter').value==='todo')refresh();else move(1)}function saveVerdict(verdict){$('verdict').value=verdict;save()}function exportCsv(){const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=[['relative_image','verdict','note','updated_at']];s.forEach(x=>{const r=saved[x.relative_image]||{};rows.push([x.relative_image,r.verdict||'',r.note||'',r.updated_at||''])});const blob=new Blob(['\ufeff'+rows.map(r=>r.map(esc).join(',')).join('\n')],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='yolo_corner仅补C2-C6人工接受拒绝结果.csv';a.click();URL.revokeObjectURL(a.href)}$('filter').onchange=()=>{at=0;refresh()};$('q').oninput=()=>{at=0;refresh()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>{at=Math.max(0,Math.min(ids.length-1,Number($('jump').value)-1));render()};$('save').onclick=save;document.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>saveVerdict(b.dataset.verdict));document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key.toLowerCase()==='a')saveVerdict('accepted');if(e.key.toLowerCase()==='r')saveVerdict('rejected')};$('export').onclick=exportCsv;refresh();
</script></body></html>'''


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {output}")
    records = discover_dataset(dataset)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No yolo_corner samples selected")
    raw_by_name = source_index(args.raw_source)
    pseudo_by_name = existing_candidate_index(args.pseudolabel_root)
    duplicates = duplicate_groups(records)

    for record in records:
        paths = raw_by_name.get(record["image"].name, [])
        record["raw_path"] = paths[0] if len(paths) == 1 else None
        record["patient_id"] = paths[0].parent.name if len(paths) == 1 else "unknown"
        record["source_mismatch"] = len(paths) != 1 or not paths[0].with_suffix(".json").is_file()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        patient_splits[record["patient_id"]].add(record["split"])
    leaking_patients = {patient for patient, splits in patient_splits.items() if len(splits) > 1}

    snapshot = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for record in records for path in (record["image"], record["label"])
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    samples: list[dict[str, Any]] = []
    model_state: dict[str, Any] = {}
    try:
        previews = staging / "previews"
        previews.mkdir()
        for index, record in enumerate(records, 1):
            image_path, label_path = record["image"], record["label"]
            baseline = parse_yolo_annotation(label_path, image_path)
            existing = pseudo_by_name.get(image_path.name)
            if existing is not None:
                candidate = RENDER.load_json(existing["json"])
                if (int(candidate.get("imageWidth") or 0), int(candidate.get("imageHeight") or 0)) != (
                    baseline["imageWidth"], baseline["imageHeight"]
                ):
                    raise ValueError(f"Candidate dimensions differ for {image_path.name}")
                annotation = RENDER.compose_c2c6_annotation(baseline, candidate)
                black_bands = existing["black_bands"]
                candidate_origin = "复用既有预测"
            else:
                annotation, black_bands = infer_annotation(image_path, baseline, model_state, args)
                candidate_origin = "本次补推理"
                print(f"inferred missing candidate [{index}/{len(records)}] {record['relative_image']}")

            present_pseudo = {
                str(shape.get("label") or "").upper()
                for shape in annotation["shapes"] if RENDER.is_pseudo(shape)
            }
            unresolved = [label for label in C2_C6 if label not in present_pseudo]
            baseline_missing = RENDER.missing_baseline_labels(baseline)
            boundary = RENDER.c2_boundary_info(
                annotation, baseline["imageWidth"], baseline["imageHeight"]
            )
            warning_parts = []
            if record["source_mismatch"]:
                warning_parts.append("旧标签错配风险")
            patient_leak = record["patient_id"] in leaking_patients
            if patient_leak:
                warning_parts.append("患者跨split")
            duplicate_group = duplicates.get(image_path.name, "")
            if duplicate_group:
                warning_parts.append("精确重复图")
            preview_name = f"{index:04d}_{record['split']}_{image_path.stem}.jpg"
            rendered = RENDER.render_preview(
                image_path, annotation, previews / preview_name, index, unresolved,
                only_c2c6=True,
                original_missing=baseline_missing,
                c2_boundary=boundary,
                baseline_name="yolo_corner现有18类四角点",
                sample_warning="；".join(warning_parts),
            )
            samples.append({
                "index": index,
                "relative_image": record["relative_image"],
                "filename": image_path.name,
                "image": str(image_path),
                "label": str(label_path),
                "split": record["split"],
                "patient_id": record["patient_id"],
                "source_mismatch": record["source_mismatch"],
                "patient_leak": patient_leak,
                "duplicate_group": duplicate_group,
                "candidate_origin": candidate_origin,
                "baseline_missing_labels": baseline_missing,
                "unresolved_labels": unresolved,
                "black_bands": black_bands,
                "preview": f"previews/{preview_name}",
                **rendered,
            })

        after = {
            str(path): (path.stat().st_size, path.stat().st_mtime_ns)
            for record in records for path in (record["image"], record["label"])
        }
        if snapshot != after:
            raise RuntimeError("yolo_corner source changed during rendering")
        for sample in samples:
            sample["preview_sha256"] = sha256_file(staging / sample["preview"])
        summary = {
            "sample_count": len(samples),
            "split": dict(Counter(sample["split"] for sample in samples)),
            "candidate_origin": dict(Counter(sample["candidate_origin"] for sample in samples)),
            "priority": dict(Counter(sample["priority"] for sample in samples)),
            "c2_boundary": dict(Counter(sample["c2_boundary"] for sample in samples)),
            "source_mismatch_count": sum(sample["source_mismatch"] for sample in samples),
            "patient_leak_count": sum(sample["patient_leak"] for sample in samples),
            "duplicate_count": sum(bool(sample["duplicate_group"]) for sample in samples),
            "baseline_missing_count": sum(bool(sample["baseline_missing_labels"]) for sample in samples),
            "missing_candidate_count": sum(bool(sample["unresolved_labels"]) for sample in samples),
        }
        payload = {"schema_version": 1, "summary": summary, "samples": samples}
        (staging / "review_data.js").write_text(
            "window.YOLO_CORNER_REVIEW = " + json.dumps(payload, ensure_ascii=False) + ";\n",
            encoding="utf-8",
        )
        (staging / "打开此文件逐张人工复核.html").write_text(HTML, encoding="utf-8")
        fields = [
            "index", "relative_image", "filename", "image", "label", "split", "patient_id", "candidate_origin",
            "priority", "added_labels", "high_labels", "medium_labels", "low_labels",
            "unresolved_labels", "baseline_missing_labels", "c2_boundary", "c2_edge_px",
            "source_mismatch", "patient_leak", "duplicate_group", "black_bands", "preview",
            "preview_sha256", "human_result", "notes",
        ]
        csv_rows = []
        for sample in samples:
            row = dict(sample)
            for field in ("added_labels", "high_labels", "medium_labels", "low_labels", "unresolved_labels", "baseline_missing_labels"):
                row[field] = ";".join(row[field])
            row["black_bands"] = ";".join(map(str, row["black_bands"]))
            row["human_result"] = ""
            row["notes"] = ""
            csv_rows.append(row)
        write_csv(staging / "人工复核索引.csv", csv_rows, fields)
        manifest = {
            "schema_version": 1,
            "purpose": "yolo_corner_add_c2_c6_accept_reject_review",
            "dataset": str(dataset),
            "dataset_modified": False,
            "allowed_pseudo_labels": list(C2_C6),
            "model": str(args.model),
            "model_sha256": model_state.get("model_hash") or sha256_file(args.model),
            "summary": summary,
            "files": {sample["preview"]: sample["preview_sha256"] for sample in samples},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "复核说明.md").write_text(
            f"""# yolo_corner仅补C2-C6人工接受/拒绝复核包

本包严格以 `{dataset}` 的 {len(samples)} 张图及现有18类YOLO Pose四角点标签为基线；原数据集未修改。

- 青色：yolo_corner现有C7-L5四角点标签。
- 绿色/橙色/红色：仅C2-C6模型候选，high也必须人工确认。
- 旧标签错配、患者跨split、精确重复、基线缺级、C2触边/近边都只提示，不自动排除。
- 双击 `打开此文件逐张人工复核.html`，逐张选择接受、拒绝、修改后接受或不确定，最后导出CSV。
- 当前包只用于判断；收到人工结果后再另建23类数据集并按患者重拆，不能直接覆盖yolo_corner。

统计：{json.dumps(summary, ensure_ascii=False)}
""",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit_package(output)


def audit_package(output: Path) -> dict[str, Any]:
    required = [
        output / "manifest.json", output / "人工复核索引.csv", output / "review_data.js",
        output / "打开此文件逐张人工复核.html", output / "复核说明.md", output / "previews",
    ]
    errors = [f"missing:{path.name}" for path in required if not path.exists()]
    if errors:
        return {"status": "failed", "errors": errors}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(output / "人工复核索引.csv")
    previews = sorted(
        path for path in (output / "previews").glob("*.jpg") if not path.name.startswith("._")
    )
    expected = manifest["summary"]["sample_count"]
    if len(rows) != expected:
        errors.append(f"index_count:{len(rows)}!={expected}")
    if len(previews) != expected:
        errors.append(f"preview_count:{len(previews)}!={expected}")
    if len({row["relative_image"] for row in rows}) != len(rows):
        errors.append("duplicate_relative_image")
    if {row["preview"] for row in rows} != {str(path.relative_to(output)) for path in previews}:
        errors.append("preview_set_mismatch")
    allowed = set(manifest["allowed_pseudo_labels"])
    for row in rows:
        if set(filter(None, row["added_labels"].split(";"))) - allowed:
            errors.append(f"unexpected_pseudo_label:{row['relative_image']}")
        if not Path(row["image"]).is_file() or not Path(row["label"]).is_file():
            errors.append(f"missing_source:{row['relative_image']}")
    for path in previews:
        try:
            with Image.open(path) as image:
                image.verify()
            relative = str(path.relative_to(output))
            if manifest["files"].get(relative) != sha256_file(path):
                errors.append(f"hash_mismatch:{relative}")
        except Exception as exc:
            errors.append(f"invalid_preview:{path.name}:{exc}")
    report = {
        "status": "passed" if not errors else "failed",
        "expected_count": expected,
        "index_rows": len(rows),
        "preview_count": len(previews),
        "split": dict(Counter(row["split"] for row in rows)),
        "candidate_origin": dict(Counter(row["candidate_origin"] for row in rows)),
        "priority": dict(Counter(row["priority"] for row in rows)),
        "c2_boundary": dict(Counter(row["c2_boundary"] for row in rows)),
        "source_mismatch_count": sum(row["source_mismatch"].lower() == "true" for row in rows),
        "patient_leak_count": sum(row["patient_leak"].lower() == "true" for row in rows),
        "duplicate_count": sum(bool(row["duplicate_group"]) for row in rows),
        "baseline_missing_count": sum(bool(row["baseline_missing_labels"]) for row in rows),
        "missing_candidate_count": sum(bool(row["unresolved_labels"]) for row in rows),
        "allowed_pseudo_labels": sorted(allowed),
        "all_previews_decodable": not any(error.startswith("invalid_preview:") for error in errors),
        "all_hashes_match": not any(error.startswith("hash_mismatch:") for error in errors),
        "errors": errors,
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-source", type=Path, default=DEFAULT_RAW_SOURCE)
    parser.add_argument("--pseudolabel-root", type=Path, default=DEFAULT_PSEUDOLABEL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--predict-conf", type=float, default=0.01)
    parser.add_argument("--candidate-conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.predict_conf <= args.candidate_conf <= 1:
        parser.error("Require 0 <= predict-conf <= candidate-conf <= 1")
    return args


def main() -> None:
    args = parse_args()
    report = audit_package(args.output) if args.audit_only else build_package(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
