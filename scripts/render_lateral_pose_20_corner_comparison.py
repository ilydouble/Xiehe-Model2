#!/usr/bin/env python3
"""Render old-vs-new corner comparisons for the reviewed 20-class dataset.

The script replays the legacy sequential farthest-point algorithm on every
included first-batch dense polygon, ranks its most triangle-like outputs, and
renders the worst unique image/vertebra pairs beside the rebuilt labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_NAMES = ["C2", "C7", *(f"T{i}" for i in range(1, 14)), *(f"L{i}" for i in range(1, 6))]
RAW_DENSE_LABELS = set(CLASS_NAMES) - {"C2"}
BACKGROUND = (16, 19, 24)
PANEL_BACKGROUND = (5, 7, 10)
WHITE = (245, 247, 250)
MUTED = (180, 190, 203)
CONTOUR = (0, 220, 245)
OLD = (255, 145, 40)
NEW = (48, 218, 105)
POINT = (255, 235, 80)


def choose_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def legacy_farthest_four(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Dependency-free replay of the original sequential farthest-point code."""
    converted = [(float(point[0]), float(point[1])) for point in points]
    center = (
        sum(point[0] for point in converted) / len(converted),
        sum(point[1] for point in converted) / len(converted),
    )
    distances = [math.dist(point, center) for point in converted]
    angles = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in converted]
    selected_indices: list[int] = []
    used: set[int] = set()
    first = max(range(len(converted)), key=lambda index: distances[index])
    selected_indices.append(first)
    used.add(first)
    current_angle = angles[first]
    for _ in range(3):
        target_angle = current_angle + math.pi / 2
        valid: list[int] = []
        for index, angle in enumerate(angles):
            difference = abs(angle - target_angle)
            difference = min(difference, 2 * math.pi - difference)
            if difference <= math.pi / 4 and index not in used:
                valid.append(index)
        candidates = valid or [index for index in range(len(converted)) if index not in used]
        chosen = max(candidates, key=lambda index: distances[index])
        selected_indices.append(chosen)
        used.add(chosen)
        current_angle = angles[chosen]
    selected = [converted[index] for index in selected_indices]
    return sorted(selected, key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))


def edge_ratio(points: Sequence[tuple[float, float]]) -> float:
    lengths = [math.dist(points[index], points[(index + 1) % 4]) for index in range(4)]
    return min(lengths) / max(lengths) if max(lengths) else 0.0


def parse_new_labels(path: Path, width: int, height: int) -> dict[str, list[tuple[float, float]]]:
    labels: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        class_id = int(parts[0])
        points = [
            (float(parts[index]) * width, float(parts[index + 1]) * height)
            for index in (5, 8, 11, 14)
        ]
        labels[CLASS_NAMES[class_id]] = points
    return labels


def dense_shapes(path: Path) -> dict[str, list[tuple[float, float]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    result: dict[str, list[tuple[float, float]]] = {}
    for shape in data.get("shapes", []):
        label = str(shape.get("label") or "").strip().upper()
        if label in RAW_DENSE_LABELS and shape.get("shape_type") == "polygon":
            result[label] = [(float(x), float(y)) for x, y in shape["points"]]
    return result


def padded_crop(points: Sequence[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    padding = max(50.0, span * 0.65)
    return (
        max(0, int(min(xs) - padding)),
        max(0, int(min(ys) - padding)),
        min(width, int(max(xs) + padding)),
        min(height, int(max(ys) + padding)),
    )


def fit_panel(image: Image.Image, crop: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[Image.Image, float, float, float]:
    cropped = image.crop(crop)
    scale = min(size[0] / cropped.width, size[1] / cropped.height)
    resized = cropped.resize((max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))))
    panel = Image.new("RGB", size, PANEL_BACKGROUND)
    left = (size[0] - resized.width) / 2
    top = (size[1] - resized.height) / 2
    panel.paste(resized, (round(left), round(top)))
    return panel, scale, left - crop[0] * scale, top - crop[1] * scale


def transform(points: Sequence[tuple[float, float]], scale: float, offset_x: float, offset_y: float) -> list[tuple[float, float]]:
    return [(x * scale + offset_x, y * scale + offset_y) for x, y in points]


def draw_geometry(
    panel: Image.Image,
    contour: Sequence[tuple[float, float]],
    corners: Sequence[tuple[float, float]],
    scale: float,
    offset_x: float,
    offset_y: float,
    color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(panel)
    contour_panel = transform(contour, scale, offset_x, offset_y)
    corner_panel = transform(corners, scale, offset_x, offset_y)
    draw.line(contour_panel + [contour_panel[0]], fill=CONTOUR, width=3)
    draw.line(corner_panel + [corner_panel[0]], fill=color, width=6)
    font = choose_font(22)
    for number, point in enumerate(corner_panel, 1):
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=POINT)
        draw.text((point[0] + 10, point[1] - 12), str(number), fill=WHITE, font=font)


def render_comparison(record: dict[str, Any], output: Path) -> None:
    image = Image.open(record["image_path"]).convert("RGB")
    crop = padded_crop(record["contour"], image.width, image.height)
    canvas = Image.new("RGB", (1680, 900), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, small_font = choose_font(32), choose_font(24), choose_font(20)
    draw.text((30, 18), f"#{record['rank']:02d}  {record['label']}  {record['source_stem']}", fill=WHITE, font=title_font)
    if record["sample_type"] == "second_direct":
        subtitle = f"second-batch direct four-point label  |  edge ratio={record['new_edge_ratio']:.4f}"
        left_crop = (0, 0, image.width, image.height)
        left_title = "SECOND BATCH: full image + final green corners"
        right_title = "SECOND BATCH: detail + final green corners"
    else:
        subtitle = f"legacy unique={record['old_unique']}  edge ratio={record['old_edge_ratio']:.4f}    |    rebuilt edge ratio={record['new_edge_ratio']:.4f}"
        left_crop = crop
        left_title = "LEGACY: cyan contour + orange corners"
        right_title = "REBUILT: cyan contour + green corners"
    draw.text((30, 60), subtitle, fill=MUTED, font=text_font)
    left, ls, lx, ly = fit_panel(image, left_crop, (800, 730))
    right, rs, rx, ry = fit_panel(image, crop, (800, 730))
    if record["sample_type"] == "second_direct":
        draw_geometry(left, record["contour"], record["new_points"], ls, lx, ly, NEW)
    else:
        draw_geometry(left, record["contour"], record["old_points"], ls, lx, ly, OLD)
    draw_geometry(right, record["contour"], record["new_points"], rs, rx, ry, NEW)
    canvas.paste(left, (25, 140))
    canvas.paste(right, (855, 140))
    draw.text((25, 106), left_title, fill=NEW if record["sample_type"] == "second_direct" else OLD, font=small_font)
    draw.text((855, 106), right_title, fill=NEW, font=small_font)
    canvas.save(output, quality=94, subsampling=0)


def collect_ranked_records(dataset_root: Path, raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest = [row for row in read_csv(dataset_root / "manifest.csv") if row["source_batch"] == "first"]
    records: list[dict[str, Any]] = []
    for row in manifest:
        image_path = dataset_root / row["image"]
        width, height = Image.open(image_path).size
        patient = row["patient_id"].split(":", 1)[1]
        raw_json = raw_root / patient / f"{row['source_stem']}.json"
        shapes = dense_shapes(raw_json)
        new_labels = parse_new_labels(dataset_root / row["label"], width, height)
        for label, contour in shapes.items():
            if label not in new_labels:
                continue
            old_points = legacy_farthest_four(contour)
            new_points = new_labels[label]
            records.append({
                "sample_type": "first_risk",
                "source_stem": row["source_stem"],
                "patient_id": row["patient_id"],
                "split": row["split"],
                "label": label,
                "image_path": image_path,
                "contour": contour,
                "old_points": old_points,
                "new_points": new_points,
                "old_unique": len(set(old_points)),
                "old_edge_ratio": edge_ratio(old_points),
                "new_edge_ratio": edge_ratio(new_points),
            })
    records.sort(key=lambda record: (record["old_unique"], record["old_edge_ratio"], record["source_stem"], record["label"]))
    summary = {
        "objects_scanned": len(records),
        "legacy_less_than_4_unique": sum(record["old_unique"] < 4 for record in records),
        "legacy_edge_ratio_below_0_05": sum(record["old_edge_ratio"] < 0.05 for record in records),
        "legacy_edge_ratio_below_0_10": sum(record["old_edge_ratio"] < 0.10 for record in records),
        "legacy_edge_ratio_below_0_15": sum(record["old_edge_ratio"] < 0.15 for record in records),
        "rebuilt_edge_ratio_below_0_05": sum(record["new_edge_ratio"] < 0.05 for record in records),
    }
    return records, summary


def collect_second_controls(dataset_root: Path, count: int) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(dataset_root / "manifest.csv") if row["source_batch"] == "second"]
    if len(rows) < count:
        raise ValueError("Not enough second-batch samples")
    positions = [round(index * (len(rows) - 1) / max(1, count - 1)) for index in range(count)]
    preferred_labels = ["C2", "C7", "T6", "L5"]
    records: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        row = rows[position]
        image_path = dataset_root / row["image"]
        width, height = Image.open(image_path).size
        labels = parse_new_labels(dataset_root / row["label"], width, height)
        preferred = preferred_labels[index % len(preferred_labels)]
        label = preferred if preferred in labels else sorted(labels)[0]
        points = labels[label]
        records.append({
            "sample_type": "second_direct",
            "source_stem": row["source_stem"],
            "patient_id": row["patient_id"],
            "split": row["split"],
            "label": label,
            "image_path": image_path,
            "contour": points,
            "old_points": points,
            "new_points": points,
            "old_unique": 4,
            "old_edge_ratio": edge_ratio(points),
            "new_edge_ratio": edge_ratio(points),
        })
    return records


def select_unique_images(records: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    stems: set[str] = set()
    for record in records:
        if record["source_stem"] in stems:
            continue
        selected.append(dict(record))
        stems.add(record["source_stem"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Could only select {len(selected)} unique images")
    return selected


def write_html(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    cards = "\n".join(
        f'<a class="card" href="previews/{row["preview"]}"><img src="previews/{row["preview"]}"><span>#{row["rank"]:02d} {row["sample_type"]} {row["label"]} old={float(row["old_edge_ratio"]):.4f}</span></a>'
        for row in rows
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Corner comparison</title>"
        "<style>body{background:#11151b;color:#eef;font:16px Arial;margin:20px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{color:#eef;text-decoration:none}.card img{width:100%;border:1px solid #445}.card span{display:block;padding:6px}</style>"
        f"<h1>Legacy triangle-risk vs rebuilt four corners</h1><p>Cyan=dense contour; orange=legacy; green=rebuilt. Click to open full resolution.</p><div class='grid'>{cards}</div>",
        encoding="utf-8",
    )


def make_montage(items: Sequence[Image.Image], output: Path) -> None:
    columns = 2
    rows = math.ceil(len(items) / columns)
    montage = Image.new("RGB", (columns * 600, rows * 322), BACKGROUND)
    for index, item in enumerate(items):
        montage.paste(item, ((index % columns) * 600, (index // columns) * 322))
    montage.save(output, quality=92, subsampling=0)


def build_package(
    dataset_root: Path,
    raw_root: Path,
    output: Path,
    risk_count: int,
    normal_count: int,
    second_count: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    ranked, summary = collect_ranked_records(dataset_root, raw_root)
    risk = select_unique_images(ranked, risk_count)
    risk_stems = {record["source_stem"] for record in risk}
    normal_candidates = [record for record in reversed(ranked) if record["source_stem"] not in risk_stems]
    normal = select_unique_images(normal_candidates, normal_count)
    for record in normal:
        record["sample_type"] = "first_normal"
    second = collect_second_controls(dataset_root, second_count)
    selected = risk + normal + second
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        previews = temporary / "previews"
        previews.mkdir()
        index_rows: list[dict[str, Any]] = []
        montage_items: dict[str, list[Image.Image]] = {"risk": [], "controls": []}
        for rank, record in enumerate(selected, 1):
            record["rank"] = rank
            filename = f"{rank:02d}_{record['label']}_{record['source_stem']}.jpg"
            render_comparison(record, previews / filename)
            index_rows.append({
                "rank": rank,
                "sample_type": record["sample_type"],
                "source_stem": record["source_stem"],
                "patient_id": record["patient_id"],
                "split": record["split"],
                "label": record["label"],
                "old_unique_points": record["old_unique"],
                "old_edge_ratio": f"{record['old_edge_ratio']:.6f}",
                "new_edge_ratio": f"{record['new_edge_ratio']:.6f}",
                "preview": filename,
            })
            thumb = Image.open(previews / filename).convert("RGB")
            thumb.thumbnail((600, 322))
            group = "risk" if record["sample_type"] == "first_risk" else "controls"
            montage_items[group].append(ImageOps.pad(thumb, (600, 322), color=BACKGROUND))
        make_montage(montage_items["risk"], temporary / "00_montage_triangle_risk.jpg")
        make_montage(montage_items["controls"], temporary / "01_montage_controls.jpg")
        with (temporary / "sample_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
        report = {
            "dataset": str(dataset_root.resolve()),
            "selection": {
                "triangle_risk": f"worst {risk_count} unique first-batch images ranked by legacy unique-point count then edge ratio",
                "first_normal": normal_count,
                "second_direct": second_count,
            },
            "summary": summary,
            "sample_count": len(index_rows),
            "risk_old_edge_ratio_range": [risk[0]["old_edge_ratio"], risk[-1]["old_edge_ratio"]],
            "risk_new_edge_ratio_min": min(record["new_edge_ratio"] for record in risk),
        }
        (temporary / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "README.md").write_text(
            "# 四角点修复抽样对照\n\n"
            f"从最终数据集第一批的{summary['objects_scanned']}个密集polygon中，按旧算法的唯一点数和最短边/最长边比例排序，选择最严重的{risk_count}张不同图像；另含{normal_count}张第一批普通对照和{second_count}张第二批直接四点对照。\n\n"
            "- 青色：原始人工密集轮廓\n- 橙色：旧顺序最远点算法\n- 绿色：最终联合数据集的新四角点\n- 黄色数字：关键点顺序\n\n"
            "先看`00_montage_triangle_risk.jpg`，再看`01_montage_controls.jpg`；需要放大时打开`index.html`或`previews/`中的原尺寸对照图。\n",
            encoding="utf-8",
        )
        write_html(temporary / "index.html", index_rows)
        temporary.rename(output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "datasets/yolo_lateral_reviewed_combined_20cls")
    parser.add_argument("--raw-root", type=Path, default=Path("/Volumes/E/spine_data/LAT202511"))
    parser.add_argument("--output", type=Path, default=root / "analysis/lateral_pose_20_corner_sample_28_20260908")
    parser.add_argument("--risk-count", type=int, default=20)
    parser.add_argument("--normal-count", type=int, default=4)
    parser.add_argument("--second-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_package(
        args.dataset_root, args.raw_root, args.output,
        args.risk_count, args.normal_count, args.second_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Created: {args.output.resolve()}")


if __name__ == "__main__":
    main()
