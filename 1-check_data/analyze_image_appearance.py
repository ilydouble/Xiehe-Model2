#!/usr/bin/env python3
"""Analyze PNG resolution, aspect ratio, color type, and black edge bands.

Images are decoded read-only by FFmpeg and downsampled to a fixed width before
edge analysis.  This keeps the scan dependency-light and fast while retaining
the relative thickness of large black borders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {
    0: "grayscale",
    2: "rgb",
    3: "indexed",
    4: "grayscale_alpha",
    6: "rgba",
}
SIDES = ("top", "bottom", "left", "right")


def read_png_header(path: Path) -> dict[str, int | str]:
    with path.open("rb") as handle:
        header = handle.read(29)
    if len(header) < 29 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    bit_depth, color_type, compression, filtering, interlace = header[24:29]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions: {path}")
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "color_mode": COLOR_TYPES.get(color_type, f"type_{color_type}"),
        "compression": compression,
        "filtering": filtering,
        "interlace": interlace,
    }


def _read_pnm_token(data: bytes, index: int) -> tuple[bytes, int]:
    while index < len(data):
        if data[index:index + 1] == b"#":
            newline = data.find(b"\n", index)
            if newline < 0:
                raise ValueError("Unterminated PGM comment")
            index = newline + 1
        elif data[index] in b" \t\r\n":
            index += 1
        else:
            break
    start = index
    while index < len(data) and data[index] not in b" \t\r\n#":
        index += 1
    if start == index:
        raise ValueError("Missing PGM header token")
    return data[start:index], index


def parse_pgm(data: bytes) -> tuple[int, int, bytes]:
    index = 0
    magic, index = _read_pnm_token(data, index)
    width_token, index = _read_pnm_token(data, index)
    height_token, index = _read_pnm_token(data, index)
    max_value_token, index = _read_pnm_token(data, index)
    if magic != b"P5":
        raise ValueError(f"Expected binary PGM (P5), got {magic!r}")
    width, height, max_value = int(width_token), int(height_token), int(max_value_token)
    if max_value != 255 or width <= 0 or height <= 0:
        raise ValueError(f"Unsupported PGM geometry/max value: {width}x{height}, {max_value}")
    if index >= len(data) or data[index] not in b" \t\r\n":
        raise ValueError("Missing PGM pixel separator")
    index += 2 if data[index:index + 2] == b"\r\n" else 1
    pixels = data[index:]
    if len(pixels) != width * height:
        raise ValueError(f"PGM pixel length mismatch: expected {width * height}, got {len(pixels)}")
    return width, height, pixels


def decode_small_grayscale(path: Path, ffmpeg: str, sample_width: int) -> tuple[int, int, bytes]:
    command = [
        ffmpeg, "-v", "error", "-threads", "1", "-i", str(path),
        "-vf", f"scale={sample_width}:-2:flags=area,format=gray",
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "pgm", "-",
    ]
    completed = subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
    )
    return parse_pgm(completed.stdout)


def edge_black_bands(
    width: int,
    height: int,
    pixels: bytes,
    near_black_threshold: int = 5,
    required_fraction: float = 0.98,
) -> dict[str, int]:
    """Return consecutive near-black rows/columns from each image edge."""
    if len(pixels) != width * height:
        raise ValueError("Pixel buffer does not match dimensions")
    if not 0 <= near_black_threshold <= 255 or not 0 < required_fraction <= 1:
        raise ValueError("Invalid black-band threshold")

    def row_is_black(y: int) -> bool:
        row = pixels[y * width:(y + 1) * width]
        dark = sum(row.count(value) for value in range(near_black_threshold + 1))
        return dark / width >= required_fraction

    def column_is_black(x: int) -> bool:
        dark = sum(pixels[y * width + x] <= near_black_threshold for y in range(height))
        return dark / height >= required_fraction

    def count_from(indices: Iterable[int], predicate: Any) -> int:
        count = 0
        for index in indices:
            if not predicate(index):
                break
            count += 1
        return count

    return {
        "top": count_from(range(height), row_is_black),
        "bottom": count_from(range(height - 1, -1, -1), row_is_black),
        "left": count_from(range(width), column_is_black),
        "right": count_from(range(width - 1, -1, -1), column_is_black),
    }


def source_prefix(stem: str) -> str:
    match = re.match(r"[A-Za-z]+", stem)
    return match.group(0).upper() if match else "OTHER"


def analyze_one(
    path: Path,
    ffmpeg: str,
    sample_width: int,
    near_black_threshold: int,
    required_fraction: float,
    notable_ratio: float,
) -> dict[str, Any]:
    header = read_png_header(path)
    width, height = int(header["width"]), int(header["height"])
    sample_w, sample_h, pixels = decode_small_grayscale(path, ffmpeg, sample_width)
    bands = edge_black_bands(
        sample_w, sample_h, pixels, near_black_threshold, required_fraction
    )
    ratios = {
        "top": bands["top"] / sample_h,
        "bottom": bands["bottom"] / sample_h,
        "left": bands["left"] / sample_w,
        "right": bands["right"] / sample_w,
    }
    estimated_pixels = {
        "top": round(ratios["top"] * height),
        "bottom": round(ratios["bottom"] * height),
        "left": round(ratios["left"] * width),
        "right": round(ratios["right"] * width),
    }
    near_black_pixels = sum(pixels.count(value) for value in range(near_black_threshold + 1))
    notable_sides = [side for side in SIDES if ratios[side] >= notable_ratio]
    vertical_total = ratios["top"] + ratios["bottom"]
    horizontal_total = ratios["left"] + ratios["right"]
    return {
        "stem": path.stem,
        "filename": path.name,
        "source": source_prefix(path.stem),
        "width": width,
        "height": height,
        "megapixels": width * height / 1_000_000,
        "aspect_width_height": width / height,
        "orientation": "portrait" if height > width else "landscape" if width > height else "square",
        **header,
        "sample_width": sample_w,
        "sample_height": sample_h,
        "mean_gray": sum(pixels) / len(pixels),
        "near_black_fraction": near_black_pixels / len(pixels),
        **{f"black_{side}_sample_px": bands[side] for side in SIDES},
        **{f"black_{side}_estimated_px": estimated_pixels[side] for side in SIDES},
        **{f"black_{side}_ratio": ratios[side] for side in SIDES},
        "detected_black_band": any(bands.values()),
        "notable_black_border": bool(notable_sides),
        "notable_sides": ";".join(notable_sides),
        "severe_black_border": vertical_total >= 0.10 or horizontal_total >= 0.10,
        "usable_height_ratio_after_edge_crop": max(0.0, 1.0 - vertical_total),
        "usable_width_ratio_after_edge_crop": max(0.0, 1.0 - horizontal_total),
    }


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def summarize(records: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not records:
        raise ValueError("No image records to summarize")
    dimension_counts = Counter(f"{row['width']}x{row['height']}" for row in records)
    ratio_bins = Counter()
    for row in records:
        ratio = row["aspect_width_height"]
        if ratio < 0.30:
            ratio_bins["<0.30"] += 1
        elif ratio < 0.35:
            ratio_bins["0.30-<0.35"] += 1
        elif ratio < 0.40:
            ratio_bins["0.35-<0.40"] += 1
        elif ratio < 0.45:
            ratio_bins["0.40-<0.45"] += 1
        else:
            ratio_bins[">=0.45"] += 1

    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        source_rows[row["source"]].append(row)
    source_summary = {}
    for source, rows in sorted(source_rows.items()):
        source_summary[source] = {
            "images": len(rows),
            "dimensions": len({(row["width"], row["height"]) for row in rows}),
            "median_width": quantile([row["width"] for row in rows], 0.5),
            "median_height": quantile([row["height"] for row in rows], 0.5),
            "median_aspect_width_height": quantile(
                [row["aspect_width_height"] for row in rows], 0.5
            ),
            "color_modes": dict(sorted(Counter(row["color_mode"] for row in rows).items())),
            "detected_black_band": sum(bool(row["detected_black_band"]) for row in rows),
            "notable_black_border": sum(bool(row["notable_black_border"]) for row in rows),
            "severe_black_border": sum(bool(row["severe_black_border"]) for row in rows),
            "median_near_black_fraction": quantile(
                [row["near_black_fraction"] for row in rows], 0.5
            ),
        }

    black_by_side = {
        side: {
            "detected_images": sum(row[f"black_{side}_sample_px"] > 0 for row in records),
            "notable_images": sum(row[f"black_{side}_ratio"] >= config["notable_ratio"] for row in records),
            "ratio_distribution": distribution([row[f"black_{side}_ratio"] for row in records]),
            "max_estimated_pixels": max(row[f"black_{side}_estimated_px"] for row in records),
        }
        for side in SIDES
    }
    return {
        "schema_version": 1,
        "dataset_root": config["source"],
        "method": {
            "sample_width": config["sample_width"],
            "near_black_threshold": config["near_black_threshold"],
            "required_near_black_fraction_per_edge_line": config["required_fraction"],
            "notable_side_ratio": config["notable_ratio"],
            "severe_rule": "top+bottom >= 10% or left+right >= 10%",
            "estimated_pixels_are_derived_from_downsampled_band_ratios": True,
        },
        "counts": {
            "images": len(records),
            "unique_dimensions": len(dimension_counts),
            "orientation": dict(sorted(Counter(row["orientation"] for row in records).items())),
            "color_modes": dict(sorted(Counter(row["color_mode"] for row in records).items())),
            "bit_depths": dict(sorted(Counter(str(row["bit_depth"]) for row in records).items())),
            "detected_black_band": sum(bool(row["detected_black_band"]) for row in records),
            "notable_black_border": sum(bool(row["notable_black_border"]) for row in records),
            "severe_black_border": sum(bool(row["severe_black_border"]) for row in records),
        },
        "resolution": {
            "width": distribution([row["width"] for row in records]),
            "height": distribution([row["height"] for row in records]),
            "megapixels": distribution([row["megapixels"] for row in records]),
            "top_dimensions": dict(dimension_counts.most_common(20)),
        },
        "aspect_width_height": {
            "distribution": distribution([row["aspect_width_height"] for row in records]),
            "bins": {key: ratio_bins[key] for key in ("<0.30", "0.30-<0.35", "0.35-<0.40", "0.40-<0.45", ">=0.45")},
        },
        "intensity": {
            "mean_gray": distribution([row["mean_gray"] for row in records]),
            "near_black_fraction": distribution([row["near_black_fraction"] for row in records]),
        },
        "black_borders": {
            "by_side": black_by_side,
            "largest_vertical_total": [
                {
                    "stem": row["stem"],
                    "source": row["source"],
                    "top_ratio": row["black_top_ratio"],
                    "bottom_ratio": row["black_bottom_ratio"],
                }
                for row in sorted(
                    records,
                    key=lambda item: item["black_top_ratio"] + item["black_bottom_ratio"],
                    reverse=True,
                )[:20]
            ],
            "largest_horizontal_total": [
                {
                    "stem": row["stem"],
                    "source": row["source"],
                    "left_ratio": row["black_left_ratio"],
                    "right_ratio": row["black_right_ratio"],
                }
                for row in sorted(
                    records,
                    key=lambda item: item["black_left_ratio"] + item["black_right_ratio"],
                    reverse=True,
                )[:20]
            ],
        },
        "sources": source_summary,
    }


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("/Volumes/E/spine_data/20260903-侧面数据第二批/labelme-export/P202607076308"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "analysis/side_labelme_20260903",
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--sample-width", type=int, default=256)
    parser.add_argument("--near-black-threshold", type=int, default=5)
    parser.add_argument("--required-fraction", type=float, default=0.98)
    parser.add_argument("--notable-ratio", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, output_dir = args.source.resolve(), args.output_dir.resolve()
    images = sorted(
        path for path in source.glob("*.png") if not path.name.startswith("._")
    )
    if not images:
        raise FileNotFoundError(f"No real PNG images found under {source}")
    if args.sample_width <= 0 or args.workers <= 0:
        raise ValueError("sample-width and workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    def worker(path: Path) -> dict[str, Any]:
        return analyze_one(
            path, args.ffmpeg, args.sample_width, args.near_black_threshold,
            args.required_fraction, args.notable_ratio,
        )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(worker, images), 1):
            records.append(record)
            if index % 50 == 0 or index == len(images):
                print(f"Analyzed {index}/{len(images)} images", flush=True)
    records.sort(key=lambda row: row["filename"])
    config = {
        "source": str(source),
        "sample_width": args.sample_width,
        "near_black_threshold": args.near_black_threshold,
        "required_fraction": args.required_fraction,
        "notable_ratio": args.notable_ratio,
    }
    summary = summarize(records, config)
    csv_path = output_dir / "image_appearance.csv"
    json_path = output_dir / "image_appearance_summary.json"
    write_csv(csv_path, records)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
