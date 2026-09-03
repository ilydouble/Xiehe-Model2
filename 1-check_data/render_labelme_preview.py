#!/usr/bin/env python3
"""Render a LabelMe annotation preview without Pillow or OpenCV.

FFmpeg performs image decoding/scaling to a binary PPM frame. This script then
draws polygons, lines, points, and compact bitmap labels using only the Python
standard library.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


FONT = {
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
}


def read_ppm(path: Path) -> tuple[int, int, bytearray]:
    data = path.read_bytes()
    if not data.startswith(b"P6\n"):
        raise ValueError("expected binary PPM (P6)")
    cursor = 3
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while cursor < len(data) and chr(data[cursor]).isspace():
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            cursor = data.index(b"\n", cursor) + 1
            continue
        end = cursor
        while end < len(data) and not chr(data[end]).isspace():
            end += 1
        tokens.append(data[cursor:end])
        cursor = end
    width, height, maximum = map(int, tokens)
    if maximum != 255:
        raise ValueError(f"unsupported PPM maximum: {maximum}")
    while cursor < len(data) and chr(data[cursor]).isspace():
        cursor += 1
    pixels = bytearray(data[cursor:])
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"PPM pixel length mismatch: {len(pixels)} != {expected}")
    return width, height, pixels


def write_ppm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset:offset + 3] = bytes(color)


def draw_disk(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_x * delta_x + delta_y * delta_y <= radius * radius:
                set_pixel(pixels, width, height, x + delta_x, y + delta_y, color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1)
    sy = 1 if y1 < y2 else -1
    error = dx + dy
    while True:
        draw_disk(pixels, width, height, x1, y1, thickness, color)
        if x1 == x2 and y1 == y2:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x1 += sx
        if doubled <= dx:
            error += dx
            y1 += sy


def draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: int = 2,
) -> None:
    normalized = text.upper()
    text_width = max(1, len(normalized) * 6 * scale - scale)
    text_height = 7 * scale
    for background_y in range(y - 2, y + text_height + 2):
        for background_x in range(x - 2, x + text_width + 2):
            set_pixel(pixels, width, height, background_x, background_y, (0, 0, 0))
    cursor_x = x
    for character in normalized:
        glyph = FONT.get(character, FONT["?"])
        for row_index, row in enumerate(glyph):
            for column_index, enabled in enumerate(row):
                if enabled == "1":
                    for offset_y in range(scale):
                        for offset_x in range(scale):
                            set_pixel(
                                pixels,
                                width,
                                height,
                                cursor_x + column_index * scale + offset_x,
                                y + row_index * scale + offset_y,
                                color,
                            )
        cursor_x += 6 * scale


def scaled_points(points: Iterable[Iterable[float]], scale_x: float, scale_y: float) -> list[tuple[int, int]]:
    return [(round(float(point[0]) * scale_x), round(float(point[1]) * scale_y)) for point in points]


def render(json_path: Path, output_path: Path, max_width: int, highlight_labels: set[str]) -> None:
    with json_path.open("r", encoding="utf-8-sig") as handle:
        annotation = json.load(handle)
    image_path = json_path.parent / Path(annotation["imagePath"].replace("\\", "/")).name
    if not image_path.is_file():
        candidates = [path for path in json_path.parent.glob(f"{json_path.stem}.*") if path.suffix.lower() != ".json"]
        if len(candidates) != 1:
            raise FileNotFoundError(f"cannot resolve image for {json_path}")
        image_path = candidates[0]

    with tempfile.TemporaryDirectory(prefix="labelme-preview-") as temporary_directory:
        base_ppm = Path(temporary_directory) / "base.ppm"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(image_path),
            "-vf", f"scale='min({max_width},iw)':-2", "-frames:v", "1", "-y", str(base_ppm),
        ], check=True)
        width, height, pixels = read_ppm(base_ppm)

    source_width = float(annotation.get("imageWidth") or width)
    source_height = float(annotation.get("imageHeight") or height)
    scale_x = width / source_width
    scale_y = height / source_height
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label", "?"))
        points = scaled_points(shape.get("points", []), scale_x, scale_y)
        if not points:
            continue
        highlighted = label.upper() in highlight_labels
        shape_type = shape.get("shape_type")
        color = (255, 55, 55) if highlighted else (0, 255, 80)
        if shape_type == "point":
            color = (255, 55, 55) if highlighted else (0, 220, 255)
            draw_disk(pixels, width, height, points[0][0], points[0][1], 7, color)
        else:
            pairs = list(zip(points, points[1:]))
            if shape_type in {"polygon", "rectangle", "circle"} and len(points) > 2:
                pairs.append((points[-1], points[0]))
            for start, end in pairs:
                draw_line(pixels, width, height, start, end, color, 2 if highlighted else 1)
        label_x = min(max(points[0][0] + 5, 2), max(2, width - 70))
        label_y = min(max(points[0][1] - 18, 2), max(2, height - 18))
        draw_text(pixels, width, height, label_x, label_y, label, color, scale=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="labelme-preview-output-") as temporary_directory:
        annotated_ppm = Path(temporary_directory) / "annotated.ppm"
        write_ppm(annotated_ppm, width, height, pixels)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(annotated_ppm),
            "-frames:v", "1", "-y", str(output_path),
        ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a LabelMe JSON preview without Pillow/OpenCV")
    parser.add_argument("json", type=Path, help="LabelMe JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--max-width", type=int, default=768, help="Maximum preview width")
    parser.add_argument("--highlight", action="append", default=[], help="Label to render in red; repeatable")
    args = parser.parse_args()
    render(args.json.resolve(), args.output.resolve(), args.max_width, {label.upper() for label in args.highlight})
    print(args.output)


if __name__ == "__main__":
    main()
