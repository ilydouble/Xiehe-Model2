import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/convert_combined_pelvis_pose.py"
SPEC = importlib.util.spec_from_file_location("convert_combined_pelvis_pose", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def write_png_header(path: Path, marker: str, width: int = 100, height: int = 200) -> None:
    path.write_bytes(
        converter.PNG_SIGNATURE
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + marker.encode()
    )


def write_annotation(path: Path, femoral: str = "CFH", reverse_s1: bool = False) -> None:
    shapes = [{
        "label": "S1", "shape_type": "line",
        "points": [[80, 180], [20, 170]] if reverse_s1 else [[20, 170], [80, 180]],
    }]
    if femoral == "CFH":
        shapes.append({"label": "CFH", "shape_type": "point", "points": [[50, 150]]})
    elif femoral == "FH":
        shapes.extend([
            {"label": "FH-1", "shape_type": "point", "points": [[40, 140]]},
            {"label": "FH-2", "shape_type": "point", "points": [[60, 160]]},
        ])
    path.write_text(json.dumps({"imageWidth": 100, "imageHeight": 200, "shapes": shapes}), encoding="utf-8")


class ConverterTests(unittest.TestCase):
    def test_extracts_direct_cfh_and_orders_s1_by_image_x(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation = Path(temp_dir) / "sample.json"
            write_annotation(annotation, "CFH", reverse_s1=True)
            width, height, points, source = converter.extract_pelvis_annotation(annotation, "old")
            self.assertEqual((width, height), (100, 200))
            self.assertEqual(points, ((50.0, 150.0), (20.0, 170.0), (80.0, 180.0)))
            self.assertEqual(source, "direct")

    def test_converts_fh_pair_to_midpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation = Path(temp_dir) / "sample.json"
            write_annotation(annotation, "FH")
            _, _, points, source = converter.extract_pelvis_annotation(annotation, "new")
            self.assertEqual(points[0], (50.0, 150.0))
            self.assertEqual(source, "FH_midpoint")

    def test_pose_line_has_one_class_and_three_visible_keypoints(self):
        line, clipped = converter.yolo_pose_line(
            ((50, 150), (-2, 170), (80, 180)), width=100, height=200
        )
        parts = line.split()
        self.assertEqual(len(parts), 14)
        self.assertEqual(parts[0], "0")
        self.assertEqual([parts[index] for index in (7, 10, 13)], ["2", "2", "2"])
        self.assertEqual(clipped, 1)
        self.assertTrue(all(0 <= float(value) <= 1 for value in parts[1:5]))

    def test_exact_duplicate_images_are_removed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a, image_b = root / "a.png", root / "b.png"
            write_png_header(image_a, "same")
            image_b.write_bytes(image_a.read_bytes())
            annotation = root / "a.json"
            write_annotation(annotation)
            base = dict(
                batch="old", annotation_path=annotation, patient_id="P", group_id="old:P",
                width=100, height=200, points=((50, 150), (20, 170), (80, 180)), cfh_source="direct",
            )
            first = converter.Candidate(image_path=image_a, **base)
            second = converter.Candidate(image_path=image_b, **base)
            kept, duplicates = converter.deduplicate_candidates([first, second])
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(duplicates), 1)

    def test_end_to_end_combines_batches_and_keeps_patients_grouped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            for index in range(4):
                patient_dir = old / f"OLD{index}"
                patient_dir.mkdir()
                stem = f"old_image_{index}"
                write_png_header(patient_dir / f"{stem}.png", f"old-{index}")
                write_annotation(patient_dir / f"{stem}.json", "CFH")
            for index in range(4):
                stem = f"NEW{index}_LAT_A"
                write_png_header(new / f"{stem}.png", f"new-{index}")
                write_annotation(new / f"{stem}.json", "FH")

            output = root / "output"
            report = converter.convert_dataset(old, new, output, 0.25, 0.25, seed=7)
            self.assertEqual(report["counts"]["included_images"], 8)
            self.assertEqual(report["counts"]["images_by_batch"], {"new": 4, "old": 4})
            self.assertEqual(report["counts"]["cfh_by_source"], {"FH_midpoint": 4, "direct": 4})
            self.assertEqual(len(list((output / "labels").rglob("*.txt"))), 8)
            for label in (output / "labels").rglob("*.txt"):
                self.assertEqual(len(label.read_text().split()), 14)
            with (output / "manifest.csv").open(encoding="utf-8") as handle:
                manifests = list(converter.csv.DictReader(handle))
            by_group: dict[str, set[str]] = {}
            for row in manifests:
                by_group.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))


if __name__ == "__main__":
    unittest.main()
