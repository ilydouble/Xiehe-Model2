import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/convert_second_lateral_pose.py"
SPEC = importlib.util.spec_from_file_location("convert_second_lateral_pose", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


def write_png_header(path: Path, width: int = 100, height: int = 200) -> None:
    path.write_bytes(
        converter.PNG_SIGNATURE
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


def write_annotation(path: Path, width: int = 100, height: int = 200) -> None:
    data = {
        "imageWidth": width,
        "imageHeight": height,
        "shapes": [
            {
                "label": "C2",
                "shape_type": "polygon",
                "points": [[30, 40], [10, 40], [12, 20], [28, 20]],
            },
            {
                "label": "S1",
                "shape_type": "line",
                "points": [[20, 180], [80, 180]],
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class ConverterTests(unittest.TestCase):
    def test_orders_quadrilateral_ul_ur_lr_ll(self):
        points = [[30, 40], [10, 40], [12, 20], [28, 20]]
        self.assertEqual(
            converter.order_quadrilateral(points),
            [(12.0, 20.0), (28.0, 20.0), (30.0, 40.0), (10.0, 40.0)],
        )

    def test_pose_line_has_four_visible_keypoints(self):
        line, clipped = converter.yolo_pose_line(
            0, [[30, 40], [10, 40], [12, -2], [28, 20]], 100, 200
        )
        parts = line.split()
        self.assertEqual(len(parts), 17)
        self.assertEqual(parts[0], "0")
        self.assertEqual(parts[7::3], ["2", "2", "2", "2"])
        self.assertEqual(clipped, 1)
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in parts[1:5]))

    def test_patient_split_keeps_repeated_images_together(self):
        split = converter.stable_group_split(["P1", "P1", "P2", "P3", "P4"], 0.25, 0.25, 42)
        self.assertEqual(set(split), {"P1", "P2", "P3", "P4"})
        self.assertIn(split["P1"], {"train", "val", "test"})

    def test_end_to_end_conversion_ignores_nonvertebral_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            for stem in ("SRC001_LAT_A", "SRC002_LAT_A", "SRC003_LAT_A", "SRC004_LAT_A"):
                write_png_header(source / f"{stem}.png")
                write_annotation(source / f"{stem}.json")
            output = root / "output"
            report = converter.convert_dataset(source, output, None, 0.25, 0.25, 7)

            self.assertEqual(report["counts"]["included_images"], 4)
            self.assertEqual(report["counts"]["objects_by_class"]["C2"], 4)
            self.assertEqual(report["counts"]["objects_by_class"]["C7"], 0)
            self.assertTrue((output / "data.yaml").is_file())
            label_files = list((output / "labels").rglob("*.txt"))
            self.assertEqual(len(label_files), 4)
            self.assertTrue(all(len(path.read_text().splitlines()) == 1 for path in label_files))


if __name__ == "__main__":
    unittest.main()
