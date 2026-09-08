import csv
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_lateral_black_roi_views.py"
SPEC = importlib.util.spec_from_file_location("build_lateral_black_roi_views", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def write_png_header(path: Path, width: int, height: int, suffix: bytes = b"") -> None:
    path.write_bytes(builder.PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x00\x00\x00\x00" + suffix)


def line(class_id: int, y: float) -> str:
    return f"{class_id} 0.5 {y} 0.2 0.1 0.4 {y-0.05} 2 0.6 {y-0.05} 2 0.6 {y+0.05} 2 0.4 {y+0.05} 2"


class LateralBlackRoiTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        rows = []
        for split, patient in (("train", "first:P1"), ("val", "second:P2"), ("test", "second:P3")):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            stem = f"{split}_sample"
            write_png_header(root / "images" / split / f"{stem}.png", 1000, 2000, split.encode())
            (root / "labels" / split / f"{stem}.txt").write_text(line(0, 0.2) + "\n" + line(1, 0.8) + "\n", encoding="utf-8")
            rows.append({"output_stem": stem, "source_batch": patient.split(":")[0], "source_stem": stem, "patient_id": patient, "split": split, "image": f"images/{split}/{stem}.png", "label": f"labels/{split}/{stem}.txt"})
        with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        return root

    @staticmethod
    def black_detector(path, ffmpeg, sample_width, threshold, fraction):
        return {"top": 0.05, "bottom": 0.05, "left": 0.10, "right": 0.10}

    def test_multi_object_transform_preserves_classes_and_convexity(self):
        objects = [builder.parse_pose_labels]
        source = builder.PoseObject(0, (0.5, 0.5, 0.2, 0.2), ((0.4, 0.4, 2), (0.6, 0.4, 2), (0.6, 0.6, 2), (0.4, 0.6, 2)))
        box = builder.CropBox(100, 100, 900, 1900)
        transformed = builder.transform_object(source, box, 1000, 2000)
        self.assertEqual(transformed.class_id, 0)
        builder.validate_pose_object(transformed, "test")

    def test_no_notable_black_band_is_skipped(self):
        obj = builder.PoseObject(0, (0.5, 0.5, 0.2, 0.2), ((0.4, 0.4, 2), (0.6, 0.4, 2), (0.6, 0.6, 2), (0.4, 0.6, 2)))
        ratios = {side: 0.005 for side in builder.SIDES}
        self.assertIsNone(builder.compute_black_crop(ratios, [obj], 1000, 2000))

    def test_plan_and_build_use_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_dataset(base / "source")
            records, summary = builder.plan_views(source, workers=1, detector=self.black_detector)
            self.assertEqual(len(records), 1)
            self.assertEqual(summary["mixed_train_views"], 2)
            self.assertEqual(len(records[0]["transformed"]), 2)
            output = base / "roi"

            def fake_crop(source_path, output_path, box, ffmpeg):
                write_png_header(output_path, box.width, box.height, b"crop")

            report = builder.build_dataset(source, output, workers=1, apply=True, detector=self.black_detector, cropper=fake_crop)
            self.assertEqual(report["summary"]["roi_train_images"], 1)
            self.assertFalse((output / "images/val").exists())
            self.assertEqual(len(builder.parse_pose_labels(next((output / "labels/train").glob("*.txt")))), 2)
            audit = builder.audit_leakage(source, output)
            self.assertEqual(audit["status"], "passed")

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_dataset(base / "source")
            output = base / "roi"; output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                builder.build_dataset(source, output, workers=1, apply=True, detector=self.black_detector)


if __name__ == "__main__":
    unittest.main()
