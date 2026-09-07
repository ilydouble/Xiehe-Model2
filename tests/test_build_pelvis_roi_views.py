import csv
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_pelvis_roi_views.py"
SPEC = importlib.util.spec_from_file_location("build_pelvis_roi_views", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def write_png_header(path: Path, width: int, height: int, suffix: bytes = b"") -> None:
    path.write_bytes(
        builder.PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR"
        + struct.pack(">II", width, height) + b"\x08\x00\x00\x00\x00" + suffix
    )


def valid_label() -> str:
    return (
        "0 0.500000 0.750000 0.200000 0.200000 "
        "0.500000 0.800000 2 0.420000 0.700000 2 0.580000 0.700000 2\n"
    )


class PelvisRoiBuilderTests(unittest.TestCase):
    def make_dataset(self, root: Path, leak: bool = False) -> Path:
        rows = []
        for split, group in (("train", "old:P1"), ("val", "new:P2"), ("test", "new:P3")):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            stem = f"{split}_sample"
            write_png_header(root / "images" / split / f"{stem}.png", 1000, 2000, split.encode())
            (root / "labels" / split / f"{stem}.txt").write_text(valid_label(), encoding="utf-8")
            rows.append({
                "batch": "old" if split == "train" else "new",
                "patient_id": group.split(":", 1)[1],
                "group_id": "old:P1" if leak and split == "val" else group,
                "split": split,
                "image": f"images/{split}/{stem}.png",
                "label": f"labels/{split}/{stem}.txt",
            })
        with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["batch", "patient_id", "group_id", "split", "image", "label"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return root

    def test_crop_is_deterministic_and_contains_target(self):
        label = builder.PoseLabel(
            0, (0.5, 0.75, 0.2, 0.2),
            ((0.5, 0.8, 2), (0.42, 0.7, 2), (0.58, 0.7, 2)),
        )
        first = builder.compute_crop_box(label, 1000, 2000, "case.png")
        second = builder.compute_crop_box(label, 1000, 2000, "case.png")
        self.assertEqual(first, second)
        builder.validate_crop_contains_target(first, label, 1000, 2000)
        transformed = builder.transform_label(label, first, 1000, 2000)
        builder.validate_transformed_label(transformed)

    def test_transform_clamps_subpixel_source_bbox_rounding(self):
        label = builder.PoseLabel(
            0, (0.486471, 0.928712, 0.135869, 0.142577),
            ((0.438536, 0.992429, 2), (0.474985, 0.909701, 2), (0.534405, 0.879924, 2)),
        )
        box = builder.compute_crop_box(label, 3037, 5327, "edge.png")
        transformed = builder.transform_label(label, box, 3037, 5327)
        builder.validate_transformed_label(transformed)
        self.assertLessEqual(transformed.bbox[1] + transformed.bbox[3] / 2, 1.0)

    def test_plan_uses_train_only_and_rejects_patient_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_dataset(Path(directory) / "ok")
            records = builder.plan_views(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["group_id"], "old:P1")
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_dataset(Path(directory) / "leak", leak=True)
            with self.assertRaisesRegex(ValueError, "cross splits"):
                builder.plan_views(root)

    def test_build_writes_only_train_views_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_dataset(base / "source")
            output = base / "roi"

            def fake_crop(source_path, output_path, box, ffmpeg):
                write_png_header(output_path, box.width, box.height, b"cropped")

            report = builder.build_dataset(
                source, output, apply=True, workers=1, cropper=fake_crop
            )
            self.assertEqual(report["summary"]["roi_train_images"], 1)
            self.assertTrue((output / "images/train/roi__train_sample.png").is_file())
            self.assertTrue((output / "labels/train/roi__train_sample.txt").is_file())
            self.assertFalse((output / "images/val").exists())
            self.assertFalse((output / "images/test").exists())
            with (output / "manifest.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["source_split"], "train")
            self.assertEqual(rows[0]["group_id"], "old:P1")
            parsed = json.loads((output / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["leakage_contract"]["derived_splits"], ["train"])
            audit = builder.audit_leakage(source, output)
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["counts"]["roi_non_train_patient_groups"], 0)
            self.assertEqual(audit["counts"]["roi_exact_matches_to_val_test"], 0)

            rows[0]["group_id"] = "new:P2"
            with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            failed = builder.audit_leakage(source, output)
            self.assertEqual(failed["status"], "failed")
            self.assertGreater(failed["counts"]["roi_non_train_patient_groups"], 0)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_dataset(base / "source")
            output = base / "roi"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                builder.build_dataset(source, output, apply=True, workers=1)


if __name__ == "__main__":
    unittest.main()
