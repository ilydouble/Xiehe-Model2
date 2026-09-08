import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "5-inference/render_lateral_pseudolabel_review.py"
SPEC = importlib.util.spec_from_file_location("render_lateral_pseudolabel_review", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def polygon(label, y, pseudo=False, quality="high"):
    shape = {
        "label": label,
        "shape_type": "polygon",
        "points": [[80, y], [120, y], [120, y + 25], [80, y + 25]],
        "flags": {},
    }
    if pseudo:
        shape["flags"] = {"pseudo_label": True, "needs_review": True, f"quality_{quality}": True}
    return shape


class ReviewRendererTests(unittest.TestCase):
    def test_cervical_crop_contains_all_cervical_points(self):
        annotation = {"shapes": [polygon("C2", 100, True), polygon("C7", 500)]}
        crop = MODULE.cervical_crop(annotation, 1000, 2000)
        self.assertLessEqual(crop[0], 80)
        self.assertGreaterEqual(crop[2], 120)
        self.assertLessEqual(crop[1], 100)
        self.assertGreaterEqual(crop[3], 525)
        self.assertTrue(0 <= crop[0] < crop[2] <= 1000)
        self.assertTrue(0 <= crop[1] < crop[3] <= 2000)

    def test_summary_and_priority(self):
        annotation = {
            "shapes": [
                polygon("C7", 500), polygon("C2", 100, True, "high"),
                polygon("C3", 180, True, "medium"), polygon("C4", 260, True, "low"),
            ]
        }
        summary = MODULE.summarize_annotation(annotation)
        self.assertEqual(summary["manual_labels"], ["C7"])
        self.assertEqual(summary["low_labels"], ["C4"])
        self.assertEqual(MODULE.priority_for([], summary), "含low")
        self.assertEqual(MODULE.priority_for(["C5"], summary), "缺候选")

    def test_build_and_audit_one_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            pseudo = root / "pseudo"
            patient = source / "P1"
            candidate_dir = pseudo / "candidates" / "P1"
            patient.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            image_path = patient / "sample.png"
            Image.new("L", (300, 800), 80).save(image_path)
            source_json = patient / "sample.json"
            source_json.write_text(json.dumps({"shapes": [polygon("C7", 260)]}))
            annotation = {"shapes": [
                polygon("C7", 260), polygon("C2", 60, True, "high"),
                polygon("C3", 100, True, "medium"), polygon("C4", 140, True, "low"),
            ]}
            candidate_json = candidate_dir / "sample.json"
            candidate_json.write_text(json.dumps(annotation))
            write_csv(
                pseudo / "manifest.csv",
                ["relative_image", "source_json", "output_json", "black_left_px", "black_right_px", "black_top_px", "black_bottom_px"],
                [{
                    "relative_image": "P1/sample.png", "source_json": str(source_json),
                    "output_json": "candidates/P1/sample.json", "black_left_px": 10,
                    "black_right_px": 12, "black_top_px": 0, "black_bottom_px": 0,
                }],
            )
            write_csv(
                pseudo / "review_queue.csv",
                ["relative_image", "label", "status"],
                [
                    {"relative_image": "P1/sample.png", "label": "C2", "status": "review_required"},
                    {"relative_image": "P1/sample.png", "label": "C5", "status": "no_candidate"},
                ],
            )
            (pseudo / "summary.json").write_text(json.dumps({"counts": {"processed_samples": 1}}))
            output = root / "review"
            report = MODULE.build_package(source, pseudo, output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["preview_count"], 1)
            self.assertTrue((output / "打开此文件逐张人工复核.html").is_file())
            rows = MODULE.read_csv(output / "人工复核索引.csv")
            self.assertEqual(rows[0]["priority"], "缺候选")
            with Image.open(output / rows[0]["preview"]) as preview:
                self.assertEqual(preview.size, (1780, 1750))
            (output / "previews" / "._metadata.jpg").write_bytes(b"AppleDouble")
            self.assertEqual(MODULE.audit_package(pseudo, output)["status"], "passed")
            with self.assertRaises(FileExistsError):
                MODULE.build_package(source, pseudo, output)


if __name__ == "__main__":
    unittest.main()
