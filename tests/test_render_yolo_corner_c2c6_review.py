import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "5-inference/render_yolo_corner_c2c6_review.py"
SPEC = importlib.util.spec_from_file_location("render_yolo_corner_c2c6_review", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def yolo_line(class_id=0):
    return f"{class_id} 0.5 0.5 0.2 0.1 0.4 0.45 2 0.6 0.45 2 0.6 0.55 2 0.4 0.55 2\n"


def pseudo_shape(label):
    return {
        "label": label,
        "points": [[100, 50], [140, 50], [140, 80], [100, 80]],
        "shape_type": "polygon",
        "flags": {"pseudo_label": True, "needs_review": True, "quality_high": True},
    }


class YoloCornerReviewTests(unittest.TestCase):
    def test_parse_yolo_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            label = root / "sample.txt"
            Image.new("L", (300, 800), 80).save(image)
            label.write_text(yolo_line())
            annotation = MODULE.parse_yolo_annotation(label, image)
            self.assertEqual(annotation["imageWidth"], 300)
            self.assertEqual(annotation["shapes"][0]["label"], "C7")
            self.assertEqual(len(annotation["shapes"][0]["points"]), 4)

    def test_build_uses_yolo_baseline_and_only_c2c6_pseudo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, raw, pseudo = root / "yolo_corner", root / "LAT", root / "pseudo"
            for split in MODULE.SPLITS:
                (dataset / "images" / split).mkdir(parents=True)
                (dataset / "labels" / split).mkdir(parents=True)
            image = dataset / "images/train/sample.png"
            label = dataset / "labels/train/sample.txt"
            Image.new("L", (300, 800), 80).save(image)
            label.write_text(yolo_line())
            patient = raw / "P1"
            patient.mkdir(parents=True)
            Image.new("L", (300, 800), 80).save(patient / "sample.png")
            (patient / "sample.json").write_text(json.dumps({"shapes": []}))
            candidate_dir = pseudo / "candidates/P1"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "sample.json").write_text(json.dumps({
                "imageWidth": 300,
                "imageHeight": 800,
                "shapes": [pseudo_shape("C2"), pseudo_shape("T2")],
            }))
            with (pseudo / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "relative_image", "output_json", "black_left_px", "black_right_px",
                    "black_top_px", "black_bottom_px",
                ])
                writer.writeheader()
                writer.writerow({
                    "relative_image": "P1/sample.png",
                    "output_json": "candidates/P1/sample.json",
                    "black_left_px": 0,
                    "black_right_px": 0,
                    "black_top_px": 0,
                    "black_bottom_px": 0,
                })
            model = root / "model.pt"
            model.write_bytes(b"test-model")
            output = root / "review"
            args = SimpleNamespace(
                dataset=dataset,
                raw_source=raw,
                pseudolabel_root=pseudo,
                model=model,
                output=output,
                limit=None,
                imgsz=1280,
                predict_conf=0.01,
                candidate_conf=0.03,
                iou=0.70,
                max_det=80,
                device="cpu",
            )
            report = MODULE.build_package(args)
            self.assertEqual(report["status"], "passed")
            rows = MODULE.read_csv(output / "人工复核索引.csv")
            self.assertEqual(rows[0]["added_labels"], "C2")
            self.assertEqual(rows[0]["candidate_origin"], "复用既有预测")
            self.assertEqual(rows[0]["source_mismatch"], "False")
            self.assertNotIn("T2", rows[0]["added_labels"])
            self.assertEqual(MODULE.audit_package(output)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
