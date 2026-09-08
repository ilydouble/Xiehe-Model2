#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "1-check_data/audit_yolo_corner_label_sources.py"
SPEC = importlib.util.spec_from_file_location("audit_yolo_corner_label_sources", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def annotation(image_name: str, x0: float) -> dict:
    points = [[x0, 10], [x0 + 20, 10], [x0 + 20, 30], [x0, 30]]
    return {
        "imagePath": image_name,
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [{"label": "C7", "shape_type": "polygon", "points": points}],
    }


class AuditTests(unittest.TestCase):
    def test_finds_same_stem_and_other_json_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, raw = root / "dataset", root / "raw"
            patient = raw / "PATIENT_1"
            patient.mkdir(parents=True)
            for split in MODULE.SPLITS:
                (dataset / "images" / split).mkdir(parents=True)
                (dataset / "labels" / split).mkdir(parents=True)
            for name in ("a.png", "b.png", "c.png"):
                Image.new("L", (100, 100), 0).save(patient / name)
            (patient / "a.json").write_text(json.dumps(annotation("a.png", 10)), encoding="utf-8")
            (patient / "c.json").write_text(json.dumps(annotation("c.png", 50)), encoding="utf-8")
            for name in ("a.png", "b.png", "c.png"):
                Image.open(patient / name).save(dataset / "images" / "train" / name)
            signature_a = MODULE.converted_signature(patient / "a.json", 100, 100)
            text_a = "\n".join(signature_a.values())
            (dataset / "labels" / "train" / "a.txt").write_text(text_a, encoding="utf-8")
            (dataset / "labels" / "train" / "b.txt").write_text(text_a, encoding="utf-8")
            (dataset / "labels" / "train" / "c.txt").write_text(text_a, encoding="utf-8")

            summary, rows = MODULE.audit(dataset, raw)
            by_name = {row["filename"]: row for row in rows}
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["status"], {"correct_same_stem": 1, "definite_mismatch": 2})
            self.assertEqual(by_name["a.png"]["status"], "correct_same_stem")
            self.assertEqual(by_name["b.png"]["matched_jsons"], "a.json")
            self.assertFalse(by_name["b.png"]["same_stem_json_exists"])
            self.assertEqual(by_name["c.png"]["matched_jsons"], "a.json")
            self.assertTrue(by_name["c.png"]["same_stem_json_exists"])
            self.assertEqual(summary["requires_full_23cls_reannotation_count"], 2)


if __name__ == "__main__":
    unittest.main()
