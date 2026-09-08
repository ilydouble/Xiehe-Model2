import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_lateral_dual_model_test.py"
SPEC = importlib.util.spec_from_file_location("render_lateral_dual_model_test", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


class DualModelRendererTests(unittest.TestCase):
    def test_parse_pelvis_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "sample.txt"
            label.write_text("0 0.5 0.6 0.4 0.2 0.4 0.7 2 0.3 0.5 2 0.7 0.5 2\n")
            parsed = renderer.parse_pelvis_gt(label, 1000, 2000)
            self.assertEqual(parsed["bbox"], [300.0, 1000.0, 700.0, 1400.0])
            self.assertEqual(parsed["keypoints"][0][:2], [400.0, 1400.0])
            self.assertEqual(parsed["keypoints"][2][:2], [700.0, 1000.0])

    def test_selects_highest_spine_instance_per_class(self):
        predictions = [
            {"class_id": 0, "score": 0.4},
            {"class_id": 0, "score": 0.9},
            {"class_id": 2, "score": 0.8},
        ]
        selected, missing, duplicates = renderer.select_spine_predictions(predictions, class_count=3)
        self.assertEqual([item["score"] for item in selected], [0.9, 0.8])
        self.assertEqual(missing, ["C3"])
        self.assertEqual(duplicates, {"C2": 1})

    def test_point_errors_use_pixel_and_diagonal_units(self):
        gt = {"keypoints": [[0, 0, 2], [30, 40, 2], [100, 100, 2]]}
        pred = {"keypoints": [[3, 4, 1], [30, 40, 1], [94, 92, 1]]}
        pixels, relative = renderer.point_errors(pred, gt, 300, 400)
        self.assertEqual(pixels, [5.0, 0.0, 10.0])
        self.assertEqual(relative, [1.0, 0.0, 2.0])

    def test_risk_tags_mark_missing_and_large_point_error(self):
        tags, score = renderer.risk_tags(["C2"], {}, {"score": 0.8}, 0, [0.2, 2.1, 0.3])
        self.assertIn("脊柱缺1类", tags)
        self.assertIn("三点误差≥2%对角线", tags)
        self.assertGreater(score, 20)


if __name__ == "__main__":
    unittest.main()
