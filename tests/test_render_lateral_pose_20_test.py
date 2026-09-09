import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_lateral_pose_20_test.py"
SPEC = importlib.util.spec_from_file_location("render_lateral_pose_20_test", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


class LateralPose20ReviewTests(unittest.TestCase):
    def test_parse_yolo_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("0 0.5 0.5 0.4 0.2 0.3 0.4 2 0.7 0.4 2 0.7 0.6 2 0.3 0.6 2\n")
            parsed = renderer.parse_yolo_pose(path, 1000, 2000)
            self.assertEqual(parsed[0]["bbox"], [300.0, 800.0, 700.0, 1200.0])
            self.assertEqual(parsed[0]["keypoints"][2], [700.0, 1200.0, 2.0])

    def test_select_predictions_keeps_highest_per_class(self):
        selected, duplicates = renderer.select_predictions([
            {"class_id": 0, "score": 0.3},
            {"class_id": 0, "score": 0.8},
            {"class_id": 1, "score": 0.7},
        ])
        self.assertEqual(selected[0]["score"], 0.8)
        self.assertEqual(duplicates, {"C2": 1})

    def test_compare_predictions_reports_missing_false_positive_and_error(self):
        ground_truth = {
            0: {"class_id": 0, "bbox": [0, 0, 10, 10], "keypoints": [[0, 0, 2], [10, 0, 2], [10, 10, 2], [0, 10, 2]]},
            1: {"class_id": 1, "bbox": [20, 20, 30, 30], "keypoints": [[20, 20, 2], [30, 20, 2], [30, 30, 2], [20, 30, 2]]},
        }
        selected = {
            0: {"class_id": 0, "score": 0.2, "bbox": [0, 0, 10, 10], "keypoints": [[3, 4, 2], [13, 4, 2], [13, 14, 2], [3, 14, 2]]},
            2: {"class_id": 2, "score": 0.8, "bbox": [40, 40, 50, 50], "keypoints": [[40, 40, 2]] * 4},
        }
        report = renderer.compare_predictions(ground_truth, selected, {"T1": 1}, 60, 80)
        self.assertEqual(report["missing_classes"], ["C7"])
        self.assertEqual(report["false_positive_classes"], ["T1"])
        self.assertEqual(report["low_confidence_classes"], ["C2"])
        self.assertEqual(report["max_keypoint_error_percent_diagonal"], 5.0)
        self.assertIn("漏检1类", report["risk_tags"])

    def test_bbox_iou(self):
        self.assertAlmostEqual(renderer.bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_remove_apple_double(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep.txt").write_text("keep")
            (root / "._keep.txt").write_text("metadata")
            self.assertEqual(renderer.remove_apple_double(root), 1)
            self.assertTrue((root / "keep.txt").exists())
            self.assertFalse((root / "._keep.txt").exists())

    def test_generated_html_keeps_javascript_newline_escaped(self):
        report = {
            "summary": {
                "images": 1,
                "first": 1,
                "second": 0,
                "risk_images": 0,
                "missing_images": 0,
                "false_positive_images": 0,
            },
            "official_metrics": {},
        }
        html_text = renderer.build_html(report)
        renderer.validate_html_runtime(html_text)
        broken = html_text.replace(chr(92) + "n')", chr(10) + "')")
        with self.assertRaisesRegex(ValueError, "invalid JavaScript newline"):
            renderer.validate_html_runtime(broken)


if __name__ == "__main__":
    unittest.main()
