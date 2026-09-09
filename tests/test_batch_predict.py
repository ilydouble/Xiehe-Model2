import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "5-inference/batch_predict.py"
SPEC = importlib.util.spec_from_file_location("batch_predict", SCRIPT)
predictor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(predictor)


class CombinedLateralInferenceTests(unittest.TestCase):
    def test_spine_selection_keeps_highest_confidence_per_class(self):
        selected = predictor.select_spine_predictions([
            {"class_id": 0, "confidence": 0.31},
            {"class_id": 0, "confidence": 0.86},
            {"class_id": 2, "confidence": 0.72},
        ])
        self.assertEqual([(row["class_id"], row["confidence"]) for row in selected], [(0, 0.86), (2, 0.72)])

    def test_pelvis_selection_keeps_highest_confidence_instance(self):
        selected = predictor.select_pelvis_prediction([
            {"confidence": 0.42},
            {"confidence": 0.91},
        ])
        self.assertEqual(selected["confidence"], 0.91)

    def test_discover_images_is_sorted_and_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "b.JPG").write_bytes(b"")
            (root / "nested/a.png").write_bytes(b"")
            (root / "ignore.txt").write_text("x")
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in predictor.discover_images(root)],
                ["b.JPG", "nested/a.png"],
            )

    def test_serialized_output_has_named_spine_and_pelvis_points(self):
        spine = [{"class_id": 0, "confidence": 0.8, "bbox_xyxy": [1, 2, 3, 4], "keypoints": [[1, 2, 0.9]] * 4}]
        pelvis = {"class_id": 0, "confidence": 0.9, "bbox_xyxy": [4, 5, 6, 7], "keypoints": [[10, 11, 0.8], [20, 21, 0.8], [30, 31, 0.8]]}
        record = predictor.serialize_record(Path("sample.png"), Path("sample.png"), 100, 200, spine, pelvis)
        self.assertEqual(record["spine_predictions"][0]["class_name"], "C2")
        self.assertEqual(record["pelvis_prediction"]["named_keypoints"]["CFH"], [10, 11, 0.8])
        self.assertEqual(record["pelvis_prediction"]["named_keypoints"]["S1_right"], [30, 31, 0.8])


if __name__ == "__main__":
    unittest.main()
