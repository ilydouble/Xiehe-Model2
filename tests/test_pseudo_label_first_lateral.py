import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "5-inference/pseudo_label_first_lateral.py"
SPEC = importlib.util.spec_from_file_location("pseudo_label_first_lateral", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PseudoLabelTests(unittest.TestCase):
    def candidate(self, label, x, y, view, confidence=0.5):
        points = ((x, y), (x + 10, y), (x + 10, y + 8), (x, y + 8))
        return MODULE.Candidate(
            label=label,
            box_conf=confidence,
            keypoint_conf=0.9,
            points=points,
            box=MODULE.points_box(points),
            view=view,
        )

    def test_discovery_requires_exact_stem_and_ignores_appledouble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patient = root / "P1"
            patient.mkdir()
            (patient / "paired.png").write_bytes(b"png")
            (patient / "paired.json").write_text("{}")
            (patient / "only.png").write_bytes(b"png")
            (patient / "._paired.png").write_bytes(b"metadata")
            pairs, image_only, json_only = MODULE.discover_samples(root)
            self.assertEqual([pair[0].name for pair in pairs], ["paired.png"])
            self.assertEqual([path.name for path in image_only], ["only.png"])
            self.assertEqual(json_only, [])

    def test_views_keep_guard_and_center_training_aspect_on_annotations(self):
        views = MODULE.plan_views(
            3000,
            4500,
            {"left": 450, "right": 300, "top": 0, "bottom": 0},
            [(1200, 500), (1800, 3800)],
        )
        by_name = {view.name: view for view in views}
        self.assertEqual(by_name["black_trim"], MODULE.View("black_trim", 448, 0, 2702, 4500))
        aspect = by_name["training_aspect"]
        self.assertEqual(aspect.height, 4500)
        self.assertEqual(aspect.width, round(4500 * 0.353))
        self.assertLessEqual(aspect.x0, 1200)
        self.assertGreaterEqual(aspect.x1, 1800)

    def test_consensus_prefers_two_agreeing_views_over_single_high_confidence(self):
        agreeing = [
            self.candidate("C3", 100, 100, "original", 0.35),
            self.candidate("C3", 102, 101, "black_trim", 0.30),
        ]
        outlier = self.candidate("C3", 300, 300, "training_aspect", 0.95)
        merged, cluster = MODULE.choose_candidate(
            [*agreeing, outlier],
            {"original": 0.5, "black_trim": 0.5, "training_aspect": 0.8},
        )
        self.assertEqual(len(cluster), 2)
        self.assertAlmostEqual(merged.points[0][0], 101.0)
        self.assertAlmostEqual(merged.points[0][1], 100.5)

    def test_append_preserves_c2_circle_and_existing_c7_polygon(self):
        annotation = {
            "shapes": [
                {"label": "C2", "shape_type": "circle", "points": [[5, 90], [8, 90]]},
                {"label": "C7", "shape_type": "polygon", "points": [[10, 70], [20, 70], [20, 80], [10, 80]]},
            ]
        }
        c2 = self.candidate("C2", 10, 10, "original", 0.6)
        c3a = self.candidate("C3", 10, 25, "original", 0.4)
        c3b = self.candidate("C3", 11, 25, "black_trim", 0.4)
        selected = {"C2": (c2, [c2]), "C3": (MODULE.merge_cluster([c3a, c3b]), [c3a, c3b])}
        output, added, unresolved = MODULE.append_missing_shapes(
            annotation, selected, {"original": 0.5, "black_trim": 0.5}, "a" * 64
        )
        self.assertEqual(annotation["shapes"], output["shapes"][:2])
        self.assertEqual([item["label"] for item in added], ["C2", "C3"])
        self.assertIn("C4", unresolved)
        new_c2 = output["shapes"][2]
        self.assertEqual(new_c2["shape_type"], "polygon")
        self.assertTrue(new_c2["flags"]["needs_review"])

    def test_anatomy_order_flags_reversed_prediction(self):
        annotation = {
            "shapes": [
                {"label": "C7", "shape_type": "polygon", "points": [[10, 70], [20, 70], [20, 80], [10, 80]]}
            ]
        }
        selected = {
            "C2": self.candidate("C2", 10, 50, "original"),
            "C3": self.candidate("C3", 10, 30, "original"),
        }
        validity = MODULE.anatomy_validity(annotation, selected)
        self.assertFalse(validity["C2"])
        self.assertFalse(validity["C3"])

    def test_output_refuses_existing_directory_before_inference_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            model = root / "model.pt"
            model.write_bytes(b"model")
            output = root / "output"
            output.mkdir()
            args = type("Args", (), {"source": source, "model": model, "output": output})()
            with self.assertRaises(FileExistsError):
                MODULE.run(args)


if __name__ == "__main__":
    unittest.main()
