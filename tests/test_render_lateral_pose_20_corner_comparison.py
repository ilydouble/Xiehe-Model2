import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_lateral_pose_20_corner_comparison.py"
SPEC = importlib.util.spec_from_file_location("corner_comparison", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class LegacyComparisonTests(unittest.TestCase):
    def test_legacy_replay_can_return_duplicate_coordinates(self):
        points = [(0, 0), (0, 0), (10, 1), (10, -1), (5, 0)]
        selected = module.legacy_farthest_four(points)
        self.assertEqual(len(selected), 4)
        self.assertLess(len(set(selected)), 4)
        self.assertEqual(module.edge_ratio(selected), 0.0)

    def test_unique_image_selection_keeps_worst_record_per_image(self):
        records = [
            {"source_stem": "a", "old_edge_ratio": 0.01},
            {"source_stem": "a", "old_edge_ratio": 0.02},
            {"source_stem": "b", "old_edge_ratio": 0.03},
        ]
        selected = module.select_unique_images(records, 2)
        self.assertEqual([record["source_stem"] for record in selected], ["a", "b"])
        self.assertEqual(selected[0]["old_edge_ratio"], 0.01)


if __name__ == "__main__":
    unittest.main()
