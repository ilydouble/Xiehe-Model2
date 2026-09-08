#!/usr/bin/env python3

import importlib.util
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_reviewed_lateral_pose_20.py"
SPEC = importlib.util.spec_from_file_location("build_reviewed_lateral_pose_20", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CornerConversionTests(unittest.TestCase):
    def assert_strict_convex_four(self, points):
        self.assertEqual(len(points), 4)
        self.assertEqual(len(set(points)), 4)
        MODULE.validate_corners(points)

    def test_dense_contour_reduces_to_four_real_hull_vertices(self):
        points = [
            (0, 0), (2, 0), (4, 0), (5, 1), (5, 3), (4, 4),
            (2, 4), (0, 4), (-1, 3), (-1, 1), (0, 0), (0, 0),
        ]
        corners, fallback = MODULE.dense_polygon_to_corners(points)
        self.assertFalse(fallback)
        self.assert_strict_convex_four(corners)
        self.assertTrue(set(corners) <= set(points))

    def test_triangular_hull_uses_four_corner_rectangle(self):
        points = [(0, 0), (6, 0), (3, 4), (3, 2), (0, 0)]
        corners, fallback = MODULE.dense_polygon_to_corners(points)
        self.assertTrue(fallback)
        self.assert_strict_convex_four(corners)

    def test_near_triangle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Near-triangular"):
            MODULE.validate_corners([(0, 0), (100, 0), (100, 1), (0, 1)])

    def test_pose_line_has_four_unique_keypoints(self):
        dense = [(10, 10), (30, 8), (50, 12), (52, 30), (45, 43), (20, 45), (8, 30)]
        corners, _ = MODULE.dense_polygon_to_corners(dense)
        line, clipped = MODULE.make_pose_line(1, corners, dense, 100, 100)
        self.assertEqual(clipped, 0)
        parts = line.split()
        self.assertEqual(len(parts), 17)
        keypoints = [(float(parts[index]), float(parts[index + 1])) for index in (5, 8, 11, 14)]
        self.assertEqual(len(set(keypoints)), 4)

    def test_second_batch_mapping_drops_c3_to_c6(self):
        suffix = " 0.5 0.5 0.1 0.1 0.4 0.4 2 0.6 0.4 2 0.6 0.6 2 0.4 0.6 2"
        self.assertEqual(MODULE.remap_second_line("0" + suffix, "x")[1], "C2")
        self.assertEqual(MODULE.remap_second_line("5" + suffix, "x")[1], "C7")
        self.assertEqual(MODULE.remap_second_line("6" + suffix, "x")[1], "T1")
        self.assertEqual(MODULE.remap_second_line("17" + suffix, "x")[1], "T12")
        self.assertEqual(MODULE.remap_second_line("18" + suffix, "x")[1], "L1")
        self.assertEqual(MODULE.remap_second_line("1" + suffix, "x"), (None, None))

    def test_forced_train_group_never_enters_validation(self):
        patients = {f"p{i}" for i in range(20)}
        split = MODULE.stable_group_split(patients, {"p0", "p1"}, 0.1, 0.1, 42)
        self.assertEqual(split["p0"], "train")
        self.assertEqual(split["p1"], "train")
        self.assertEqual(sum(value == "val" for value in split.values()), 2)
        self.assertEqual(sum(value == "test" for value in split.values()), 2)


if __name__ == "__main__":
    unittest.main()
