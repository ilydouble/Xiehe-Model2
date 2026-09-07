import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "1-check_data/analyze_image_appearance.py"
SPEC = importlib.util.spec_from_file_location("analyze_image_appearance", SCRIPT)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


class AppearanceAnalysisTests(unittest.TestCase):
    def test_discover_pngs_recurses_and_ignores_appledouble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "patient"
            nested.mkdir()
            expected = nested / "image.png"
            expected.touch()
            (nested / "._image.png").touch()
            (root / "note.txt").touch()
            self.assertEqual(analyzer.discover_pngs(root), [expected])

    def test_parse_pgm_preserves_whitespace_valued_first_pixel(self):
        pixels = bytes([10, 0, 100, 255])
        width, height, decoded = analyzer.parse_pgm(b"P5\n2 2\n255\n" + pixels)
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(decoded, pixels)

    def test_edge_black_bands(self):
        width = height = 10
        pixels = bytearray([100] * (width * height))
        for y in range(2):
            pixels[y * width:(y + 1) * width] = bytes([0] * width)
        for y in range(height):
            pixels[y * width] = 0
        bands = analyzer.edge_black_bands(width, height, bytes(pixels), 5, 0.98)
        self.assertEqual(bands, {"top": 2, "bottom": 0, "left": 1, "right": 0})

    def test_quantile_interpolates(self):
        self.assertEqual(analyzer.quantile([0, 10], 0.25), 2.5)

    def test_summary_counts_notable_and_severe_borders(self):
        base = {
            "width": 100, "height": 200, "megapixels": 0.02,
            "aspect_width_height": 0.5, "orientation": "portrait",
            "color_mode": "grayscale", "bit_depth": 8, "mean_gray": 50.0,
            "near_black_fraction": 0.2, "source": "SRC",
            "black_top_sample_px": 10, "black_bottom_sample_px": 10,
            "black_left_sample_px": 0, "black_right_sample_px": 0,
            "black_top_estimated_px": 20, "black_bottom_estimated_px": 20,
            "black_left_estimated_px": 0, "black_right_estimated_px": 0,
            "black_top_ratio": 0.05, "black_bottom_ratio": 0.05,
            "black_left_ratio": 0.0, "black_right_ratio": 0.0,
            "detected_black_band": True, "notable_black_border": True,
            "severe_black_border": True, "stem": "sample",
        }
        summary = analyzer.summarize([base], {
            "source": "/source", "sample_width": 256, "near_black_threshold": 5,
            "required_fraction": 0.98, "notable_ratio": 0.01,
        })
        self.assertEqual(summary["counts"]["notable_black_border"], 1)
        self.assertEqual(summary["counts"]["severe_black_border"], 1)
        self.assertEqual(summary["black_borders"]["by_side"]["top"]["notable_images"], 1)


if __name__ == "__main__":
    unittest.main()
