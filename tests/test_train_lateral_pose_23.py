import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "3-model_training/train_lateral_pose_23.py"
SPEC = importlib.util.spec_from_file_location("train_lateral_pose_23", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def valid_line(class_id: int) -> str:
    return " ".join(
        [str(class_id), "0.5", "0.5", "0.2", "0.2"]
        + ["0.4", "0.4", "2", "0.6", "0.4", "2", "0.6", "0.6", "2", "0.4", "0.6", "2"]
    )


class TrainingScriptTests(unittest.TestCase):
    def build_dataset(self, root: Path) -> Path:
        data_yaml = root / "data.yaml"
        data_yaml.write_text("nc: 23\nkpt_shape: [4, 3]\n", encoding="utf-8")
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for class_id in range(23):
                stem = f"{split}_{class_id}"
                (image_dir / f"{stem}.png").write_bytes(b"png")
                (label_dir / f"{stem}.txt").write_text(valid_line(class_id) + "\n", encoding="utf-8")
        return data_yaml

    def test_validate_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = trainer.validate_dataset(self.build_dataset(Path(temp_dir)))
            self.assertEqual(report["objects"], 69)
            self.assertEqual(report["splits"]["train"]["images"], 23)

    def test_validate_dataset_rejects_invalid_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_yaml = self.build_dataset(root)
            (root / "labels/train/train_0.txt").write_text(valid_line(23) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class id 23"):
                trainer.validate_dataset(data_yaml)

    def test_build_train_args_keeps_medical_augmentations_conservative(self):
        args = Namespace(
            data=Path("data.yaml"), epochs=2, imgsz=640, batch=2, device="0", workers=1,
            project=Path("runs"), name="test", exist_ok=False, lr0=0.001, patience=5,
            amp=True, seed=42, save_period=1, cache="none",
        )
        result = trainer.build_train_args(args)
        self.assertTrue(result["rect"])
        self.assertEqual(result["mosaic"], 0.0)
        self.assertEqual(result["flipud"], 0.0)
        self.assertEqual(result["fliplr"], 0.5)


if __name__ == "__main__":
    unittest.main()
