import csv
import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "4-model_training_CFH/train_pelvis_pose_3kpt.py"
SPEC = importlib.util.spec_from_file_location("train_pelvis_pose_3kpt", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def valid_line(class_id: int = 0) -> str:
    return " ".join([
        str(class_id), "0.5", "0.7", "0.7", "0.4",
        "0.5", "0.6", "2", "0.2", "0.8", "2", "0.8", "0.8", "2",
    ])


class TrainingScriptTests(unittest.TestCase):
    def build_dataset(self, root: Path) -> Path:
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            "nc: 1\nkpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\n", encoding="utf-8"
        )
        manifest_rows = []
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            stem = f"{split}_sample"
            (image_dir / f"{stem}.png").write_bytes(b"png")
            (label_dir / f"{stem}.txt").write_text(valid_line() + "\n", encoding="utf-8")
            manifest_rows.append({"group_id": f"old:{split}", "split": split})
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["group_id", "split"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        return data_yaml

    def test_validate_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = trainer.validate_dataset(self.build_dataset(Path(temp_dir)))
            self.assertEqual(report["objects"], 3)
            self.assertEqual(report["patients"], 3)

    def test_validate_dataset_rejects_detection_or_wrong_pose_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_yaml = self.build_dataset(root)
            (root / "labels/train/train_sample.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected 14 fields"):
                trainer.validate_dataset(data_yaml)

    def test_validate_dataset_rejects_patient_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_yaml = self.build_dataset(root)
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["group_id", "split"])
                writer.writeheader()
                writer.writerows([
                    {"group_id": "P1", "split": "train"},
                    {"group_id": "P1", "split": "val"},
                    {"group_id": "P2", "split": "test"},
                ])
            with self.assertRaisesRegex(ValueError, "cross splits"):
                trainer.validate_dataset(data_yaml)

    def test_build_train_args_uses_pose_safe_augmentation(self):
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
        self.assertEqual(result["max_det"], 5)


if __name__ == "__main__":
    unittest.main()
