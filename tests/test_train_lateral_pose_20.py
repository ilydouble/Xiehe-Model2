import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "3-model_training/train_lateral_pose_20.py"
SPEC = importlib.util.spec_from_file_location("train_lateral_pose_20", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def valid_line(class_id: int) -> str:
    return " ".join([str(class_id), "0.5", "0.5", "0.2", "0.2"] + ["0.4", "0.4", "2", "0.6", "0.4", "2", "0.6", "0.6", "2", "0.4", "0.6", "2"])


class TrainingScriptTests(unittest.TestCase):
    def build_dataset(self, root: Path) -> Path:
        names = "\n".join(f"  {index}: {name}" for index, name in enumerate(trainer.CLASS_NAMES))
        data_yaml = root / "data.yaml"
        data_yaml.write_text(f"nc: 20\nnames:\n{names}\nkpt_shape: [4, 3]\n", encoding="utf-8")
        manifest_rows = []
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            class_ids = range(20) if split == "train" else (0, 1)
            for class_id in class_ids:
                stem = f"{split}_{class_id}"
                (image_dir / f"{stem}.png").write_bytes(b"png")
                (label_dir / f"{stem}.txt").write_text(valid_line(class_id) + "\n", encoding="utf-8")
                manifest_rows.append(f"{stem},{split}:p{class_id},{split}\n")
        (root / "manifest.csv").write_text("output_stem,patient_id,split\n" + "".join(manifest_rows), encoding="utf-8")
        return data_yaml

    def test_validate_dataset_accepts_t13_only_in_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = trainer.validate_dataset(self.build_dataset(Path(temporary)))
            self.assertEqual(report["objects"], 24)
            self.assertEqual(report["splits"]["test"]["classes"]["C2"], 1)
            self.assertEqual(report["splits"]["test"]["classes"]["T13"], 0)

    def test_validate_dataset_rejects_triangle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yaml = self.build_dataset(root)
            bad = "0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.6 0.4 2 0.6 0.6 2 0.6 0.6 2\n"
            (root / "labels/train/train_0.txt").write_text(bad, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not unique"):
                trainer.validate_dataset(yaml)

    def test_augmentations_are_conservative(self):
        args = Namespace(data=Path("data.yaml"), epochs=2, imgsz=640, batch=2, device="0", workers=1, project=Path("runs"), name="test", exist_ok=False, lr0=0.001, patience=5, amp=True, seed=42, save_period=1, cache="none")
        result = trainer.build_train_args(args)
        self.assertTrue(result["rect"])
        self.assertEqual(result["mosaic"], 0.0)
        self.assertEqual(result["mixup"], 0.0)
        self.assertEqual(result["fliplr"], 0.5)
        self.assertFalse(result["pretrained"])

    def test_defaults_are_large_scratch_best_with_batch_four(self):
        with patch.object(sys, "argv", ["train_lateral_pose_20.py"]):
            args = trainer.parse_args()
        self.assertEqual(args.model, "yolo11l-pose.yaml")
        self.assertEqual(args.epochs, 300)
        self.assertEqual(args.imgsz, 1280)
        self.assertEqual(args.batch, 4)
        self.assertEqual(args.name, "yolo11l_lateral_20cls_scratch_best")

    def test_final_test_covers_full_spine_without_class_filter(self):
        args = Namespace(
            data=Path("data.yaml"), imgsz=1280, batch=4, device="0", workers=8,
            project=Path("runs"), name="experiment",
        )
        full = trainer.build_test_args(args)
        self.assertNotIn("classes", full)
        self.assertEqual(full["name"], "experiment_test_all_spine")

    def test_shell_best_is_single_scratch_mixed_preset(self):
        shell = (SCRIPT.parent / "train_lateral_pose_20.sh").read_text(encoding="utf-8")
        self.assertIn('--best) MODEL="yolo11l-pose.yaml"; EPOCHS=300; IMGSZ=1280; BATCH=4', shell)
        self.assertIn('DATA="${SCRIPT_DIR}/lateral_pose_20_black_roi_mixed.yaml"', shell)
        self.assertNotIn("TRANSFER_MODEL=", shell)

    def test_validate_mixed_train_keeps_holdout_original_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            datasets = base / "datasets"
            source = datasets / "yolo_lateral_reviewed_combined_20cls"
            source.mkdir(parents=True)
            self.build_dataset(source)
            roi = datasets / "yolo_lateral_reviewed_combined_20cls_black_roi"
            (roi / "images/train").mkdir(parents=True)
            (roi / "labels/train").mkdir(parents=True)
            (roi / "images/train/blackroi__train_0.png").write_bytes(b"png")
            (roi / "labels/train/blackroi__train_0.txt").write_text(valid_line(0) + "\n", encoding="utf-8")
            (roi / "manifest.csv").write_text(
                "patient_id,source_split,source_image,output_image\n"
                "train:p0,train,images/train/train_0.png,images/train/blackroi__train_0.png\n",
                encoding="utf-8",
            )
            config_dir = base / "3-model_training"
            config_dir.mkdir()
            names = "\n".join(f"  {index}: {name}" for index, name in enumerate(trainer.CLASS_NAMES))
            yaml = config_dir / "mixed.yaml"
            yaml.write_text(
                "path: ../datasets\ntrain:\n"
                "  - yolo_lateral_reviewed_combined_20cls/images/train\n"
                "  - yolo_lateral_reviewed_combined_20cls_black_roi/images/train\n"
                "val: yolo_lateral_reviewed_combined_20cls/images/val\n"
                "test: yolo_lateral_reviewed_combined_20cls/images/test\n"
                f"nc: 20\nnames:\n{names}\nkpt_shape: [4, 3]\n",
                encoding="utf-8",
            )
            report = trainer.validate_dataset(yaml)
            self.assertEqual(report["splits"]["train"]["images"], 21)
            self.assertEqual(report["splits"]["val"]["images"], 2)
            self.assertEqual(report["roi_patient_groups"], 1)


if __name__ == "__main__":
    unittest.main()
