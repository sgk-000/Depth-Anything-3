import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import da3_kitti


class TestDa3Kitti(unittest.TestCase):
    def test_build_overlapping_inference_windows_names_boundary_outputs(self) -> None:
        self.assertEqual(
            da3_kitti.build_overlapping_inference_windows(300, 150),
            [
                da3_kitti.InferenceWindow("0", 0, 150),
                da3_kitti.InferenceWindow("0_1", 75, 225),
                da3_kitti.InferenceWindow("1", 150, 300),
            ],
        )

    def test_build_overlapping_inference_windows_names_end_aligned_tail(self) -> None:
        self.assertEqual(
            da3_kitti.build_overlapping_inference_windows(325, 150),
            [
                da3_kitti.InferenceWindow("0", 0, 150),
                da3_kitti.InferenceWindow("0_1", 75, 225),
                da3_kitti.InferenceWindow("1", 150, 300),
                da3_kitti.InferenceWindow("2", 175, 325),
            ],
        )

    def test_build_overlapping_batch_ranges_remains_range_only(self) -> None:
        self.assertEqual(
            da3_kitti.build_overlapping_batch_ranges(300, 150),
            [(0, 150), (75, 225), (150, 300)],
        )

    def test_resume_skips_existing_boundary_file_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_root = tmp_path / "data"
            drive_dir = data_root / "2011_09_26_drive_0001_sync_02"
            drive_dir.mkdir(parents=True)
            for idx in range(6):
                (drive_dir / f"{idx:010d}.png").touch()

            output_root = tmp_path / "output"
            existing_output = output_root / drive_dir.name / "0_1.npz"
            existing_output.parent.mkdir(parents=True)
            existing_output.write_bytes(b"already done")

            model = RecordingModel()
            with (
                patch.object(da3_kitti, "DATA_PATH", data_root),
                patch.object(da3_kitti, "OUTPUT_ROOT", output_root),
                patch.object(da3_kitti, "NUM_IMAGES", 3),
                patch.object(
                    da3_kitti,
                    "parse_kitti_intrinsics",
                    fake_parse_kitti_intrinsics,
                ),
                patch.object(da3_kitti.DepthAnything3, "from_pretrained", return_value=model),
                patch.object(da3_kitti, "tqdm", passthrough_tqdm),
            ):
                da3_kitti.run_kitti_raw_inference(resume=True)

            self.assertEqual(
                model.inference_windows,
                [
                    ["0000000000.png", "0000000001.png", "0000000002.png"],
                    ["0000000003.png", "0000000004.png", "0000000005.png"],
                ],
            )
            self.assertEqual(existing_output.read_bytes(), b"already done")


class RecordingModel:
    def __init__(self) -> None:
        self.inference_windows: list[list[str]] = []

    def to(self, device):
        return self

    def inference(self, image_paths, intrinsics, process_res):
        self.inference_windows.append([Path(path).name for path in image_paths])
        return SimpleNamespace(
            extrinsics=np.zeros((len(image_paths), 3, 4), dtype=np.float32)
        )


def fake_parse_kitti_intrinsics(scene):
    intrinsics = np.eye(3, dtype=np.float32)
    distortion = np.zeros(5, dtype=np.float32)
    rect_intrinsics = np.eye(3, dtype=np.float32)
    return intrinsics, distortion, rect_intrinsics


def passthrough_tqdm(iterable, *args, **kwargs):
    return iterable


if __name__ == "__main__":
    unittest.main()
