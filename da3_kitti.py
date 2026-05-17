import argparse
from pathlib import Path
from typing import NamedTuple
import torch
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3
import numpy as np
import pandas as pd
import cv2

GPU_ID = 0
NUM_IMAGES = 150  # They used 100 images for pose estimation in the paper
LONG_EDGE = 730  # num_images=150
MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"
DATA_PATH = Path("/home/kobayashi/dataset/kitti/raw_formatted_dataset")
OUTPUT_ROOT = Path(
    f"/home/kobayashi/dataset/kitti/depth_anything3_{NUM_IMAGES}_{LONG_EDGE}_new"
)
# LONG_EDGE = 930 # num_images=lower than 75
# LONG_EDGE = 504 # num_images=300
# LONG_EDGE = 1224 # num_images=30

# num_image=300 -> long_edge=504
# num_image=250 -> long_edge=550
# num_image=200 -> long_edge=630
# num_image=175 -> long_edge=690
# num_image=150 -> long_edge=730
# num_image=100 -> long_edge=720
# num_image=75 -> long_edge=850
# num_image=30 -> long_edge=1224


class InferenceWindow(NamedTuple):
    save_stem: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 pose inference on KITTI Raw."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip inference windows whose deterministic output file already exists."
        ),
    )
    return parser.parse_args()


def parse_kitti_intrinsics(scene) -> np.ndarray:
    drive_with_num = Path(scene).stem
    date = "_".join(drive_with_num.split("_")[:3])
    image_num = drive_with_num.split("_")[-1]
    calib_path = Path(
        f"/home/kobayashi/dataset/kitti/raw_data/{date}/calib_cam_to_cam.txt"
    )
    calib_df = pd.read_csv(
        calib_path,
        sep=r":\s+",
        header=None,
        names=["key", "raw"],
        engine="python",
        comment="#",
    )
    k_idx = calib_df[calib_df["key"].str.contains(f"K_{image_num}")].index[0]
    k_values = calib_df.loc[k_idx, "raw"].split(" ")
    k_values = [float(v) for v in k_values]
    d_idx = calib_df[calib_df["key"].str.contains(f"D_{image_num}")].index[0]
    d_values = calib_df.loc[d_idx, "raw"].split(" ")
    d_values = [float(v) for v in d_values]
    rect_idx = calib_df[calib_df["key"].str.contains(f"P_rect_{image_num}")].index[0]
    rect_values = calib_df.loc[rect_idx, "raw"].split(" ")
    rect_values = [float(v) for v in rect_values]
    rect_matrix = np.array(rect_values).astype(np.float32).reshape(3, 4)
    rect_intrinsics = rect_matrix[:, :3]

    return (
        np.array(k_values).astype(np.float32).reshape(3, 3),
        np.array(d_values).astype(np.float32),
        rect_intrinsics,
    )


def build_primary_batch_ranges(num_frames: int, window_size: int) -> list[tuple[int, int]]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but got {window_size}")
    if num_frames <= 0:
        return []
    if num_frames <= window_size:
        return [(0, num_frames)]

    batch_ranges: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= num_frames:
        batch_ranges.append((start, start + window_size))
        start += window_size

    if batch_ranges[-1][1] < num_frames:
        final_range = (num_frames - window_size, num_frames)
        if final_range not in batch_ranges:
            batch_ranges.append(final_range)

    return batch_ranges


def build_overlapping_inference_windows(
    num_frames: int, window_size: int
) -> list[InferenceWindow]:
    primary_ranges = build_primary_batch_ranges(num_frames, window_size)
    if not primary_ranges:
        return []

    half_window = max(window_size // 2, 1)
    primary_range_set = set(primary_ranges)
    inference_windows: list[InferenceWindow] = []
    seen_ranges: set[tuple[int, int]] = set()

    def add_window(save_stem: str, start: int, end: int) -> None:
        batch_range = (start, end)
        if batch_range in seen_ranges:
            return
        inference_windows.append(InferenceWindow(save_stem, start, end))
        seen_ranges.add(batch_range)

    for idx, (start, end) in enumerate(primary_ranges):
        add_window(str(idx), start, end)

        if idx + 1 >= len(primary_ranges):
            continue

        # Boundary windows are named after the two primary windows they connect:
        # the overlap between 0.npz and 1.npz is saved as 0_1.npz. This keeps the
        # fixed-window outputs easy to identify while making boundary predictions
        # explicit.
        boundary = end
        boundary_start = max(0, boundary - half_window)
        boundary_end = min(num_frames, boundary_start + window_size)
        if boundary_end - boundary_start < window_size and num_frames >= window_size:
            boundary_start = num_frames - window_size
            boundary_end = num_frames
        boundary_range = (boundary_start, boundary_end)

        # Near sequence ends, the boundary-centered window can collapse to the
        # same range as the final primary window. The primary filename wins so we
        # do not create duplicate pose entries for the same frame span.
        if boundary_range not in primary_range_set:
            add_window(f"{idx}_{idx + 1}", boundary_start, boundary_end)

    return inference_windows


def build_overlapping_batch_ranges(num_frames: int, window_size: int) -> list[tuple[int, int]]:
    return [
        (inference_window.start, inference_window.end)
        for inference_window in build_overlapping_inference_windows(num_frames, window_size)
    ]


def run_kitti_raw_inference(resume: bool = False) -> None:
    device = torch.device(f"cuda:{GPU_ID}")
    model = DepthAnything3.from_pretrained(MODEL_ID)
    model = model.to(device=device)

    drive_paths = sorted(DATA_PATH.glob("2011*"))
    for drive_path in tqdm(drive_paths, desc="drive folders"):
        image_paths = sorted(drive_path.glob("*.png"))
        image_paths = [str(p) for p in image_paths]
        if not image_paths:
            continue

        _, _, intrinsics = parse_kitti_intrinsics(drive_path)
        intrinsics = intrinsics.astype(np.float32)
        save_dir = OUTPUT_ROOT / drive_path.name

        save_dir.mkdir(parents=True, exist_ok=True)
        inference_windows = build_overlapping_inference_windows(len(image_paths), NUM_IMAGES)
        for inference_window in tqdm(
            inference_windows,
            total=len(inference_windows),
            desc="inference batches",
            leave=False,
        ):
            save_path = save_dir / f"{inference_window.save_stem}.npz"
            if resume and save_path.exists():
                print(f"Skipping existing {save_path}")
                continue

            start = inference_window.start
            end = inference_window.end
            in_image_paths = image_paths[start:end]
            batch_intrinsics = np.stack(
                [intrinsics] * len(in_image_paths), axis=0
            )
            prediction = model.inference(
                in_image_paths,
                intrinsics=batch_intrinsics,
                process_res=LONG_EDGE,
            )
            if prediction.extrinsics is None:
                raise RuntimeError(
                    f"Model did not return extrinsics for {drive_path.name} window {start}-{end}"
                )
            np.savez(
                save_path,
                extrinsics=prediction.extrinsics,
                window_start=np.array(start, dtype=np.int64),
                window_end=np.array(end, dtype=np.int64),
            )
            print(f"Saved {save_path}")


if __name__ == "__main__":
    args = parse_args()
    run_kitti_raw_inference(resume=args.resume)
