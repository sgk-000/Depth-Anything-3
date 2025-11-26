from pathlib import Path
import torch
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3
import numpy as np
import pandas as pd
import cv2

GPU_ID = 0
NUM_IMAGES = 100 # They used 100 images for pose estimation in the paper
LONG_EDGE = 720 # num_images=100
# LONG_EDGE = 930 # num_images=lower than 75
# LONG_EDGE = 504 # num_images=300

def parse_kitti_intrinsics(scene) -> np.ndarray:
    drive_with_num = Path(scene).stem
    date = "_".join(drive_with_num.split("_")[:3])
    image_num = drive_with_num.split("_")[-1]
    calib_path = Path(f"/home/kobayashi/dataset/kitti/raw_data/{date}/calib_cam_to_cam.txt")
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

    return np.array(k_values).astype(np.float32).reshape(3, 3), np.array(d_values).astype(np.float32)


device = torch.device(f"cuda:{GPU_ID}")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

data_path = Path("/home/kobayashi/dataset/kitti/raw_formatted_dataset")

drive_paths = sorted(data_path.glob("2011*"))
for drive_path in tqdm(drive_paths, desc=f"drive folders"):
    image_paths = sorted(drive_path.glob("*.png"))
    image_paths = [str(p) for p in image_paths]
    intrinsics, distortion = parse_kitti_intrinsics(drive_path)
    intrinsics_opt, _ = cv2.getOptimalNewCameraMatrix(
        intrinsics,
        distortion,
        (cv2.imread(image_paths[0]).shape[1], cv2.imread(image_paths[0]).shape[0]),
        1,
    )
    save_dir = (
        Path(f"/home/kobayashi/dataset/kitti/depth_anything3_{NUM_IMAGES}") / drive_path.name
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    num_inference = int(len(image_paths) / NUM_IMAGES) + 1
    for idx in tqdm(range(num_inference), desc="inference batches"):
        save_path = save_dir / f"{idx}.npz"    
        if save_path.exists():
            continue
        in_image_idx = range(
            idx * NUM_IMAGES, (idx + 1) * NUM_IMAGES
        )
        if max(in_image_idx) >= len(image_paths):
            in_image_idx = range(
                len(image_paths) - NUM_IMAGES, len(image_paths)
            )
        in_image_paths = [image_paths[i] for i in in_image_idx]
        intrinsics_opt = np.stack(
            [intrinsics_opt.astype(np.float32)] * len(in_image_paths), axis=0
        )
        prediction = model.inference(
            in_image_paths,
            intrinsics=intrinsics_opt,
            process_res=LONG_EDGE,
        )
        np.savez(
            save_path,
            depth=prediction.depth,
            conf=prediction.conf,
            extrinsics=prediction.extrinsics,
            intrinsics=prediction.intrinsics,
        )
            
