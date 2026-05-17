from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from kornia.geometry.conversions import (
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
)
from scipy.spatial.transform import Rotation as R


def to_kmh(vels: Float[np.ndarray, "3"], fps: float) -> Float[np.ndarray, "3"]:
    """Convert velocity from meter/frame to km/h.
    Args:
        vels (np.array): velocity in meter/frame
        fps (int): frames per second
    Returns:
        np.array: velocity in km/h
    """
    return vels * 3.6 * fps


def to_ms(vels: Float[np.ndarray, "N 3"]) -> Float[np.ndarray, "N 3"]:
    """Convert velocity from km/h to m/s.
    Args:
        vels (np.array): velocity in km/h
    Returns:
        np.array: velocity in m/s
    """
    return vels * 0.277778


def to_homogeneous(poses: Float[np.ndarray, "N 3 4"]) -> Float[np.ndarray, "N 4 4"]:
    """
    Convert poses to homogeneous coordinates.
    Args:
        poses (np.array):
    Returns:
        np.array: poses in homogeneous coordinates
    """
    h_poses = np.zeros((len(poses), 4, 4))
    for i in range(len(poses)):
        h_poses[i] = np.vstack((poses[i], np.array([[0, 0, 0, 1]])))
    return h_poses


def get_rel_poses(
    poses: Float[np.ndarray, "N 3 4"], fps: float
) -> Float[np.ndarray, "N 3 4"]:
    """
    Get relative poses from absolute poses.
    Args:
        poses (np.array): poses
        fps (int): frame per second
    Returns:
        np.array: relative poses
    """
    h_poses = to_homogeneous(poses)
    rel_poses = np.zeros((len(poses), 3, 4))
    for i in range(1, len(poses)):
        base_pose = h_poses[i - 1]
        pose_in_base_frame = np.linalg.inv(base_pose) @ h_poses[i]
        vel = to_kmh(pose_in_base_frame[:3, -1], fps)
        rot = pose_in_base_frame[:3, :3]
        rel_poses[i] = np.concatenate([rot, vel[..., np.newaxis]], axis=-1)
    return rel_poses


def get_cum_rot(rots: Float[np.ndarray, "N 3 3"]) -> R:
    """
    Get cumulative rotation from rotation matrices.
    Args:
        rots (np.array): rotation matrices
    Returns:
        R: cumulative rotation
    """
    cum_rot = R.from_matrix(np.eye(3))
    for i in range(len(rots)):
        if np.isnan(rots[i]).any():
            continue
        rot = R.from_matrix(rots[i])
        cum_rot = cum_rot * rot
    return cum_rot


def to_traj(preds: dict, fps: float) -> Float[np.ndarray, "N 3"]:
    """
    Convert relative poses to trajectory.
    Args:
        preds (dict): predictions
        fps (int): frame per second
    Returns:
        np.array: trajectory
    """
    # make projections array
    trans = np.zeros((len(preds["z_vel"]), 3))
    trans[:, 0] = preds["x_vel"]
    trans[:, 1] = preds["y_vel"]
    trans[:, 2] = preds["z_vel"]
    trans = to_ms(trans) / fps  # to meter
    projections_array = np.concatenate(
        [
            preds["rot"],
            trans[..., np.newaxis],
        ],
        axis=-1,
    )

    traj = np.zeros((len(projections_array), 3))
    initial_pose = to_homogeneous(projections_array)[2]  # projections_array[0] is nan
    cum_rot = np.eye(3)
    cum_trans = np.zeros(3)
    for i in range(1, len(projections_array)):
        if np.isnan(projections_array[i]).any():
            traj[i] = np.nan
            continue
        rel_rot = projections_array[i, :3, :3]
        rel_trans = projections_array[i, :3, -1]
        cum_trans = cum_trans + cum_rot @ rel_trans
        cum_rot = cum_rot @ rel_rot
        curr_pose = np.concatenate([cum_rot, cum_trans[..., np.newaxis]], axis=-1)
        curr_pose = np.linalg.inv(initial_pose) @ to_homogeneous(curr_pose[None])[0]

        traj[i] = curr_pose[:3, -1]
    return traj


def compose_target_to_forward_traj(
    rel_poses: Float[np.ndarray, "N 3 4"],
) -> Float[np.ndarray, "N 3"]:
    """
    Compose trajectory from consecutive relative poses in target->forward convention.

    The input pose at index ``i`` is interpreted as ``T_{t->t+1}`` (target frame to
    forward/reference frame). For this convention, we accumulate transforms on the
    right (``T_{0->k} = T_{0->k-1} @ T_{k-1->k}``) and read translation directly
    from ``T_{0->k}``.
    """
    if rel_poses.ndim != 3 or rel_poses.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected rel_poses with shape (N,3,4), got {rel_poses.shape}"
        )
    if rel_poses.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # TODO: add first frame as (0, 0, 0)
    rel_poses = rel_poses.astype(np.float64, copy=False)
    traj = np.zeros((rel_poses.shape[0], 3), dtype=np.float64)
    T_start_to_curr = np.eye(4, dtype=np.float64)

    for i in range(rel_poses.shape[0]):
        if np.isnan(rel_poses[i]).any():
            traj[i] = np.nan
            continue

        T_curr_to_next = np.eye(4, dtype=np.float64)
        T_curr_to_next[:3, :] = rel_poses[i]
        T_start_to_curr = T_start_to_curr @ T_curr_to_next
        traj[i] = T_start_to_curr[:3, 3]

    return traj


def align_traj(
    aligned_traj: Float[np.ndarray, "N 3"], traj: Float[np.ndarray, "N 3"]
) -> Float[np.ndarray, "N 3"]:
    """
    Align trajectory.
    Args:
        aligned_traj (np.array): aligned trajectory. Usually predicted trajectory
        traj (np.array): trajectory. Usually ground truth trajectory
    Returns:
        np.array: aligned trajectory
    """
    offset = traj[0] - aligned_traj[0]
    aligned_traj += offset
    return aligned_traj


def pose_post_process(
    poses: Float[torch.Tensor, "B Refs 6"] | Float[torch.Tensor, "B Refs 7"],
    batch_size: int,
    rotation_mode: Literal["euler", "quat", "quaternion"] = "euler",
    seq_length: int = 3,
    first_forward_index: bool = False,
) -> Float[np.ndarray, "B 3 4"]:
    """
    Convert per-reference pose vectors into one transformation matrix per sample.

    The selected center-reference pose predicted by the network is inverted before
    returning (``R, t -> R^T, -R^T t``) so that downstream trajectory composition
    uses the same global convention as GT/baseline pipelines.

    Args:
        poses (torch.Tensor): relative poses for ``Seq-1`` references
        batch_size (int): batch size
        rotation_mode (Literal["euler", "quat", "quaternion"]): rotation mode
    Returns:
        np.ndarray: transformation matrices with shape ``(B, 3, 4)``
    """
    if len(poses) < batch_size:
        batch_size = len(poses)

    poses = poses[:batch_size].detach().cpu()
    num_refs = poses.size(1)
    expected_num_refs = seq_length - 1
    if num_refs != expected_num_refs:
        raise ValueError(
            f"Expected {expected_num_refs} reference poses for seq_length={seq_length}, got {num_refs}."
        )

    if first_forward_index:
        forward_idx = 0
    else:
        forward_idx = seq_length // 2 - 1
    pose = poses[:, forward_idx]

    if rotation_mode == "euler":
        euler = pose[:, :3]
        quat = torch.stack(
            quaternion_from_euler(euler[:, 0], euler[:, 1], euler[:, 2]), dim=-1
        )
        trans = pose[:, 3:6]
    elif rotation_mode in {"quat", "quaternion"}:
        quat = pose[:, :4]
        trans = pose[:, 4:7]
    else:
        raise ValueError(
            f"Unsupported rotation_mode={rotation_mode!r}. Expected 'euler', 'quat', or 'quaternion'."
        )

    rot = quaternion_to_rotation_matrix(quat)
    # rot_inv = rot.transpose(-2, -1)
    # trans_inv = -(rot_inv @ trans.unsqueeze(-1))
    projections_array = torch.cat([rot, trans.unsqueeze(-1)], dim=-1).cpu().numpy()
    # projections_array = torch.cat([rot_inv, trans.unsqueeze(-1)], dim=-1).cpu().numpy()
    # projections_array = torch.cat([rot_inv, trans_inv], dim=-1).cpu().numpy()

    return projections_array


def kpts2mask(
    kpts: Float[torch.Tensor, "N 2"],
    img_shape: tuple[int, int],
    indices: torch.Tensor | None = None,
) -> Float[torch.Tensor, "H W"]:
    """
    Convert keypoints to image mask with correspondence indices.
    Args:
        kpts (torch.Tensor): keypoints of shape [N, 2] where each row is (x, y)
        img_shape (tuple[int, int]): (H, W) of the output mask
        indices (torch.Tensor | None): 1-based correspondence indices for each keypoint.
            If None, uses sequential indices starting from 1.
    Returns:
        torch.Tensor: mask of shape [H, W] where non-zero values are correspondence indices
    """
    mask = torch.zeros((img_shape[0], img_shape[1])).to(kpts.device)
    if indices is None:
        # Use 1-based sequential indices
        indices = torch.arange(
            1, kpts.shape[0] + 1, device=kpts.device, dtype=torch.float32
        )
    mask[kpts[..., 1].round().long(), kpts[..., 0].round().long()] = indices.float()
    return mask


def mkpts2mask(
    mkpts: Float[torch.Tensor, "N 2 2"], img_shape: tuple[int, int]
) -> Float[torch.Tensor, "2 H W"]:
    N = mkpts.shape[0]
    ar = torch.arange(N, device=mkpts.device)

    # maskからkptに戻すときに対応関係を保つために、重複するkeypointは最初のものだけ残す
    # 重複しているkeypointはmask上では1つにまとめられるが、どのkeypointが残るかは不定なので、対応関係が崩れる可能性がある
    # Remove duplicates by keeping only the first occurrence of each unique keypoint
    u0, inv0 = torch.unique(mkpts[:, 0, :].round().long(), dim=0, return_inverse=True)
    first0 = torch.full((u0.shape[0],), N, device=mkpts.device, dtype=torch.long)
    first0.scatter_reduce_(0, inv0, ar, reduce="amin")  # groupごとの最小idx
    keep0 = ar == first0[inv0]  # 最初だけ True

    # --- 1側も同様 ---
    u1, inv1 = torch.unique(mkpts[:, 1, :].round().long(), dim=0, return_inverse=True)
    first1 = torch.full((u1.shape[0],), N, device=mkpts.device, dtype=torch.long)
    first1.scatter_reduce_(0, inv1, ar, reduce="amin")
    keep1 = ar == first1[inv1]

    # Combine masks
    keep = keep0 & keep1
    mkpts = mkpts[keep]

    # Use 1-based indices to preserve correspondence order
    # Both masks will have the same indices for corresponding keypoints

    indices = torch.arange(
        1, mkpts.shape[0] + 1, device=mkpts.device, dtype=torch.float32
    )

    mask0 = kpts2mask(mkpts[:, 0, :], img_shape, indices=indices)
    mask1 = kpts2mask(mkpts[:, 1, :], img_shape, indices=indices)
    if not torch.nonzero(mask0).shape[0] == torch.nonzero(mask1).shape[0]:
        print("mask0:", torch.nonzero(mask0).shape[0])
        print("mask1:", torch.nonzero(mask1).shape[0])
    assert torch.nonzero(mask0).shape[0] == torch.nonzero(mask1).shape[0], (
        "Number of keypoints do not match"
    )
    return torch.stack([mask0, mask1], dim=0)


def mask2kpts(mask: Float[torch.Tensor, "B H W"]) -> list[Float[torch.Tensor, "_ 2"]]:
    """
    Convert mask to keypoints, sorted by correspondence index stored in mask values.
    Args:
        mask (torch.Tensor): mask of shape [B, H, W] where non-zero values are correspondence indices
    Returns:
        list[torch.Tensor]: list of keypoints tensors, each of shape [N, 2] with (x, y) coordinates,
            sorted by correspondence index so that kpts[i] from mask[0] corresponds to kpts[i] from mask[1]
    """
    batch_size, H, W = mask.shape
    kpts_list = []
    for b in range(batch_size):
        ys, xs = torch.nonzero(mask[b], as_tuple=True)
        # Get correspondence indices from mask values
        indices = mask[b, ys, xs]
        # Sort by correspondence index to preserve correspondence order
        sorted_order = torch.argsort(indices)
        xs = xs[sorted_order]
        ys = ys[sorted_order]
        kpts = torch.stack([xs, ys], dim=-1)
        kpts_list.append(kpts)
    return kpts_list


def pad_keypoints(
    kpts: torch.Tensor, max_kpts: int
) -> tuple[Float[torch.Tensor, "max_kpts 2"], Float[torch.Tensor, "max_kpts"]]:
    """
    キーポイントを固定長にパディングする
    Args:
        kpts: (N, 2) tensor, [x, y]
        max_kpts: 最大キーポイント数
    Returns:
        padded_kpts: (max_kpts, 2)
        valid_mask: (max_kpts,) True if valid
    """
    num_kpts = kpts.shape[0]

    padded_kpts = torch.zeros((max_kpts, 2), dtype=torch.float32)
    valid_mask = torch.zeros(max_kpts, dtype=torch.bool)

    if num_kpts > max_kpts:
        raise ValueError(
            f"Number of keypoints ({num_kpts}) exceeds max_kpts ({max_kpts})"
        )
    padded_kpts[:num_kpts] = kpts
    valid_mask[:num_kpts] = True

    return padded_kpts, valid_mask


def pad_matches(
    matches: Float[torch.Tensor, "N 2 2"], max_matches: int
) -> tuple[Float[torch.Tensor, "max_matches 2 2"], Float[torch.Tensor, "max_matches"]]:
    """
    Pad matches to a fixed length.
    Args:
        matches (torch.Tensor): matches of shape [N, 2, 2]
        max_matches (int): maximum number of matches
    Returns:
        tuple[torch.Tensor, torch.Tensor]: padded matches of shape [max_matches, 2, 2] and valid mask of shape [max_matches]
    """
    num_matches = matches.shape[0]
    padded_matches = torch.zeros(
        (max_matches, 2, 2), dtype=matches.dtype, device=matches.device
    )
    valid_mask = torch.zeros(max_matches, dtype=torch.bool, device=matches.device)
    if num_matches > max_matches:
        raise ValueError(
            f"Number of matches ({num_matches}) exceeds max_matches ({max_matches})"
        )
    padded_matches[:num_matches] = matches[:num_matches]
    valid_mask[:num_matches] = True
    return padded_matches, valid_mask
