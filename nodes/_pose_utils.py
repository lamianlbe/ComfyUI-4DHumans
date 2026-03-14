"""Shared utilities for pose estimation nodes."""

import numpy as np


# ---------------------------------------------------------------------------
# OpenPose 25 -> COCO WholeBody 133 joint mapping
# ---------------------------------------------------------------------------
# Maps OpenPose body index -> COCO-WB index.
# 23 of 25 OpenPose joints have a 1:1 mapping.
# OpenPose 1 (Neck) and 8 (MidHip) have no direct COCO equivalent.

OPENPOSE25_TO_COCO_WB = {
    0: 0,    # Nose
    # 1: Neck  -> no direct mapping
    2: 6,    # R_Shoulder
    3: 8,    # R_Elbow
    4: 10,   # R_Wrist
    5: 5,    # L_Shoulder
    6: 7,    # L_Elbow
    7: 9,    # L_Wrist
    # 8: MidHip -> no direct mapping
    9: 12,   # R_Hip
    10: 14,  # R_Knee
    11: 16,  # R_Ankle
    12: 11,  # L_Hip
    13: 13,  # L_Knee
    14: 15,  # L_Ankle
    15: 2,   # R_Eye
    16: 1,   # L_Eye
    17: 4,   # R_Ear
    18: 3,   # L_Ear
    19: 17,  # L_BigToe
    20: 18,  # L_SmallToe
    21: 19,  # L_Heel
    22: 20,  # R_BigToe
    23: 21,  # R_SmallToe
    24: 22,  # R_Heel
}


def openpose25_to_coco_wholebody(op_kp2d):
    """
    Convert OpenPose 25-joint body keypoints to COCO WholeBody 133 format.

    Only body (0-16) and feet (17-22) slots are filled.
    Face (23-90) and hands (91-132) remain zero.

    Parameters
    ----------
    op_kp2d : (25, 2) or (25, 3) ndarray
        OpenPose 2D keypoints. If 3 columns, col 2 is confidence.

    Returns
    -------
    coco_wb : (133, 3) float32 array  (x, y, confidence)
    """
    coco_wb = np.zeros((133, 3), dtype=np.float32)
    has_conf = op_kp2d.shape[1] >= 3

    for op_idx, coco_idx in OPENPOSE25_TO_COCO_WB.items():
        coco_wb[coco_idx, 0] = op_kp2d[op_idx, 0]
        coco_wb[coco_idx, 1] = op_kp2d[op_idx, 1]
        coco_wb[coco_idx, 2] = op_kp2d[op_idx, 2] if has_conf else 1.0

    return coco_wb


def fuse_3d_body_with_sapiens(op_kp2d, sapiens_kp):
    """
    Fuse PromptHMR 3D body+feet with Sapiens face+hands.

    Body+feet (COCO-WB 0-22) come from PromptHMR's OpenPose 25 joints.
    Face (23-90) and hands (91-132) come from Sapiens if available.

    Parameters
    ----------
    op_kp2d : (25, 2) or (25, 3) ndarray
        OpenPose 25-joint 2D keypoints from PromptHMR.
    sapiens_kp : (133, 3) ndarray or None
        Sapiens COCO-WholeBody keypoints. If None, face/hands are zero.

    Returns
    -------
    coco_wb : (133, 3) float32 array
    """
    # Start with 3D body+feet
    coco_wb = openpose25_to_coco_wholebody(op_kp2d)

    # Fill face + hands from Sapiens
    if sapiens_kp is not None:
        coco_wb[23:91] = sapiens_kp[23:91]    # face (68 keypoints)
        coco_wb[91:133] = sapiens_kp[91:133]  # hands (42 keypoints)

    return coco_wb


# ---------------------------------------------------------------------------
# Frame rate resampling utilities
# ---------------------------------------------------------------------------

def resample_keypoints(timeline, fps_in, target_fps=30.0):
    """
    Resample a single-person keypoint timeline from fps_in to target_fps
    using linear interpolation between adjacent frames.

    Parameters
    ----------
    timeline : list of (K, 3) arrays or None
    fps_in : float
    target_fps : float

    Returns
    -------
    resampled : list of (K, 3) arrays or None
    src_indices : list of int, nearest source frame index per output frame
    """
    n_in = len(timeline)
    if n_in < 2:
        return list(timeline), list(range(n_in))

    duration = (n_in - 1) / fps_in
    n_out = max(1, int(round(duration * target_fps)) + 1)

    resampled = []
    src_indices = []
    for i in range(n_out):
        t_sec = i / target_fps
        t_in = t_sec * fps_in

        j0 = min(int(t_in), n_in - 1)
        j1 = min(j0 + 1, n_in - 1)
        alpha = t_in - j0

        src_indices.append(min(int(round(t_in)), n_in - 1))

        if j0 == j1 or alpha < 1e-6:
            resampled.append(
                timeline[j0].copy() if timeline[j0] is not None else None)
        elif timeline[j0] is not None and timeline[j1] is not None:
            resampled.append(
                timeline[j0] * (1 - alpha) + timeline[j1] * alpha)
        elif timeline[j0] is not None:
            resampled.append(timeline[j0].copy())
        elif timeline[j1] is not None:
            resampled.append(timeline[j1].copy())
        else:
            resampled.append(None)

    return resampled, src_indices


def compute_resampled_indices(n_in, fps_in, target_fps=30.0):
    """
    Compute nearest source frame indices for resampling from fps_in to
    target_fps. Works for both upsampling and downsampling.

    Returns list of source frame indices (length = output frame count).
    """
    if n_in < 2:
        return list(range(n_in))

    duration = (n_in - 1) / fps_in
    n_out = max(1, int(round(duration * target_fps)) + 1)

    indices = []
    for i in range(n_out):
        t_sec = i / target_fps
        j = min(int(round(t_sec * fps_in)), n_in - 1)
        indices.append(j)

    return indices
