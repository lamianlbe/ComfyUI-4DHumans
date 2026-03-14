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


# ---------------------------------------------------------------------------
# OpenPose 25 -> DWPose 18-body mapping (for SCAIL compatibility)
# ---------------------------------------------------------------------------
# DWPose body 18 joints: Nose, Neck, R_Shoulder, R_Elbow, R_Wrist,
# L_Shoulder, L_Elbow, L_Wrist, R_Hip, R_Knee, R_Ankle,
# L_Hip, L_Knee, L_Ankle, R_Eye, L_Eye, R_Ear, L_Ear

_OP25_TO_DW18 = {
    0: 0,    # Nose
    # 1: Neck -> synthesise from (R_Shoulder + L_Shoulder) / 2
    2: 2,    # R_Shoulder
    3: 3,    # R_Elbow
    4: 4,    # R_Wrist
    5: 5,    # L_Shoulder
    6: 6,    # L_Elbow
    7: 7,    # L_Wrist
    9: 8,    # R_Hip
    10: 9,   # R_Knee
    11: 10,  # R_Ankle
    12: 11,  # L_Hip
    13: 12,  # L_Knee
    14: 13,  # L_Ankle
    15: 14,  # R_Eye
    16: 15,  # L_Eye
    17: 16,  # R_Ear
    18: 17,  # L_Ear
}


def openpose25_to_dwpose_body(op_kp2d, img_w, img_h):
    """
    Convert OpenPose 25-joint 2D keypoints to DWPose 18-joint body format.

    Parameters
    ----------
    op_kp2d : (25, 2) or (25, 3) ndarray  – pixel coordinates
    img_w, img_h : int – image dimensions for normalisation

    Returns
    -------
    candidate : (18, 2) float32 – normalised [0, 1]
    subset : (18,) float32 – joint index if valid, -1 if missing
    """
    candidate = np.zeros((18, 2), dtype=np.float32)
    subset = np.full(18, -1.0, dtype=np.float32)

    for op_idx, dw_idx in _OP25_TO_DW18.items():
        x, y = op_kp2d[op_idx, 0], op_kp2d[op_idx, 1]
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            continue
        candidate[dw_idx, 0] = x / img_w
        candidate[dw_idx, 1] = y / img_h
        subset[dw_idx] = dw_idx

    # Neck = midpoint of R_Shoulder and L_Shoulder
    r_sh = op_kp2d[2]
    l_sh = op_kp2d[5]
    if not (abs(r_sh[0]) < 1e-6 and abs(r_sh[1]) < 1e-6) and \
       not (abs(l_sh[0]) < 1e-6 and abs(l_sh[1]) < 1e-6):
        candidate[1, 0] = (r_sh[0] + l_sh[0]) / 2 / img_w
        candidate[1, 1] = (r_sh[1] + l_sh[1]) / 2 / img_h
        subset[1] = 1.0

    return candidate, subset


def coco_wb133_to_dwpose_face_hands(coco_wb, img_w, img_h, conf_thr=0.3):
    """
    Extract face (68 pts) and hands (left 21 + right 21) from
    COCO WholeBody 133-format keypoints in DWPose format.

    Keypoints with confidence below *conf_thr* are zeroed out so the
    renderer's ``x > eps`` check will skip them (avoids spurious lines
    from noisy heatmap argmax positions).

    Parameters
    ----------
    coco_wb : (133, 3) ndarray – pixel coords + confidence
    img_w, img_h : int
    conf_thr : float – minimum confidence to keep a keypoint

    Returns
    -------
    face : (68, 2) float32 – normalised [0, 1]
    right_hand : (21, 2) float32 – normalised [0, 1]
    left_hand : (21, 2) float32 – normalised [0, 1]
    """
    # COCO-WB: face = 23..90 (68 pts), left hand = 91..111, right hand = 112..132
    face_slice = coco_wb[23:91]        # (68, 3)
    left_hand_slice = coco_wb[91:112]  # (21, 3)
    right_hand_slice = coco_wb[112:133]  # (21, 3)

    def _normalise(kp_slice, n, w, h):
        out = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            if kp_slice[i, 2] >= conf_thr:
                out[i, 0] = kp_slice[i, 0] / w
                out[i, 1] = kp_slice[i, 1] / h
        return out

    face = _normalise(face_slice, 68, img_w, img_h)
    left_hand = _normalise(left_hand_slice, 21, img_w, img_h)
    right_hand = _normalise(right_hand_slice, 21, img_w, img_h)

    return face, right_hand, left_hand


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
