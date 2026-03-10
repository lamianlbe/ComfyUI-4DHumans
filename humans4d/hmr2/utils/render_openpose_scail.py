"""
Render SCAIL-style pose images from SMPLest-X 137-joint keypoints.

Uses the original SCAIL rendering pipeline (copied into scail/ subpackage):
  - Body: 3D cylinder ray marching (Taichi or PyTorch) via render_whole
  - Hands + Face: 2D DWPose overlay via draw_pose

This module converts SMPLest-X data formats into the NLF 3D + DWPose 2D
formats expected by the SCAIL renderer, and handles the coordinate system
transformation between SMPLest-X camera space and output image space.

For each frame, the inv_trans affine matrix is used to compute output-space
camera intrinsics, allowing direct rendering at output resolution without
intermediate warpAffine.
"""
import logging

import cv2
import numpy as np

from .scail.nlf_render import get_single_pose_cylinder_specs, process_data_to_COCO_format
from .scail.draw_pose_utils import draw_pose_to_canvas_np

try:
    from .scail.taichi_cylinder import render_whole as render_whole_taichi
except Exception:
    render_whole_taichi = None

from .scail.render_torch import render_whole_batch as render_whole_batch_torch


# ── FLAME 72-face → iBUG 68-face mapping ─────────────────────────────────────
# SMPLest-X face points (offset from index 65) use FLAME ordering.
# DWPose draw_facepose expects iBUG 68-point ordering.
# Offsets 0-3 (jaw, head, eyeballs) have no iBUG equivalent and are skipped.
_FLAME_TO_IBUG68 = {}
for _i in range(17): _FLAME_TO_IBUG68[55 + _i] = _i        # contour → iBUG 0-16
for _i in range(5):  _FLAME_TO_IBUG68[4 + _i]  = 17 + _i   # L eyebrow → iBUG 17-21
for _i in range(5):  _FLAME_TO_IBUG68[9 + _i]  = 22 + _i   # R eyebrow → iBUG 22-26
for _i in range(4):  _FLAME_TO_IBUG68[14 + _i] = 27 + _i   # nose bridge → iBUG 27-30
for _i in range(5):  _FLAME_TO_IBUG68[18 + _i] = 31 + _i   # nose bottom → iBUG 31-35
for _i in range(6):  _FLAME_TO_IBUG68[23 + _i] = 36 + _i   # L eye → iBUG 36-41
for _i in range(6):  _FLAME_TO_IBUG68[29 + _i] = 42 + _i   # R eye → iBUG 42-47
for _i in range(12): _FLAME_TO_IBUG68[35 + _i] = 48 + _i   # outer mouth → iBUG 48-59
for _i in range(8):  _FLAME_TO_IBUG68[47 + _i] = 60 + _i   # inner mouth → iBUG 60-67


# ── SMPLest-X → SMPL 24-joint mapping ────────────────────────────────────────
# Maps SMPLest-X 137-joint indices to SMPL 24-joint format for the cylinder
# renderer. Note: index 66 = Face_2 = SMPL-X joint 15 (Head top), which is
# the correct Head joint (not index 24 = Nose which would make neck too long).
_SMPLESTX_TO_SMPL24 = {
    0: 0,    # Pelvis
    1: 1,    # L.Hip
    2: 2,    # R.Hip
    3: 4,    # L.Knee
    4: 5,    # R.Knee
    5: 7,    # L.Ankle
    6: 8,    # R.Ankle
    7: 12,   # Neck
    8: 16,   # L.Shoulder
    9: 17,   # R.Shoulder
    10: 18,  # L.Elbow
    11: 19,  # R.Elbow
    12: 20,  # L.Wrist
    13: 21,  # R.Wrist
    66: 15,  # Face_2 (SMPL-X Head joint) → Head
}


def _smplestx_3d_to_smpl24(joint_cam_3d, root_cam):
    """
    Convert SMPLest-X root-relative 3D joints to SMPL 24-joint
    absolute camera-space format for the NLF cylinder renderer.
    """
    joints_abs = joint_cam_3d + root_cam[None, :]  # all 137 joints
    smpl24 = np.zeros((24, 3), dtype=np.float32)
    for src, dst in _SMPLESTX_TO_SMPL24.items():
        smpl24[dst] = joints_abs[src]

    # Shorten neck: SMPL-X Head (joint 15) is at the skull top, which makes
    # the neck→head bone too long compared to NLF/SCAIL. Move head position
    # to 60% of the way from neck to head top for a more natural look.
    neck = smpl24[12]  # Neck
    head = smpl24[15]  # Head (skull top)
    if np.sum(np.abs(neck)) > 0.01 and np.sum(np.abs(head)) > 0.01:
        smpl24[15] = neck + 0.5 * (head - neck)

    return smpl24


def _smplestx_2d_to_dwpose(keypoints_2d, img_h, img_w, threshold=0.1):
    """
    Convert SMPLest-X 137-joint 2D keypoints to DWPose dict format.
    DWPose uses normalized coordinates (x/W, y/H).
    """
    kp = keypoints_2d

    # ── Body candidate (18 joints, normalized) ────────────────────────────
    smplestx_to_coco18 = {
        24: 0,   # Nose
        7: 1,    # Neck
        9: 2,    # R.Shoulder
        11: 3,   # R.Elbow
        13: 4,   # R.Wrist
        8: 5,    # L.Shoulder
        10: 6,   # L.Elbow
        12: 7,   # L.Wrist
        2: 8,    # R.Hip
        4: 9,    # R.Knee
        6: 10,   # R.Ankle
        1: 11,   # L.Hip
        3: 12,   # L.Knee
        5: 13,   # L.Ankle
        23: 14,  # R.Eye
        22: 15,  # L.Eye
        21: 16,  # R.Ear
        20: 17,  # L.Ear
    }

    candidate = np.zeros((18, 2), dtype=np.float32)
    subset = np.zeros((1, 18), dtype=np.float32)

    for src, dst in smplestx_to_coco18.items():
        if src < kp.shape[0] and kp[src, 2] > threshold:
            candidate[dst, 0] = kp[src, 0] / img_w
            candidate[dst, 1] = kp[src, 1] / img_h
            subset[0, dst] = dst
        else:
            subset[0, dst] = -1

    # ── Hands (21 joints each, normalized) ────────────────────────────────
    hands = []
    for hand_start, wrist_idx in [(45, 13), (25, 12)]:  # right, left (DWPose order)
        hand = np.zeros((21, 2), dtype=np.float32)
        if kp[wrist_idx, 2] > threshold:
            hand[0, 0] = kp[wrist_idx, 0] / img_w
            hand[0, 1] = kp[wrist_idx, 1] / img_h
        for j in range(20):
            idx = hand_start + j
            if idx < kp.shape[0] and kp[idx, 2] > threshold:
                hand[j + 1, 0] = kp[idx, 0] / img_w
                hand[j + 1, 1] = kp[idx, 1] / img_h
        hands.append(hand.tolist())

    # ── Face (68 landmarks, normalized) ───────────────────────────────────
    # SMPLest-X face (indices 65-136) uses FLAME ordering (72 points):
    #   offset 0-1: jaw, head  |  2-3: eyeballs  |  4-13: eyebrows (10)
    #   14-17: nose (4)  |  18-22: below nose (5)  |  23-34: eyes (12)
    #   35-46: outer mouth (12)  |  47-54: inner mouth (8)  |  55-71: contour (17)
    # DWPose expects iBUG 68 ordering:
    #   0-16: contour  |  17-21: L eyebrow  |  22-26: R eyebrow
    #   27-30: nose bridge  |  31-35: nose bottom  |  36-41: L eye
    #   42-47: R eye  |  48-59: outer mouth  |  60-67: inner mouth
    face = np.zeros((68, 2), dtype=np.float32)
    for flame_off, ibug_idx in _FLAME_TO_IBUG68.items():
        idx = 65 + flame_off  # absolute index in 137-joint array
        if idx < kp.shape[0] and kp[idx, 2] > threshold:
            face[ibug_idx, 0] = kp[idx, 0] / img_w
            face[ibug_idx, 1] = kp[idx, 1] / img_h

    return {
        "bodies": {
            "candidate": [candidate.tolist()],
            "subset": [subset[0].tolist()],
        },
        "faces": [face.tolist()],
        "hands": hands,
    }


def _compute_output_intrinsics(fx_img, fy_img, cx_img, cy_img, inv_trans):
    """
    Transform camera intrinsics from input_img_shape space to output image
    space using the inv_trans affine matrix.

    For axis-aligned bboxes, inv_trans is [sx 0 tx; 0 sy ty], giving:
      fx' = sx * fx,  cx' = sx * cx + tx
      fy' = sy * fy,  cy' = sy * cy + ty

    For general affine (with rotation), this is approximate but sufficient.
    """
    # inv_trans is (2, 3): [[a, b, tx], [c, d, ty]]
    sx = inv_trans[0, 0]
    sy = inv_trans[1, 1]
    tx = inv_trans[0, 2]
    ty = inv_trans[1, 2]

    fx_out = sx * fx_img
    fy_out = sy * fy_img
    cx_out = sx * cx_img + tx
    cy_out = sy * cy_img + ty

    return fx_out, fy_out, cx_out, cy_out


def render_scail_pose_batch(timeline, timeline_3d, img_h, img_w, cfg=None,
                            render_backend="taichi"):
    """
    Batch-render SCAIL-style pose images for a full timeline.

    For each frame, computes output-space camera intrinsics from inv_trans
    and renders 3D cylinders directly at output resolution, then overlays
    2D hand/face drawing.

    Args:
        timeline: list of (137, 3) 2D keypoint arrays (or None per frame)
        timeline_3d: list of dicts {"joint_cam", "root_cam", "inv_trans"} (or None)
        img_h, img_w: output image dimensions
        cfg: SMPLest-X config (required for intrinsic matrix computation)
        render_backend: "taichi" or "torch"

    Returns:
        list of (H, W, 3) uint8 images
    """
    import copy

    B = len(timeline)

    # ── SMPLest-X intrinsics scaled to input_img_shape ───────────────────
    focal = cfg.model.focal          # (fx, fy)
    princpt = cfg.model.princpt      # (cx, cy)
    input_body = cfg.model.input_body_shape  # (H, W)
    input_img = cfg.model.input_img_shape    # (H, W)

    scale_x = input_img[1] / input_body[1]  # 384/192 = 2.0
    scale_y = input_img[0] / input_body[0]  # 512/256 = 2.0

    fx_img = focal[0] * scale_x    # 5000 * 2 = 10000
    fy_img = focal[1] * scale_y    # 5000 * 2 = 10000
    cx_img = princpt[0] * scale_x  # 96 * 2 = 192
    cy_img = princpt[1] * scale_y  # 128 * 2 = 256

    # ── Build SMPL 24-joint 3D poses + per-frame output intrinsics ───────
    smpl_poses = []
    frame_intrinsics = []  # (fx, fy, cx, cy) per frame in output space
    for t in range(B):
        if timeline_3d is not None and t < len(timeline_3d) and timeline_3d[t] is not None:
            d3d = timeline_3d[t]
            smpl24 = _smplestx_3d_to_smpl24(d3d["joint_cam"], d3d["root_cam"])
            smpl_poses.append([smpl24])
            fx_o, fy_o, cx_o, cy_o = _compute_output_intrinsics(
                fx_img, fy_img, cx_img, cy_img, d3d["inv_trans"])
            frame_intrinsics.append((fx_o, fy_o, cx_o, cy_o))
        else:
            smpl_poses.append([np.zeros((24, 3), dtype=np.float32)])
            frame_intrinsics.append((fx_img, fy_img, cx_img, cy_img))

    # ── Compute dynamic cylinder radius ──────────────────────────────────
    # NLF reference: radius=21.5 at focal≈700, Z≈400
    #   → projected ≈ 21.5*700/400 ≈ 37.6 px on 720px height
    # We target similar visual proportion at output resolution.
    z_values = []
    for t in range(B):
        if timeline_3d is not None and t < len(timeline_3d) and timeline_3d[t] is not None:
            root_z = timeline_3d[t]["root_cam"][2]
            if root_z > 0:
                z_values.append(root_z)

    avg_z = np.mean(z_values) if z_values else 3000.0
    # Use the average output focal to compute radius for desired visual thickness
    avg_fy_out = np.mean([fi[1] for fi in frame_intrinsics])
    target_px = 0.015 * img_h  # ~1.5% of output height
    radius = target_px * avg_z / avg_fy_out

    # ── Build cylinder specs + colors ────────────────────────────────────
    base_colors_255 = [
        [255, 0, 0], [0, 255, 255], [255, 85, 0], [255, 170, 0],
        [0, 170, 255], [0, 85, 255], [180, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [150, 150, 150], [255, 0, 170], [50, 0, 255],
        [255, 0, 170], [50, 0, 255],
    ]
    colors = [[c / 300 + 0.15 for c in rgb] + [0.8] for rgb in base_colors_255]

    limb_seq = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    ]
    draw_seq = [0, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # ── Batch render all frames with per-frame intrinsics ──────────────
    cylinder_specs_list = []
    for t in range(B):
        specs = get_single_pose_cylinder_specs(
            (t, smpl_poses[t], None, None, None, None, colors, limb_seq, draw_seq))
        cylinder_specs_list.append(specs)

    if render_backend == "taichi" and render_whole_taichi is not None:
        # Taichi doesn't support per-frame intrinsics, use first frame's
        fx_0, fy_0, cx_0, cy_0 = frame_intrinsics[0]
        try:
            frames_rgba = render_whole_taichi(
                cylinder_specs_list, H=img_h, W=img_w,
                fx=fx_0, fy=fy_0, cx=cx_0, cy=cy_0, radius=radius)
        except Exception:
            logging.warning("Taichi rendering failed, falling back to torch.")
            frames_rgba = render_whole_batch_torch(
                cylinder_specs_list, H=img_h, W=img_w, radius=radius,
                intrinsics_list=frame_intrinsics)
    else:
        frames_rgba = render_whole_batch_torch(
            cylinder_specs_list, H=img_h, W=img_w, radius=radius,
            intrinsics_list=frame_intrinsics)

    # ── Build DWPose 2D overlay at output resolution ─────────────────────
    dw_poses = []
    for t in range(B):
        if timeline[t] is not None:
            dw_poses.append(
                _smplestx_2d_to_dwpose(timeline[t], img_h, img_w))
        else:
            dw_poses.append({
                "bodies": {
                    "candidate": [np.zeros((18, 2)).tolist()],
                    "subset": [[-1] * 18],
                },
                "faces": [np.zeros((68, 2)).tolist()],
                "hands": [np.zeros((21, 2)).tolist(), np.zeros((21, 2)).tolist()],
            })

    canvas_2d = draw_pose_to_canvas_np(
        copy.deepcopy(dw_poses), pool=None, H=img_h, W=img_w, reshape_scale=0,
        show_feet_flag=False, show_body_flag=False, show_cheek_flag=True,
        dw_hand=True, show_face_flag=True, show_hand_flag=True)

    # ── Composite: 3D cylinders + 2D overlay ─────────────────────────────
    result = []
    for t in range(B):
        frame = frames_rgba[t]
        if frame.shape[2] == 4:
            rgb = frame[:, :, :3].copy()
        else:
            rgb = frame.copy()

        # Overlay 2D hand/face/cheekbone drawing
        canvas_img = canvas_2d[t]
        mask_2d = canvas_img != 0
        rgb[mask_2d] = canvas_img[mask_2d]

        result.append(rgb)

    return result
