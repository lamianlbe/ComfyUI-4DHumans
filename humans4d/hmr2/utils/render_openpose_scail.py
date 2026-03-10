"""
Render SCAIL-style pose images from SMPLest-X 137-joint keypoints.

Uses the original SCAIL rendering pipeline (copied into scail/ subpackage):
  - Body: 3D cylinder ray marching (Taichi or PyTorch) via render_whole
  - Hands + Face: 2D DWPose overlay via draw_pose

This module converts SMPLest-X data formats into the NLF 3D + DWPose 2D
formats expected by the SCAIL renderer, and handles the coordinate system
transformation between SMPLest-X camera space and output image space.

Key insight: SMPLest-X 3D joints live in a camera space with
focal=(5000,5000) at input_body_shape=(256,192). We must render cylinders
using these intrinsics (scaled to input_img_shape), then warpAffine each
frame to output resolution using per-frame inv_trans.
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

from .scail.render_torch import render_whole as render_whole_torch


# ── SMPLest-X 25-body → SMPL 24-joint mapping ───────────────────────────────
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
    24: 15,  # Nose → Head
}


def _smplestx_3d_to_smpl24(joint_cam_3d, root_cam):
    """
    Convert SMPLest-X root-relative 3D body joints to SMPL 24-joint
    absolute camera-space format for the NLF cylinder renderer.
    """
    joints_abs = joint_cam_3d[:25] + root_cam[None, :]
    smpl24 = np.zeros((24, 3), dtype=np.float32)
    for src, dst in _SMPLESTX_TO_SMPL24.items():
        smpl24[dst] = joints_abs[src]
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
    face = np.zeros((68, 2), dtype=np.float32)
    for j in range(min(68, 72)):
        idx = 65 + j
        if idx < kp.shape[0] and kp[idx, 2] > threshold:
            face[j, 0] = kp[idx, 0] / img_w
            face[j, 1] = kp[idx, 1] / img_h

    return {
        "bodies": {
            "candidate": [candidate.tolist()],
            "subset": [subset[0].tolist()],
        },
        "faces": [face.tolist()],
        "hands": hands,
    }


def render_scail_pose_batch(timeline, timeline_3d, img_h, img_w, cfg=None,
                            render_backend="taichi"):
    """
    Batch-render SCAIL-style pose images for a full timeline.

    Strategy:
      1. Render 3D cylinders at input_img_shape resolution using SMPLest-X
         intrinsics (scaled from input_body_shape to input_img_shape).
      2. Per-frame warpAffine using inv_trans to map rendered cylinders
         from input_img_shape to output image space.
      3. Draw 2D hand/face overlay in output image space.

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
    # cfg.model.focal = (5000, 5000) at input_body_shape = (256, 192)
    # cfg.model.princpt = (96, 128) = (input_body_W/2, input_body_H/2)
    # Scale to input_img_shape = (512, 384)
    focal = cfg.model.focal          # (fx, fy)
    princpt = cfg.model.princpt      # (cx, cy)
    input_body = cfg.model.input_body_shape  # (H, W)
    input_img = cfg.model.input_img_shape    # (H, W)

    render_h, render_w = int(input_img[0]), int(input_img[1])
    scale_x = input_img[1] / input_body[1]  # 384/192 = 2.0
    scale_y = input_img[0] / input_body[0]  # 512/256 = 2.0

    fx_img = focal[0] * scale_x    # 5000 * 2 = 10000
    fy_img = focal[1] * scale_y    # 5000 * 2 = 10000
    cx_img = princpt[0] * scale_x  # 96 * 2 = 192
    cy_img = princpt[1] * scale_y  # 128 * 2 = 256

    # ── Build SMPL 24-joint 3D poses ─────────────────────────────────────
    smpl_poses = []
    inv_transforms = []
    for t in range(B):
        if timeline_3d is not None and t < len(timeline_3d) and timeline_3d[t] is not None:
            d3d = timeline_3d[t]
            smpl24 = _smplestx_3d_to_smpl24(d3d["joint_cam"], d3d["root_cam"])
            smpl_poses.append([smpl24])
            inv_transforms.append(d3d["inv_trans"])
        else:
            smpl_poses.append([np.zeros((24, 3), dtype=np.float32)])
            inv_transforms.append(None)

    # ── Compute dynamic cylinder radius ──────────────────────────────────
    # NLF reference: radius=21.5 at focal≈700, Z≈400, image≈720p
    #   → projected size ≈ 21.5*700/400 ≈ 37.6 px on 720px height ≈ 5.2%
    # For SMPLest-X: focal_img≈10000, Z≈2000-5000
    #   → target ~4% of render_h in projected pixels
    z_values = []
    for t in range(B):
        if timeline_3d is not None and t < len(timeline_3d) and timeline_3d[t] is not None:
            root_z = timeline_3d[t]["root_cam"][2]
            if root_z > 0:
                z_values.append(root_z)

    avg_z = np.mean(z_values) if z_values else 3000.0
    target_px = 0.015 * render_h  # ~8 pixels on 512px (warpAffine will scale up)
    radius = target_px * avg_z / fy_img

    # ── Build cylinder specs using SCAIL pipeline logic ──────────────────
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

    cylinder_specs_list = []
    for i in range(B):
        specs = get_single_pose_cylinder_specs(
            (i, smpl_poses[i], None, None, None, None, colors, limb_seq, draw_seq))
        cylinder_specs_list.append(specs)

    # ── Render 3D cylinders at input_img_shape resolution ────────────────
    if render_backend == "taichi" and render_whole_taichi is not None:
        try:
            frames_rgba = render_whole_taichi(
                cylinder_specs_list, H=render_h, W=render_w,
                fx=fx_img, fy=fy_img, cx=cx_img, cy=cy_img, radius=radius)
        except Exception:
            logging.warning("Taichi rendering failed, falling back to torch.")
            frames_rgba = render_whole_torch(
                cylinder_specs_list, H=render_h, W=render_w,
                fx=fx_img, fy=fy_img, cx=cx_img, cy=cy_img, radius=radius)
    else:
        frames_rgba = render_whole_torch(
            cylinder_specs_list, H=render_h, W=render_w,
            fx=fx_img, fy=fy_img, cx=cx_img, cy=cy_img, radius=radius)

    # ── Per-frame warpAffine to output resolution + 2D overlay ───────────
    # Build DWPose data for 2D overlay (hands/face/cheekbones)
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

    # Draw 2D overlay (hands/face) at output resolution
    canvas_2d = draw_pose_to_canvas_np(
        copy.deepcopy(dw_poses), pool=None, H=img_h, W=img_w, reshape_scale=0,
        show_feet_flag=False, show_body_flag=False, show_cheek_flag=True,
        dw_hand=True, show_face_flag=True, show_hand_flag=True)

    result = []
    for t in range(B):
        frame = frames_rgba[t]
        # Extract RGB from RGBA
        if frame.shape[2] == 4:
            rgb = frame[:, :, :3].copy()
            alpha = frame[:, :, 3]
        else:
            rgb = frame.copy()
            alpha = np.ones((render_h, render_w), dtype=np.uint8) * 255

        # warpAffine: input_img_shape → output image space
        if inv_transforms[t] is not None:
            inv_trans = inv_transforms[t]  # (2, 3) affine matrix
            warped_rgb = cv2.warpAffine(
                rgb, inv_trans, (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            warped_alpha = cv2.warpAffine(
                alpha, inv_trans, (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            # No inv_trans — resize to output (fallback, shouldn't normally happen)
            warped_rgb = cv2.resize(rgb, (img_w, img_h))
            warped_alpha = cv2.resize(alpha, (img_w, img_h))

        # Composite: 3D cylinders on black background
        out = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        mask_3d = warped_alpha > 10  # threshold to avoid aliasing artifacts
        out[mask_3d] = warped_rgb[mask_3d]

        # Overlay 2D hand/face/cheekbone drawing
        canvas_img = canvas_2d[t]
        mask_2d = canvas_img != 0
        out[mask_2d] = canvas_img[mask_2d]

        result.append(out)

    return result
