"""
Render SCAIL-style pose images from SMPLest-X 137-joint keypoints.

Uses the original SCAIL rendering pipeline (copied into scail/ subpackage):
  - Body: 3D cylinder ray marching (Taichi or PyTorch) via render_nlf_as_images
  - Hands + Face: 2D DWPose overlay via draw_pose

This module is responsible only for converting SMPLest-X data formats
into the NLF 3D + DWPose 2D formats expected by the SCAIL renderer.
"""
import numpy as np

from .scail.nlf_render import render_nlf_as_images, intrinsic_matrix_from_field_of_view


# ── SMPLest-X 25-body → SMPL 24-joint mapping ───────────────────────────────
# SMPLest-X body (0-24):
#   0=Pelvis 1=L.Hip 2=R.Hip 3=L.Knee 4=R.Knee 5=L.Ankle 6=R.Ankle
#   7=Neck 8=L.Shoulder 9=R.Shoulder 10=L.Elbow 11=R.Elbow
#   12=L.Wrist 13=R.Wrist 14-16=L.Foot 17-19=R.Foot
#   20=L.Ear 21=R.Ear 22=L.Eye 23=R.Eye 24=Nose
#
# SMPL 24 joints (used by NLF):
#   0=Pelvis 1=L.Hip 2=R.Hip 3=Spine1 4=L.Knee 5=R.Knee 6=Spine2
#   7=L.Ankle 8=R.Ankle 9=Spine3 10=L.Foot 11=R.Foot
#   12=Neck 13=L.Collar 14=R.Collar 15=Head 16=L.Shoulder 17=R.Shoulder
#   18=L.Elbow 19=R.Elbow 20=L.Wrist 21=R.Wrist 22=L.Hand 23=R.Hand
#
# We map the joints we have and leave others as zero (spine/collar/hand).
# The SCAIL renderer only uses: 1,2,4,5,7,8,12,15,16,17,18,19,20,21
# via process_data_to_COCO_format, so missing spine/collar/hand is fine.

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

    Args:
        joint_cam_3d: (137, 3) root-relative 3D joints
        root_cam: (3,) absolute root position

    Returns:
        (24, 3) numpy array in absolute camera space
    """
    joints_abs = joint_cam_3d[:25] + root_cam[None, :]
    smpl24 = np.zeros((24, 3), dtype=np.float32)
    for src, dst in _SMPLESTX_TO_SMPL24.items():
        smpl24[dst] = joints_abs[src]
    return smpl24


def _smplestx_2d_to_dwpose(keypoints_2d, img_h, img_w, threshold=0.1):
    """
    Convert SMPLest-X 137-joint 2D keypoints to DWPose dict format.

    The SCAIL pipeline overlays:
      - Cheek bones (nose-eye-ear) from body candidate
      - Hands (HSV rainbow) from hands array
      - Face (white dots, optimized subset) from faces array

    DWPose format uses normalized coordinates (x/W, y/H).

    Args:
        keypoints_2d: (137, 3) array with (x, y, confidence)
        img_h, img_w: image dimensions for normalization
        threshold: confidence threshold

    Returns:
        DWPose dict with bodies/faces/hands
    """
    kp = keypoints_2d

    # ── Body candidate (18 joints, normalized) ────────────────────────────
    # COCO 18: 0=Nose 1=Neck 2=R.Shoulder 3=R.Elbow 4=R.Wrist
    #          5=L.Shoulder 6=L.Elbow 7=L.Wrist
    #          8=R.Hip 9=R.Knee 10=R.Ankle
    #          11=L.Hip 12=L.Knee 13=L.Ankle
    #          14=R.Eye 15=L.Eye 16=R.Ear 17=L.Ear
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
            candidate[dst, 0] = kp[src, 0] / img_w  # normalized x
            candidate[dst, 1] = kp[src, 1] / img_h  # normalized y
            subset[0, dst] = dst  # index into candidate
        else:
            subset[0, dst] = -1

    # ── Hands (21 joints each, normalized) ────────────────────────────────
    # SMPLest-X: left hand = 25-44 (wrist=12), right hand = 45-64 (wrist=13)
    hands = []
    for hand_start, wrist_idx in [(45, 13), (25, 12)]:  # right, left (DWPose order)
        hand = np.zeros((21, 2), dtype=np.float32)
        # Joint 0 = wrist
        if kp[wrist_idx, 2] > threshold:
            hand[0, 0] = kp[wrist_idx, 0] / img_w
            hand[0, 1] = kp[wrist_idx, 1] / img_h
        # Joints 1-20 = finger joints
        for j in range(20):
            idx = hand_start + j
            if idx < kp.shape[0] and kp[idx, 2] > threshold:
                hand[j + 1, 0] = kp[idx, 0] / img_w
                hand[j + 1, 1] = kp[idx, 1] / img_h
        hands.append(hand.tolist())

    # ── Face (68 landmarks, normalized) ───────────────────────────────────
    # SMPLest-X face: indices 65-136 (72 points)
    # DWPose face: 68 points
    # We map the first 68 of our 72 face points
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

    This is the main entry point called from phalp_pose_node.py when
    scail_pose mode is active.

    Args:
        timeline: list of (137, 3) 2D keypoint arrays (or None per frame)
        timeline_3d: list of dicts {"joint_cam", "root_cam", "inv_trans"} (or None)
        img_h, img_w: output image dimensions
        cfg: SMPLest-X config (needed for intrinsic matrix computation)
        render_backend: "taichi" or "torch"

    Returns:
        list of (H, W, 3) uint8 images
    """
    B = len(timeline)

    # ── Build NLF 3D poses (for cylinder rendering) ───────────────────────
    # smpl_poses[t] = list of persons, each person = (24, 3) numpy
    smpl_poses = []
    for t in range(B):
        if timeline_3d is not None and t < len(timeline_3d) and timeline_3d[t] is not None:
            d3d = timeline_3d[t]
            smpl24 = _smplestx_3d_to_smpl24(d3d["joint_cam"], d3d["root_cam"])
            smpl_poses.append([smpl24])
        else:
            # No 3D data — empty person list (no body cylinders)
            smpl_poses.append([np.zeros((24, 3), dtype=np.float32)])

    # ── Build DWPose 2D poses (for hand/face overlay) ─────────────────────
    dw_poses = []
    for t in range(B):
        if timeline[t] is not None:
            dw_poses.append(
                _smplestx_2d_to_dwpose(timeline[t], img_h, img_w))
        else:
            # Empty pose
            dw_poses.append({
                "bodies": {
                    "candidate": [np.zeros((18, 2)).tolist()],
                    "subset": [[-1] * 18],
                },
                "faces": [np.zeros((68, 2)).tolist()],
                "hands": [np.zeros((21, 2)).tolist(), np.zeros((21, 2)).tolist()],
            })

    # ── Compute intrinsic matrix ──────────────────────────────────────────
    # Use FOV-based intrinsics (same as NLF default: 55 degrees)
    intrinsic_matrix = intrinsic_matrix_from_field_of_view(
        (img_h, img_w), fov_degrees=55)

    # ── Render via SCAIL pipeline ─────────────────────────────────────────
    frames_rgba = render_nlf_as_images(
        smpl_poses, dw_poses,
        height=img_h, width=img_w, video_length=B,
        intrinsic_matrix=intrinsic_matrix,
        draw_2d=True, draw_face=True, draw_hands=True,
        render_backend=render_backend,
    )

    # Convert RGBA → RGB (drop alpha, black background)
    result = []
    for frame in frames_rgba:
        if frame.shape[2] == 4:
            rgb = frame[:, :, :3].copy()
        else:
            rgb = frame
        result.append(rgb)

    return result
