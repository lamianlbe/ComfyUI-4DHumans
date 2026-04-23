"""
NLF inference helper for SapiensPromptHMRPoseNode.

Keeps the NLF-specific branch out of the main node file.  Goal:
produce the same per-person fields (body_joints2d, body_joints,
smpl_j3d) and per-frame camera params that the PromptHMR path
produces, so downstream code (Pose Editor, NLF renderer, Sapiens
fusion) is backend-agnostic.

NLF's detect_smpl_batched returns parametric and non-parametric
joints.  The multi model uses an SMPL-style skeleton internally;
exact joint ordering is logged on first use so skeleton mappings
can be tuned based on observed shapes.
"""

import logging
import time

import numpy as np
import torch

from ..humans4d.hmr2.utils.sapiens_inference import (  # noqa: F401
    run_sapiens_batched,
)

_logger = logging.getLogger(__name__)


# NLF's multi model outputs its joints in SMPL's canonical 24-joint
# ordering (same J_regressor indices PromptHMR uses for smpl_j3d).
# Mapping from SMPL-24 indices to the 25-joint OpenPose skeleton that
# PromptHMR's body_joints / body_joints2d follow.  Entries without a
# direct SMPL equivalent are synthesised from neighbouring joints or
# left at (0, 0).
#
# OpenPose 25 joints:
#   0 Nose, 1 Neck, 2 R_Shoulder, 3 R_Elbow, 4 R_Wrist,
#   5 L_Shoulder, 6 L_Elbow, 7 L_Wrist, 8 MidHip, 9 R_Hip,
#   10 R_Knee, 11 R_Ankle, 12 L_Hip, 13 L_Knee, 14 L_Ankle,
#   15 R_Eye, 16 L_Eye, 17 R_Ear, 18 L_Ear,
#   19 L_BigToe, 20 L_SmallToe, 21 L_Heel,
#   22 R_BigToe, 23 R_SmallToe, 24 R_Heel
#
# SMPL 24 joints (J_regressor output):
#   0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee,
#   6 Spine2, 7 L_Ankle, 8 R_Ankle, 9 Spine3, 10 L_Foot, 11 R_Foot,
#   12 Neck, 13 L_Collar, 14 R_Collar, 15 Head, 16 L_Shoulder,
#   17 R_Shoulder, 18 L_Elbow, 19 R_Elbow, 20 L_Wrist, 21 R_Wrist,
#   22 L_Hand, 23 R_Hand

# Direct OpenPose25 <- SMPL24 mappings (when a clean single-joint match exists).
# Keys = OpenPose 25 index, values = SMPL 24 index.
_OP25_FROM_SMPL24 = {
    1:  12,  # Neck
    2:  17,  # R_Shoulder
    3:  19,  # R_Elbow
    4:  21,  # R_Wrist
    5:  16,  # L_Shoulder
    6:  18,  # L_Elbow
    7:  20,  # L_Wrist
    8:  0,   # MidHip ← Pelvis
    9:  2,   # R_Hip
    10: 5,   # R_Knee
    11: 8,   # R_Ankle
    12: 1,   # L_Hip
    13: 4,   # L_Knee
    14: 7,   # L_Ankle
}


def _smpl24_to_openpose25(j_smpl24):
    """Convert a (..., 24, D) SMPL joint tensor to (..., 25, D) OpenPose 25.

    Works for both 2D (D=2) and 3D (D=3) joints.  The facial/foot joints
    that aren't in SMPL-24 (Nose, eyes, ears, toes, heels) are left at
    zero; downstream renderers treat (0, 0) as invalid, and Sapiens
    provides much better facial landmarks anyway (for 2D).
    """
    shape = list(j_smpl24.shape)
    shape[-2] = 25
    out = np.zeros(shape, dtype=np.float32)
    for op_idx, smpl_idx in _OP25_FROM_SMPL24.items():
        out[..., op_idx, :] = j_smpl24[..., smpl_idx, :]
    # Synthesise Head (SMPL 15) as a rough proxy for Nose (op 0) so that
    # downstream person-ID labels and skeleton rendering still have a
    # head anchor.
    out[..., 0, :] = j_smpl24[..., 15, :]
    return out


def _bbox_iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aw = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aw + bw - inter
    return inter / union if union > 0 else 0.0


def _joints2d_to_bbox(j2d):
    """Tight (x1, y1, x2, y2) around valid 2D joints."""
    # j2d: (K, 2); assume all rows are valid (NLF outputs pixel coords).
    xs = j2d[:, 0]
    ys = j2d[:, 1]
    return (
        float(xs.min()), float(ys.min()),
        float(xs.max()), float(ys.max()),
    )


def _mask_bbox(mask_frame):
    """Tight (x1, y1, x2, y2) around a binary mask."""
    ys, xs = np.where(mask_frame > 0.5)
    if len(xs) == 0:
        return None
    return (
        float(xs.min()), float(ys.min()),
        float(xs.max()), float(ys.max()),
    )


def _nlf_default_intrinsic(img_h, img_w, fov_deg=55.0):
    """NLF's rendering convention: 55 deg FOV pinhole."""
    larger = max(img_h, img_w)
    focal = larger / (2.0 * np.tan(fov_deg * np.pi / 360.0))
    return np.array([
        [focal, 0.0, img_w / 2.0],
        [0.0, focal, img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def run_nlf_inference(
    model, images_np, masks_np, valid_frames, phmr_frame_inputs,
    persons, cam_int_per_frame, scale_per_frame, offset_per_frame,
    batch_size, dtype, img_h, img_w, pbar, B,
):
    """Run NLF batched inference and scatter results into `persons`.

    Returns the elapsed inference time in seconds (GPU-synchronised).
    """
    total_time = 0.0
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cuda")

    # NLF uses a static camera per video.  We synthesise a 55° FOV
    # pinhole intrinsic and use scale=1, offset=0 (joints2d are already
    # in original image pixel coords).
    K_default = _nlf_default_intrinsic(img_h, img_w)

    autocast_enabled = (dtype != torch.float32 and device.type == "cuda")

    logged_shapes = False

    for chunk_start in range(0, len(valid_frames), batch_size):
        chunk_frames = valid_frames[chunk_start:chunk_start + batch_size]
        # Stack frames into (N, 3, H, W) uint8 tensor on device.
        frame_batch = torch.from_numpy(
            np.stack([images_np[t] for t in chunk_frames], axis=0)
        ).permute(0, 3, 1, 2).contiguous().to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            if autocast_enabled:
                ctx = torch.autocast(device.type, dtype=dtype)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                # detect_smpl_batched returns a dict with per-frame
                # ragged tensors (padded with NaN for short frames).
                pred = model.detect_smpl_batched(frame_batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        # Log output shapes once so we can verify skeleton conventions.
        if not logged_shapes:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    _logger.info(
                        "NLF output %s: shape=%s dtype=%s",
                        k, tuple(v.shape), v.dtype,
                    )
            logged_shapes = True

        # Move per-frame predictions to CPU for numpy processing.
        joints3d_all = pred["joints3d"].detach().float().cpu().numpy()
        joints2d_all = pred["joints2d"].detach().float().cpu().numpy()
        # Expected shapes (best-effort; NLF may pad short frames):
        #   joints3d_all: (chunk_N, max_persons_in_chunk, K, 3)
        #   joints2d_all: (chunk_N, max_persons_in_chunk, K, 2)
        # If chunk contains any "no detection" entries they may be
        # represented as NaN rows — we filter those below.

        for chunk_i, t in enumerate(chunk_frames):
            frame_inputs = phmr_frame_inputs[t]
            person_indices = frame_inputs["person_indices"]
            masks_frame_np = masks_np[t]

            # Per-frame NLF predictions (after possible padding).
            if joints3d_all.ndim == 4:
                frame_j3d = joints3d_all[chunk_i]  # (P, K, 3)
                frame_j2d = joints2d_all[chunk_i]  # (P, K, 2)
            else:
                # Some NLF versions may return a flat (chunk_N * P, ...) layout
                frame_j3d = joints3d_all
                frame_j2d = joints2d_all

            # Filter out padding rows (NaN anywhere).
            valid = ~np.isnan(frame_j3d).any(axis=(-1, -2))
            frame_j3d = frame_j3d[valid]
            frame_j2d = frame_j2d[valid]

            # Match each requested person_index to an NLF detection
            # by bbox IoU (mask bbox vs. NLF joints2d bbox).
            used = set()
            for p_idx in person_indices:
                mbbox = _mask_bbox(masks_frame_np[p_idx])
                if mbbox is None or len(frame_j3d) == 0:
                    continue
                best_i = -1
                best_iou = 0.1  # minimum acceptable IoU
                for i in range(len(frame_j3d)):
                    if i in used:
                        continue
                    nbbox = _joints2d_to_bbox(frame_j2d[i])
                    iou = _bbox_iou(mbbox, nbbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                if best_i < 0:
                    continue

                used.add(best_i)
                # Assume NLF returns SMPL-24 joints; map to OpenPose 25.
                j3d_smpl = frame_j3d[best_i]  # (K, 3)
                j2d_smpl = frame_j2d[best_i]  # (K, 2)

                if j3d_smpl.shape[0] == 24:
                    body_j3d = _smpl24_to_openpose25(j3d_smpl)
                    body_j2d = _smpl24_to_openpose25(j2d_smpl)
                    smpl_j3d = j3d_smpl.astype(np.float32)
                else:
                    # Unexpected joint count: store raw and let user report.
                    body_j3d = j3d_smpl.astype(np.float32)
                    body_j2d = j2d_smpl.astype(np.float32)
                    smpl_j3d = (
                        j3d_smpl[:24].astype(np.float32)
                        if j3d_smpl.shape[0] >= 24
                        else np.zeros((24, 3), dtype=np.float32)
                    )
                    if not hasattr(run_nlf_inference, "_warned_skeleton"):
                        _logger.warning(
                            "NLF returned K=%d joints, expected 24 (SMPL). "
                            "Skeleton mapping may be incorrect — please "
                            "report the joint dimension so we can fix it.",
                            j3d_smpl.shape[0],
                        )
                        run_nlf_inference._warned_skeleton = True

                persons[p_idx]["body_joints2d"][t] = body_j2d.astype(np.float32)
                persons[p_idx]["body_joints"][t] = body_j3d.astype(np.float32)
                persons[p_idx]["smpl_j3d"][t] = smpl_j3d

            # Camera params: same for every frame (no padded-image transform)
            cam_int_per_frame[t] = K_default.copy()
            scale_per_frame[t] = 1.0
            offset_per_frame[t] = np.zeros(2, dtype=np.float64)

            pbar.update(1)

    # Pbar catch-up for frames that had no valid persons at all.
    skipped = B - len(valid_frames)
    for _ in range(skipped):
        pbar.update(1)

    return total_time
