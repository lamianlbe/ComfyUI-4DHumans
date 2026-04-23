"""
NLF inference helper for SapiensPromptHMRPoseNode.

Reference implementations:
- kijai/ComfyUI-WanVideoWrapper (MTV/nodes.py): the authoritative
  working pattern for calling nlf_l_multi_0.3.2.torchscript.
- isarandi/nlf (demo.ipynb): API for output field layout.

Key facts from the kijai reference (important, otherwise JIT fails):
- Input is float [0, 1] BCHW (not uint8).  Just do a .permute(0,3,1,2)
  on the ComfyUI IMAGE tensor.
- detect_smpl_batched must be wrapped with
  torch._C._jit_set_profiling_executor(True) / restore to avoid
  JIT-profiling-related crashes like:
     RuntimeError: vector::_M_range_check: __n (which is 18446744073709551615)
- Output fields for 0.3.2 are ragged:
     pred['boxes']: list of length batch_size.
         Each item is (Ni, 5) = [x1, y1, x2, y2, conf] for that frame.
     pred['joints3d_nonparam']: list of length batch_size.
         Each item is (Ni, 24, 3) = SMPL-24 non-parametric 3D joints.
     pred['joints2d_nonparam'] (if present): same shape but (Ni, 24, 2)
         in original image pixel coords.
"""

import logging
import time

import numpy as np
import torch

_logger = logging.getLogger(__name__)


# OpenPose 25 <- SMPL 24 mapping.  Matches PromptHMR's body_joints
# layout so downstream (Pose Editor, SAM3/NLF renderer, DWPose
# conversion) works without branching on backend.
#
# SMPL-24 ordering (J_regressor):
#   0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee,
#   6 Spine2, 7 L_Ankle, 8 R_Ankle, 9 Spine3, 10 L_Foot, 11 R_Foot,
#   12 Neck, 13 L_Collar, 14 R_Collar, 15 Head, 16 L_Shoulder,
#   17 R_Shoulder, 18 L_Elbow, 19 R_Elbow, 20 L_Wrist, 21 R_Wrist,
#   22 L_Hand, 23 R_Hand
#
# OpenPose-25 ordering:
#   0 Nose, 1 Neck, 2 R_Shoulder, 3 R_Elbow, 4 R_Wrist,
#   5 L_Shoulder, 6 L_Elbow, 7 L_Wrist, 8 MidHip, 9 R_Hip,
#   10 R_Knee, 11 R_Ankle, 12 L_Hip, 13 L_Knee, 14 L_Ankle,
#   15 R_Eye, 16 L_Eye, 17 R_Ear, 18 L_Ear,
#   19 L_BigToe, 20 L_SmallToe, 21 L_Heel,
#   22 R_BigToe, 23 R_SmallToe, 24 R_Heel
# NOTE: OP25[0] (Nose) is deliberately NOT mapped from SMPL[15] (Head).
# SMPL's Head joint is at the skull top/center, while OpenPose's Nose
# sits at the face center — projecting Head into the Nose slot makes
# downstream renderers place the head marker ~10-20 cm too high.
# Leaving OP25[0] at zeros means "not detected"; Sapiens 2D keypoints
# fill in proper facial landmarks via fuse_3d_body_with_sapiens.
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
    """Convert (..., 24, D) SMPL joints to (..., 25, D) OpenPose-25.

    Works for D=2 (pixel space) and D=3.  OP-25 slots without SMPL
    equivalents (eyes, ears, toes, heels) stay zero.
    """
    shape = list(j_smpl24.shape)
    shape[-2] = 25
    out = np.zeros(shape, dtype=np.float32)
    for op_idx, smpl_idx in _OP25_FROM_SMPL24.items():
        out[..., op_idx, :] = j_smpl24[..., smpl_idx, :]
    return out


def _bbox_iou_xywh_like(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    au = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bu = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = au + bu - inter
    return inter / union if union > 0 else 0.0


def _mask_bbox(mask_frame):
    ys, xs = np.where(mask_frame > 0.5)
    if len(xs) == 0:
        return None
    return (
        float(xs.min()), float(ys.min()),
        float(xs.max()), float(ys.max()),
    )


def _nlf_default_intrinsic(img_h, img_w, fov_deg=55.0):
    """NLF's rendering convention: 55° FOV pinhole."""
    larger = max(img_h, img_w)
    focal = larger / (2.0 * np.tan(fov_deg * np.pi / 360.0))
    return np.array([
        [focal, 0.0, img_w / 2.0],
        [0.0, focal, img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _project_3d_to_2d(joints3d, K):
    """Perspective-project (N, 3) joints with 3x3 intrinsic K.

    Returns (N, 2).  Points with z<=0 become (0, 0).
    """
    out = np.zeros((joints3d.shape[0], 2), dtype=np.float32)
    z = joints3d[:, 2]
    valid = z > 1e-3
    if np.any(valid):
        x = joints3d[valid, 0]
        y = joints3d[valid, 1]
        zv = z[valid]
        u = (K[0, 0] * x + K[0, 2] * zv) / zv
        v = (K[1, 1] * y + K[1, 2] * zv) / zv
        out[valid, 0] = u
        out[valid, 1] = v
    return out


def run_nlf_inference(
    model, images_np, masks_np, valid_frames, phmr_frame_inputs,
    persons, cam_int_per_frame, scale_per_frame, offset_per_frame,
    batch_size, dtype, img_h, img_w, pbar, B,
):
    """Run NLF batched inference and scatter results into `persons`.

    Returns elapsed inference time in seconds (CUDA-synchronised).
    """
    total_time = 0.0

    # Build input tensor once in float [0, 1] BCHW.  images_np is uint8
    # (B, H, W, 3) RGB; convert and keep on CPU until each batch slice.
    # NOTE: This costs ~B * H * W * 3 * 4 bytes in CPU RAM for the full
    # video.  For reasonable resolutions that's fine; if it becomes a
    # problem, move the conversion inside the per-chunk loop.
    images_f32_bchw = (
        torch.from_numpy(images_np).float().div_(255.0)
              .permute(0, 3, 1, 2).contiguous()
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device("cpu")

    K_default = _nlf_default_intrinsic(img_h, img_w)
    logged_keys = False

    # Force JIT profiling executor ON — mirrors kijai's wrapper; avoids
    # the infamous 'vector::_M_range_check' crash when this TorchScript
    # model is run after other ops have set the executor to False.
    jit_prev = torch._C._jit_set_profiling_executor(True)
    try:
        for chunk_start in range(0, len(valid_frames), batch_size):
            chunk_frames = valid_frames[chunk_start:chunk_start + batch_size]
            frame_batch = images_f32_bchw[chunk_frames].to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                pred = model.detect_smpl_batched(frame_batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            # Log output keys/shapes once so skeleton assumptions can
            # be verified.
            if not logged_keys:
                info_parts = []
                for k, v in pred.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0 and \
                            isinstance(v[0], torch.Tensor):
                        info_parts.append(
                            f"{k}=list[len={len(v)}, item0={tuple(v[0].shape)}]"
                        )
                    elif isinstance(v, torch.Tensor):
                        info_parts.append(f"{k}=Tensor{tuple(v.shape)}")
                    else:
                        info_parts.append(f"{k}={type(v).__name__}")
                _logger.info("NLF pred keys: %s", "; ".join(info_parts))
                logged_keys = True

            # Extract ragged per-frame outputs.  Defensive: the demo
            # notebook listed many fields but the 0.3.2 TorchScript we
            # tested emits only boxes and joints3d_nonparam.
            boxes_list = pred.get("boxes", None)
            j3d_list = pred.get("joints3d_nonparam", None)
            j2d_list = pred.get("joints2d_nonparam", None)

            if j3d_list is None:
                raise RuntimeError(
                    "NLF output has no 'joints3d_nonparam' key. "
                    f"Got keys: {list(pred.keys())}"
                )

            for chunk_i, t in enumerate(chunk_frames):
                masks_frame_np = masks_np[t]
                person_indices = phmr_frame_inputs[t]["person_indices"]

                # Per-frame ragged tensors.
                boxes_t = (
                    boxes_list[chunk_i].detach().float().cpu().numpy()
                    if boxes_list is not None and boxes_list[chunk_i] is not None
                    else None
                )
                j3d_t = j3d_list[chunk_i].detach().float().cpu().numpy()
                j2d_t = (
                    j2d_list[chunk_i].detach().float().cpu().numpy()
                    if j2d_list is not None and j2d_list[chunk_i] is not None
                    else None
                )

                if j3d_t.shape[0] == 0:
                    # No detections for this frame — leave person entries
                    # as None; interpolation Phase 1c will try to fill.
                    pass
                else:
                    # Build bboxes from NLF output: prefer `boxes`, else
                    # compute from j2d or j3d.
                    if boxes_t is not None and boxes_t.shape[0] == j3d_t.shape[0]:
                        det_bboxes = [
                            (float(bb[0]), float(bb[1]),
                             float(bb[2]), float(bb[3]))
                            for bb in boxes_t
                        ]
                    elif j2d_t is not None:
                        det_bboxes = [
                            (float(p[:, 0].min()), float(p[:, 1].min()),
                             float(p[:, 0].max()), float(p[:, 1].max()))
                            for p in j2d_t
                        ]
                    else:
                        # No 2D info: project 3D via default intrinsic.
                        det_bboxes = []
                        for p3d in j3d_t:
                            p2d = _project_3d_to_2d(p3d, K_default)
                            det_bboxes.append((
                                float(p2d[:, 0].min()),
                                float(p2d[:, 1].min()),
                                float(p2d[:, 0].max()),
                                float(p2d[:, 1].max()),
                            ))

                    used = set()
                    for p_idx in person_indices:
                        mbbox = _mask_bbox(masks_frame_np[p_idx])
                        if mbbox is None:
                            continue
                        best_i = -1
                        best_iou = 0.1
                        for i in range(len(det_bboxes)):
                            if i in used:
                                continue
                            iou = _bbox_iou_xywh_like(mbbox, det_bboxes[i])
                            if iou > best_iou:
                                best_iou = iou
                                best_i = i
                        if best_i < 0:
                            continue

                        used.add(best_i)
                        j3d = j3d_t[best_i]  # (K, 3) — expect K=24

                        # 2D: use provided joints2d_nonparam if present,
                        # else project via default intrinsic.
                        if j2d_t is not None:
                            j2d = j2d_t[best_i]
                        else:
                            j2d = _project_3d_to_2d(j3d, K_default)

                        if j3d.shape[0] != 24:
                            if not hasattr(run_nlf_inference, "_warned_k"):
                                _logger.warning(
                                    "NLF returned K=%d joints (expected 24). "
                                    "Skeleton mapping may be incorrect.",
                                    j3d.shape[0],
                                )
                                run_nlf_inference._warned_k = True
                            # Fall back: try to still fill downstream.
                            body_j3d = np.zeros((25, 3), dtype=np.float32)
                            body_j2d = np.zeros((25, 2), dtype=np.float32)
                            smpl_j3d = np.zeros((24, 3), dtype=np.float32)
                            limit = min(24, j3d.shape[0])
                            smpl_j3d[:limit] = j3d[:limit]
                        else:
                            body_j3d = _smpl24_to_openpose25(j3d)
                            body_j2d = _smpl24_to_openpose25(j2d)
                            smpl_j3d = j3d.astype(np.float32)

                        persons[p_idx]["body_joints2d"][t] = \
                            body_j2d.astype(np.float32)
                        persons[p_idx]["body_joints"][t] = \
                            body_j3d.astype(np.float32)
                        persons[p_idx]["smpl_j3d"][t] = smpl_j3d

                # Camera: constant 55° FOV for every frame; scale=1, offset=0.
                cam_int_per_frame[t] = K_default.copy()
                scale_per_frame[t] = 1.0
                offset_per_frame[t] = np.zeros(2, dtype=np.float64)

                pbar.update(1)
    finally:
        torch._C._jit_set_profiling_executor(jit_prev)

    # Pbar catch-up for frames that had no valid persons at all.
    skipped = B - len(valid_frames)
    for _ in range(skipped):
        pbar.update(1)

    return total_time
