"""
Fast SAM 3D Body inference helper.

Given a video and per-frame mask bboxes (already aligned to persons),
run Fast SAM 3D Body to produce MHR-70 keypoints + MHR-18439 mesh for
every valid (frame, person), then fold those outputs into POSES-compatible
fields:

  body_joints2d  (25, 2) px   OpenPose-25 layout
  body_joints    (25, 3) m    OpenPose-25 layout, camera space
  smpl_j3d       (24, 3) m    SMPL-24 via MHR2SMPL mapper (+ optional smoother)
  coco_wb_body_feet (23, 3) px  COCO-WholeBody slots 0..22 from MHR
  coco_wb_hands     (42, 3) px  COCO-WholeBody slots 91..132 from MHR

Camera params per frame:
  cam_int (3, 3)   default FOV intrinsic (Fast SAM 3D Body convention)
  scale=1.0, offset=(0, 0)   (no padding applied)
"""

import contextlib
import io
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch

from ..fastsam3dbody_lib import ensure_lib_importable

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect both Python-level stdout and C-level fd-1 writes.

    Fast SAM 3D Body's internal timing uses bare `print(...)` calls
    *and* direct C-level writes (e.g. from TorchScript / CUDA kernels),
    so only redirecting `sys.stdout` isn't enough on its own.  We
    double-wrap: a Python-level redirect handles `print`, and a dup2
    on fd-1 handles anything that bypasses the io module.
    """
    old_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(old_stdout_fd, 1)
        os.close(devnull_fd)
        os.close(old_stdout_fd)


# =============================================================================
# Skeleton mapping tables
# =============================================================================

# MHR-70 layout (key indices we care about — see
# fastsam3dbody_lib/sam_3d_body/metadata/mhr70.py for the full list):
#   0-4   : nose, L_eye, R_eye, L_ear, R_ear
#   5-14  : body (L_shoulder, R_shoulder, L_elbow, R_elbow, L_hip, R_hip,
#           L_knee, R_knee, L_ankle, R_ankle)
#   15-20 : feet (L_big_toe, L_small_toe, L_heel, R_big_toe, R_small_toe, R_heel)
#   21-40 : right hand 20 finger joints
#   41    : right_wrist
#   42-61 : left hand 20 finger joints
#   62    : left_wrist
#   63-66 : olecranon × 2, cubital_fossa × 2     (unused here)
#   67-68 : L_acromion, R_acromion               (unused here)
#   69    : neck

# ---------------------------------------------------------------------------
# OpenPose-25 <- MHR-70
# MidHip (OP25[8]) synthesised from L_hip + R_hip midpoint.
# ---------------------------------------------------------------------------
_OP25_FROM_MHR70 = {
    0:  0,   # Nose
    1:  69,  # Neck
    2:  6,   # R_Shoulder
    3:  8,   # R_Elbow
    4:  41,  # R_Wrist
    5:  5,   # L_Shoulder
    6:  7,   # L_Elbow
    7:  62,  # L_Wrist
    # 8: MidHip  <- synthesized from MHR 9 (L_hip) + 10 (R_hip)
    9:  10,  # R_Hip
    10: 12,  # R_Knee
    11: 14,  # R_Ankle
    12: 9,   # L_Hip
    13: 11,  # L_Knee
    14: 13,  # L_Ankle
    15: 2,   # R_Eye
    16: 1,   # L_Eye
    17: 4,   # R_Ear
    18: 3,   # L_Ear
    19: 15,  # L_BigToe
    20: 16,  # L_SmallToe
    21: 17,  # L_Heel
    22: 18,  # R_BigToe
    23: 19,  # R_SmallToe
    24: 20,  # R_Heel
}


# ---------------------------------------------------------------------------
# COCO-WholeBody body+feet slots (0..22) <- MHR-70
# Layout:  0-16 body (COCO keypoints)    17-22 feet
# ---------------------------------------------------------------------------
_COCO_WB_BODY_FEET_FROM_MHR70 = [
    # 0-4 head: nose, L_eye, R_eye, L_ear, R_ear
    0, 1, 2, 3, 4,
    # 5-10 shoulders / elbows / wrists
    5, 6, 7, 8, 62, 41,       # L_sh, R_sh, L_el, R_el, L_wr, R_wr
    # 11-16 hips / knees / ankles
    9, 10, 11, 12, 13, 14,
    # 17-22 feet: L_BigToe, L_SmallToe, L_Heel, R_BigToe, R_SmallToe, R_Heel
    15, 16, 17, 18, 19, 20,
]


# ---------------------------------------------------------------------------
# COCO-WholeBody hand slots (91..132) <- MHR-70
#
# COCO-WB hand convention (21 pts per hand):
#   0       : wrist
#   1-4     : thumb  (CMC -> MCP -> IP  -> TIP)
#   5-8     : index  (MCP -> PIP -> DIP -> TIP)
#   9-12    : middle (MCP -> PIP -> DIP -> TIP)
#   13-16   : ring   (MCP -> PIP -> DIP -> TIP)
#   17-20   : pinky  (MCP -> PIP -> DIP -> TIP)
#
# MHR convention per finger:  [tip, 1st-joint, 2nd-joint, 3rd-joint]
# where 1st/2nd/3rd are counted *from the tip back to the palm*, so:
#   tip  = TIP       (the actual fingertip)
#   1st  = DIP/IP    (one joint from the tip)
#   2nd  = PIP/MCP-of-thumb
#   3rd  = MCP/CMC-of-thumb   (closest to the wrist)
#
# Therefore we need to REVERSE each 4-joint finger group to line up with
# COCO's root→tip order.
# ---------------------------------------------------------------------------
def _hand_mhr70_slice(wrist_idx: int, first_finger_idx: int) -> list[int]:
    """Build a 21-entry MHR-70 index list for one hand in COCO order."""
    out = [wrist_idx]  # COCO[0] = wrist
    for finger in range(5):  # thumb, index, middle, ring, pinky
        # MHR stores this finger as 4 consecutive entries:
        #   [tip, 1st, 2nd, 3rd]
        # starting at (first_finger_idx + finger * 4).
        base = first_finger_idx + finger * 4
        tip, j1, j2, j3 = base + 0, base + 1, base + 2, base + 3
        # COCO finger order: CMC/MCP → PIP/IP → DIP → TIP (root → tip)
        out.extend([j3, j2, j1, tip])
    return out


# Right hand: MHR index 21..40 (fingers), 41 (wrist)
_COCO_WB_RHAND_FROM_MHR70 = _hand_mhr70_slice(wrist_idx=41, first_finger_idx=21)
# Left hand: MHR index 42..61 (fingers), 62 (wrist)
_COCO_WB_LHAND_FROM_MHR70 = _hand_mhr70_slice(wrist_idx=62, first_finger_idx=42)

assert len(_COCO_WB_RHAND_FROM_MHR70) == 21
assert len(_COCO_WB_LHAND_FROM_MHR70) == 21


# =============================================================================
# Keypoint transformation helpers
# =============================================================================

def mhr70_to_openpose25(kp_mhr70: np.ndarray) -> np.ndarray:
    """Convert (70, D) MHR-70 keypoints → (25, D) OpenPose-25.

    MidHip (OP25[8]) is synthesised as the midpoint of MHR L_hip/R_hip.
    Works for D=2 (pixel) or D=3 (metric).
    """
    D = kp_mhr70.shape[-1]
    out = np.zeros((25, D), dtype=np.float32)
    for op_idx, mhr_idx in _OP25_FROM_MHR70.items():
        out[op_idx] = kp_mhr70[mhr_idx]
    # MidHip
    out[8] = 0.5 * (kp_mhr70[9] + kp_mhr70[10])
    return out.astype(np.float32)


def mhr70_to_coco_wb_body_feet_hands(
    kp2d_mhr70: np.ndarray,  # (70, 2) or (70, 3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three COCO-WholeBody slices derived from MHR-70 2D keypoints.

    Returns
    -------
    body_feet_23 : (23, D)  for COCO-WB indices 0..22
    rhand_21     : (21, D)  for COCO-WB indices 112..132
    lhand_21     : (21, D)  for COCO-WB indices 91..111
    """
    body_feet = kp2d_mhr70[_COCO_WB_BODY_FEET_FROM_MHR70]
    rhand = kp2d_mhr70[_COCO_WB_RHAND_FROM_MHR70]
    lhand = kp2d_mhr70[_COCO_WB_LHAND_FROM_MHR70]
    return body_feet.astype(np.float32), rhand.astype(np.float32), lhand.astype(np.float32)


def _cam_int_default(img_h: int, img_w: int) -> np.ndarray:
    """Fast SAM 3D Body's default camera intrinsic used when no FOV
    estimator is attached (mirrors prepare_batch.py)."""
    focal = (img_h * img_h + img_w * img_w) ** 0.5
    return np.array([
        [focal, 0.0, img_w / 2.0],
        [0.0, focal, img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# =============================================================================
# YOLO11m-Pose → Fast SAM 3D Body alignment
# =============================================================================

def _bbox_iou(a, b) -> float:
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


def _align_yolo_pose_to_mask_bboxes(
    yolo_result, mask_bboxes: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Greedy IoU-match YOLO11m-Pose detections to mask_bboxes.

    Parameters
    ----------
    yolo_result : ultralytics Result
        A single-frame result with .boxes.xyxy and .keypoints.data
        (the latter is shape (M, 17, 3) or (M, 17, 2)).
    mask_bboxes : (N, 4)  xyxy

    Returns
    -------
    kpts_aligned : (N, 17, 3) or None
        Per-person keypoints aligned to mask_bboxes.  None for persons
        with no IoU match.  When all matches fail, returns (None, None).
    bboxes_aligned : (N, 4) or None
        YOLO's own bboxes for the matched persons.  Same alignment rules.
    """
    if yolo_result is None or yolo_result.boxes is None or yolo_result.boxes.xyxy is None:
        return None, None
    if yolo_result.keypoints is None or yolo_result.keypoints.data is None:
        return None, None

    y_boxes = yolo_result.boxes.xyxy.cpu().numpy()
    y_kp = yolo_result.keypoints.data.cpu().numpy()  # (M, 17, 2 or 3)
    if y_kp.shape[-1] == 2:  # no conf -> pad with 1.0
        conf = np.ones(y_kp.shape[:-1] + (1,), dtype=y_kp.dtype)
        y_kp = np.concatenate([y_kp, conf], axis=-1)

    N = len(mask_bboxes)
    M = len(y_boxes)
    if M == 0:
        return None, None

    # Greedy match by IoU
    kpts_aligned = [None] * N
    bboxes_aligned = [None] * N
    used = set()
    iou_pairs = []
    for i in range(N):
        for j in range(M):
            if j in used:
                continue
            iou_pairs.append((i, j, _bbox_iou(mask_bboxes[i], y_boxes[j])))
    iou_pairs.sort(key=lambda t: t[2], reverse=True)

    for i, j, iou in iou_pairs:
        if iou < 0.2:
            break
        if kpts_aligned[i] is None and j not in used:
            kpts_aligned[i] = y_kp[j]
            bboxes_aligned[i] = y_boxes[j]
            used.add(j)

    # If nothing matched at all return (None, None) so caller falls back to body_decoder
    if all(k is None for k in kpts_aligned):
        return None, None

    # Pack into (N, 17, 3) and (N, 4) with zeros for missing rows
    kpts_out = np.zeros((N, 17, 3), dtype=np.float32)
    bboxes_out = np.zeros((N, 4), dtype=np.float32)
    has_kp = np.zeros(N, dtype=bool)
    for i in range(N):
        if kpts_aligned[i] is not None:
            kpts_out[i] = kpts_aligned[i]
            bboxes_out[i] = bboxes_aligned[i]
            has_kp[i] = True
    return (kpts_out, bboxes_out) if has_kp.any() else (None, None)


# =============================================================================
# Main entry point: run Fast SAM 3D Body on a video
# =============================================================================

def run_fastsam3db_video(
    images_np_u8: np.ndarray,          # (B, H, W, 3) uint8 RGB
    mask_bboxes_per_frame: list,       # [ [(x1,y1,x2,y2), ...] for each frame ]
    masks_np: np.ndarray | None,       # (B, n_persons, H, W) bool, or None
    person_indices_per_frame: list,    # [ [p_idx, ...] for each frame ]  maps bbox slot → POSES person id
    fastsam3db_dict: dict,             # from LoadFastSAM3DBody
    yolo11pose_dict: dict | None,      # from LoadYOLO11Pose, optional
    n_persons: int,
    img_h: int,
    img_w: int,
    batch_size_yolo: int = 128,
    pbar=None,
):
    """Run Fast SAM 3D Body per-frame, plus MHR2SMPL, and return per-person
    timelines usable by the POSES dict.

    Returns
    -------
    result : dict with keys
        persons : list of length n_persons, each
                  {"body_joints2d": [None|(25,2)] * B,
                   "body_joints":   [None|(25,3)] * B,
                   "smpl_j3d":      [None|(24,3)] * B,
                   "mhr_kp2d":      [None|(70,2)] * B,  (used by RTMFace bbox derivation)
                   "mhr_vertices":  [None|(18439,3)] * B  (kept in memory; not saved)
                   "mhr_cam_t":     [None|(3,)] * B,
                   }
        cam_int  : [None|(3,3)] * B
        fastsam3db_time_s : float
        mhr2smpl_time_s   : float
    """
    ensure_lib_importable()

    estimator = fastsam3db_dict["estimator"]
    mhr2smpl  = fastsam3db_dict["mhr2smpl"]
    smpl_pkl  = fastsam3db_dict["smpl_model_path"]
    device    = fastsam3db_dict["device"]

    B = images_np_u8.shape[0]

    # Per-person timelines
    persons = []
    for _ in range(n_persons):
        persons.append({
            "body_joints2d": [None] * B,
            "body_joints":   [None] * B,
            "smpl_j3d":      [None] * B,
            "mhr_kp2d":      [None] * B,
            "mhr_vertices":  [None] * B,
            "mhr_cam_t":     [None] * B,
        })
    cam_int_per_frame = [None] * B

    # -------------------------------------------------------------------
    # (Optional) YOLO11m-Pose pass — run in chunks on the whole video
    # -------------------------------------------------------------------
    yolo_results_per_frame = [None] * B
    if yolo11pose_dict is not None:
        yolo_model = yolo11pose_dict["model"]
        # Move to GPU if not already (LoadYOLO could have offloaded previously)
        if torch.cuda.is_available():
            try:
                if hasattr(yolo_model, "model") and yolo_model.model is not None:
                    yolo_model.model.to("cuda")
            except Exception as e:
                _logger.warning("YOLO11 restore-to-CUDA failed: %s", e)

        # YOLO runs on uint8 HWC RGB directly (cv2 convention == ours).
        # ultralytics accepts a list of np arrays.
        frame_list_bgr = [images_np_u8[t, :, :, ::-1] for t in range(B)]  # RGB→BGR
        t0 = time.perf_counter()
        for chunk_start in range(0, B, batch_size_yolo):
            chunk = frame_list_bgr[chunk_start:chunk_start + batch_size_yolo]
            results = list(yolo_model.predict(
                source=chunk, verbose=False, classes=[0], conf=0.25, stream=False,
            ))
            for i, r in enumerate(results):
                yolo_results_per_frame[chunk_start + i] = r
        _logger.info(
            "YOLO11m-Pose on %d frames: %.2fs", B, time.perf_counter() - t0
        )

        # Offload YOLO back to CPU to free VRAM for Fast SAM 3D Body
        try:
            yolo_model.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Fast SAM 3D Body per-frame inference
    # -------------------------------------------------------------------
    fastsam3db_time_s = 0.0
    logged = False

    for t in range(B):
        mbboxes = mask_bboxes_per_frame[t]
        pidxs = person_indices_per_frame[t]
        if len(mbboxes) == 0:
            if pbar is not None:
                pbar.update(1)
            continue

        mbboxes_np = np.asarray(mbboxes, dtype=np.float32)  # (N, 4) xyxy
        # Masks: SAM 3D Body expects (N, H, W, 1) uint8
        masks_arr = None
        if masks_np is not None:
            per_frame = np.zeros((len(mbboxes), img_h, img_w), dtype=np.uint8)
            for k, p_idx in enumerate(pidxs):
                per_frame[k] = (masks_np[t, p_idx] > 0.5).astype(np.uint8) * 255
            masks_arr = per_frame

        # YOLO11-Pose alignment (stored on the frame but NOT yet injected
        # into Fast SAM 3D Body — its `process_one_image` reads
        # yolo_pose_keypoints only when the detector is of type
        # "yolo_pose", which requires a detector we deliberately don't
        # wire up.  Step 3 runs hand_box_source="body_decoder".  Step 4
        # will monkey-patch run_inference so external keypoints get
        # forwarded.  The YOLO pass still runs for validation/timing.
        yolo_kp, yolo_bx = None, None
        if yolo_results_per_frame[t] is not None:
            yolo_kp, yolo_bx = _align_yolo_pose_to_mask_bboxes(
                yolo_results_per_frame[t], mbboxes_np
            )

        # RGB frame as ndarray — estimator expects RGB when given ndarray
        img_rgb = images_np_u8[t]  # (H, W, 3) uint8 RGB

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with _suppress_stdout():
            outputs = estimator.process_one_image(
                img=img_rgb,
                bboxes=mbboxes_np,
                masks=masks_arr,
                cam_int=None,   # use default focal = sqrt(H^2 + W^2)
                bbox_thr=0.0,
                nms_thr=0.3,
                use_mask=masks_arr is not None,
                inference_type="full",
                hand_box_source="body_decoder",
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fastsam3db_time_s += time.perf_counter() - t0

        if outputs is None or len(outputs) == 0:
            if pbar is not None:
                pbar.update(1)
            continue

        if not logged:
            keys = list(outputs[0].keys()) if isinstance(outputs[0], dict) else []
            _logger.info(
                "Fast SAM 3D Body output keys (first person, first frame): %s",
                keys,
            )
            logged = True

        # Camera intrinsic default (Fast SAM 3D Body default: focal = diag)
        if cam_int_per_frame[t] is None:
            cam_int_per_frame[t] = _cam_int_default(img_h, img_w)

        # Scatter per-person outputs back to POSES layout
        for k, out in enumerate(outputs):
            if k >= len(pidxs):
                break  # estimator dropped a detection
            p_idx = pidxs[k]
            kp2d = np.asarray(out["pred_keypoints_2d"], dtype=np.float32)
            kp3d = np.asarray(out["pred_keypoints_3d"], dtype=np.float32)
            verts = np.asarray(out["pred_vertices"], dtype=np.float32)
            cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32).reshape(3)

            # MHR uses 70+ keypoints (304/308 total when face head is enabled).
            # We only need the first 70 for body+feet+hands.
            kp2d = kp2d[:70] if len(kp2d) > 70 else kp2d
            kp3d = kp3d[:70] if len(kp3d) > 70 else kp3d

            # IMPORTANT: pred_keypoints_3d is in CANONICAL frame (pelvis at
            # origin). To align with PromptHMR / NLF output format that
            # downstream nodes expect (NLF renderer, SCAIL transforms, the
            # Pose Editor 3D-distance code, etc.), we need to translate it
            # into CAMERA space via:
            #     j3d_cam = j3d_canonical + pred_cam_t
            # This mirrors what camera_head.py:perspective_projection does
            # internally before projecting to 2D.
            kp3d_cam = kp3d + cam_t[None, :]

            persons[p_idx]["body_joints2d"][t] = mhr70_to_openpose25(kp2d)
            persons[p_idx]["body_joints"][t]   = mhr70_to_openpose25(kp3d_cam)
            persons[p_idx]["mhr_kp2d"][t]      = kp2d
            persons[p_idx]["mhr_vertices"][t]  = verts
            persons[p_idx]["mhr_cam_t"][t]     = cam_t

            # Use the per-person focal_length if available for more accurate cam
            fl = out.get("focal_length", None)
            if fl is not None and cam_int_per_frame[t] is None:
                fl_val = float(fl)
                cam_int_per_frame[t] = np.array([
                    [fl_val, 0.0, img_w / 2.0],
                    [0.0, fl_val, img_h / 2.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

        if pbar is not None:
            pbar.update(1)

    # -------------------------------------------------------------------
    # MHR → SMPL, per person SEQUENTIAL (smoother is stateful)
    #
    # NOTE: MHR2SMPLMultiView.infer_smpl_joints internally does
    #       `j -= j[0:1]` so the returned 24 joints are *pelvis-centered
    #       canonical* (pelvis at origin). Downstream (NLF renderer,
    #       SCAIL transforms, POSES format in general) expect camera-
    #       space coordinates where the pelvis sits at the person's
    #       actual depth. Add pred_cam_t back so the canonical pelvis
    #       lines up with where the body actually is in the scene.
    # -------------------------------------------------------------------
    mhr2smpl_time_s = 0.0
    t0 = time.perf_counter()
    for p_idx in range(n_persons):
        mhr2smpl.reset()
        for t in range(B):
            verts = persons[p_idx]["mhr_vertices"][t]
            cam_t = persons[p_idx]["mhr_cam_t"][t]
            if verts is None or cam_t is None:
                continue
            _, _, _, _, joints24 = mhr2smpl.infer_smpl_joints(
                views=[(verts, cam_t)],
                smpl_model_path=smpl_pkl,
            )
            # Translate canonical → camera space
            joints24_cam = joints24.astype(np.float32) + cam_t[None, :]
            persons[p_idx]["smpl_j3d"][t] = joints24_cam
    mhr2smpl_time_s = time.perf_counter() - t0

    return {
        "persons": persons,
        "cam_int": cam_int_per_frame,
        "fastsam3db_time_s": fastsam3db_time_s,
        "mhr2smpl_time_s": mhr2smpl_time_s,
    }
