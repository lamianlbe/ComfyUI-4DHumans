"""
BMPRTMWPose — composite 2D pose pipeline.

Layout decisions (each part of COCO-WB 133 sources from its strongest model):

    0..16    body  ← BMP   (mask-conditioned PMPose, OCHuman-SOTA)
    17..22   feet  ← RTMW-x (cocktail-14 trained, decent feet)
    23..90   face  ← FaRL  (per-person via BMP head 5pt → synthesised
                            RetinaFace 5pt → FaRL face_aligner)
    91..132  hands ← RTMW-x default; WiLoR override when connected

Each upgrade path is optional (degrades gracefully to RTMW for that
section) so the same node serves "all best models" and "RTMW-only
baseline" workflows without graph rewiring.

Inputs:
    images       : (B, H, W, 3) float in [0, 1], RGB
    bmp_masks    : (B*N_total, H, W) — frame-grouped per-tracked-person
                   masks from BMPInstanceSegmentation
    rtmw         : RTMW dict from LoadRTMWNode (133-keypoint base estimator)

Optional:
    bmp_pose     : YOLO11POSE-shaped dict from BMPInstanceSegmentation
                   — when connected, body 0..16 comes from BMP's track-
                   aligned 17-pt instead of RTMW's body slice.
    farl_face    : FARLFACE dict from LoadFaRLFace — when connected,
                   face 23..90 is regenerated per-person by feeding
                   BMP head keypoints into FaRL's face_aligner. NOT
                   based on RetinaFace+mask matching (which fails in
                   tightly overlapping multi-person scenes).
    wilor        : WILOR dict from LoadWiLoR — when connected, hands
                   91..132 get replaced with WiLoR's MANO-projected
                   21-point hands. Match each WiLoR-detected hand to
                   the closest BMP-tracked wrist; greedy 1-to-1.

Output:
    poses           : POSES dict (NPZ-compatible 133-keypoint COCO-WB layout)
    debug_overlay   : (B, H, W, 3) float — color-coded mask + skeleton
                      visualization for sanity-checking

Score handling: RTMW's SimCC head returns TWO score channels when the
codec has ``decode_visibility=True`` (our vendored RTMW-x configs do).
The first is ``keypoint_scores`` — raw ``min(max simcc_x, max simcc_y)``
unnormalised (typically 0.2 - 3+). The second is ``keypoints_visible``
— same peaks with ``simcc * decode_beta * sigma`` then softmax, giving
a clean [0, 1] probability. We prefer ``keypoints_visible`` when
available, falling back to ``clip(keypoint_scores, 0, 3) / 3`` only if
a future variant ships with ``decode_visibility=False``.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import comfy.utils

from ._farl_face_inference import run_farl_face_per_person
from ._mask_utils import build_debug_overlay, pack_mask, unpack_mask

_logger = logging.getLogger(__name__)


# COCO-17 / COCO-WB body wrist indices used for WiLoR-hand matching.
_LEFT_WRIST_IDX  = 9
_RIGHT_WRIST_IDX = 10

# RTMW SimCC scores are unnormalized; this is the empirical max we clamp
# at when normalising to [0, 1]. Picked so that "very confident" lands
# around 0.7-1.0 rather than always 1.0.
_RTMW_SCORE_NORM = 3.0


# Body skeleton edges — COCO-17 indices 0..16. Source: BMP body.
_DEBUG_BODY_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
    (5, 6), (5, 11), (6, 12), (11, 12),     # torso
    (0, 1), (0, 2), (1, 3), (2, 4),         # head
    (0, 5), (0, 6),                         # neck → shoulders
]

# Foot edges — COCO-WB indices 17..22. Source: RTMW.
# 17 left_big_toe, 18 left_small_toe, 19 left_heel,
# 20 right_big_toe, 21 right_small_toe, 22 right_heel.
_DEBUG_FOOT_EDGES = [
    (15, 17), (15, 18), (15, 19),  # left ankle → toes/heel
    (16, 20), (16, 21), (16, 22),  # right ankle → toes/heel
]

# Hand bone topology (MANO standard): wrist + 4 finger chains × 4 bones.
# Indices are 0..20 for left hand (COCO-WB 91..111) and same offsets
# for right hand (COCO-WB 112..132). Order:
#   0 wrist
#   1-4 thumb (CMC, MCP, IP, TIP)
#   5-8 index (MCP, PIP, DIP, TIP)
#   9-12 middle
#   13-16 ring
#   17-20 little
_HAND_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],          # thumb
    [0, 5, 6, 7, 8],          # index
    [0, 9, 10, 11, 12],       # middle
    [0, 13, 14, 15, 16],      # ring
    [0, 17, 18, 19, 20],      # little
]


def _bbox_from_mask(
    mask_bool: np.ndarray, padding_frac: float = 0.10,
) -> Optional[np.ndarray]:
    """Tight xyxy bbox from a bool mask plus fractional padding.

    Returns ``None`` for empty masks.
    """
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    w, h = x2 - x1, y2 - y1
    if padding_frac > 0:
        px, py = int(round(w * padding_frac)), int(round(h * padding_frac))
        x1 -= px; y1 -= py; x2 += px; y2 += py
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _normalize_rtmw_score_fallback(score: np.ndarray) -> np.ndarray:
    """Fallback for older / non-default RTMW configs that ship with
    ``decode_visibility=False`` and therefore don't expose
    ``keypoints_visible``. clip(s, 0, 3) / 3 is a coarse approximation
    of the softmax-normalised visibility — usable but not as clean as
    the real visibility output. Our vendored RTMW-x configs have
    ``decode_visibility=True`` so this path is rarely hit."""
    return np.clip(score, 0.0, _RTMW_SCORE_NORM) / _RTMW_SCORE_NORM


def _extract_kpt_scores(pred_instances) -> np.ndarray:
    """Pick the best [0, 1]-normalised score available from an mmpose
    ``InstanceData`` result. Prefers the softmax-normalised
    ``keypoints_visible`` field (set by SimCCLabel.decode when
    ``decode_visibility=True``); falls back to a clipped version of
    raw ``keypoint_scores`` otherwise.

    Returns ``(K,)`` float32 in [0, 1].
    """
    visible = getattr(pred_instances, "keypoints_visible", None)
    if visible is not None:
        # Already softmax-normalised in [0, 1]; just take instance 0.
        return np.asarray(visible[0], dtype=np.float32)
    raw = np.asarray(pred_instances.keypoint_scores[0], dtype=np.float32)
    return _normalize_rtmw_score_fallback(raw)


# --------------------------------------------------------------------------
# WiLoR helpers
# --------------------------------------------------------------------------

def _project_full_img(points_3d: np.ndarray, cam_t: np.ndarray,
                       focal: float, img_size: np.ndarray) -> np.ndarray:
    """Project 3D points (in camera frame, after cam_t translation) to
    original-image 2D pixel coordinates. Mirrors WiLoR's
    ``demo.py::project_full_img``.

    Args:
        points_3d: (N, 3) joints in WiLoR's normalised coords.
        cam_t:     (3,)  translation from cam_crop_to_full.
        focal:     scalar focal length scaled to original image.
        img_size:  (W, H) original image size.

    Returns:
        (N, 2) pixel coords in the original image.
    """
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(img_size[0]) / 2.0
    K[1, 2] = float(img_size[1]) / 2.0

    pts = points_3d.astype(np.float64) + cam_t.astype(np.float64)
    pts = pts / pts[..., -1:]                  # perspective divide
    pts2d = (K @ pts.T).T                       # (N, 3)
    return pts2d[..., :2].astype(np.float32)


# --------------------------------------------------------------------------
# ViTPose fallback (two backends: HF transformers + ONNX)
# --------------------------------------------------------------------------
#
# Both backends produce the same ``(N, 17, 3)`` output (body 0..16) so
# BMPRTMWPose's fallback path is backend-agnostic. ``vitpose["backend"]``
# selects which path to run. The HF backend uses transformers'
# VitPoseImageProcessor + post_process_pose_estimation. The ONNX
# backend rolls its own top-down affine + heatmap decode.

# ImageNet normalisation, used by both backends. Values match mmpose
# ViTPose default and HF transformers VitPoseImageProcessor default.
_VITPOSE_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VITPOSE_STD_RGB  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _topdown_affine(image_rgb: np.ndarray, bbox_xyxy: np.ndarray,
                     input_h: int, input_w: int,
                     padding: float = 1.25):
    """Crop+resize to (input_h, input_w) keeping bbox aspect-aligned.

    Returns:
        crop:  (input_h, input_w, 3) uint8 RGB
        M_inv: (2, 3) affine matrix mapping crop coords → image coords
    """
    import cv2 as _cv2

    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Match input aspect ratio by inflating the shorter side.
    aspect = float(input_w) / float(input_h)
    if bw > bh * aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    bw *= padding
    bh *= padding

    # Build affine via 3-point correspondence (mmpose convention).
    src = np.array([
        [cx,            cy],
        [cx,            cy + bh * 0.5],
        [cx + bw * 0.5, cy],
    ], dtype=np.float32)
    dst = np.array([
        [input_w * 0.5, input_h * 0.5],
        [input_w * 0.5, input_h],
        [input_w,       input_h * 0.5],
    ], dtype=np.float32)
    M     = _cv2.getAffineTransform(src, dst)
    M_inv = _cv2.getAffineTransform(dst, src)

    crop = _cv2.warpAffine(
        image_rgb, M, (input_w, input_h), flags=_cv2.INTER_LINEAR,
    )
    return crop, M_inv


def _run_vitpose_onnx_batch(
    vitpose_dict: dict,
    img_rgb: np.ndarray,
    bboxes_xyxy: List[np.ndarray],
) -> Optional[np.ndarray]:
    """ONNX backend. Top-down affine crop → ImageNet normalise →
    onnxruntime → heatmap argmax decode → inverse affine. Returns
    ``(N, K, 3)`` where K is the model's full keypoint count
    (17 for body-only ViTPose, 133 for wholebody)."""
    if not bboxes_xyxy:
        return None

    session    = vitpose_dict["session"]
    input_name = vitpose_dict["input_name"]
    input_h    = vitpose_dict["input_h"]
    input_w    = vitpose_dict["input_w"]

    # Per-bbox top-down affine
    crops = []
    inv_mats = []
    for bb in bboxes_xyxy:
        crop, M_inv = _topdown_affine(img_rgb, bb, input_h, input_w)
        crops.append(crop)
        inv_mats.append(M_inv)
    crops_arr = np.stack(crops, axis=0).astype(np.float32) / 255.0  # (N, H, W, 3) [0,1] RGB
    crops_arr = (crops_arr - _VITPOSE_MEAN_RGB) / _VITPOSE_STD_RGB
    crops_arr = crops_arr.transpose(0, 3, 1, 2).astype(np.float32)   # (N, 3, H, W)

    try:
        outputs = session.run(None, {input_name: crops_arr})
    except Exception as e:
        _logger.error(
            "ViTPose ONNX inference failed (%d boxes): %s",
            len(bboxes_xyxy), e,
        )
        return None

    heatmaps = outputs[0]  # (N, K, hm_h, hm_w)
    if heatmaps.ndim != 4:
        _logger.warning(
            "ViTPose ONNX unexpected output shape %s — expected "
            "(N, K, h, w). Skipping batch.", heatmaps.shape,
        )
        return None

    N, K, hm_h, hm_w = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = flat.argmax(axis=-1)
    y_hm = (idx // hm_w).astype(np.float32)
    x_hm = (idx %  hm_w).astype(np.float32)
    scores = flat.max(axis=-1)

    # Heatmap → input crop space
    sx = float(input_w) / float(hm_w)
    sy = float(input_h) / float(hm_h)
    x_in = x_hm * sx
    y_in = y_hm * sy

    # Return ALL K keypoints (caller picks slices: 0..16 body, 17..22
    # feet, 91..132 hands, etc.). For 17-keypoint body-only ViTPose
    # K==17 so caller can only fill body. For wholebody K==133 caller
    # can fill body+feet+hands.
    out = np.zeros((N, K, 3), dtype=np.float32)
    for n in range(N):
        M_inv = inv_mats[n]
        for k in range(K):
            xc, yc = x_in[n, k], y_in[n, k]
            x_orig = M_inv[0, 0] * xc + M_inv[0, 1] * yc + M_inv[0, 2]
            y_orig = M_inv[1, 0] * xc + M_inv[1, 1] * yc + M_inv[1, 2]
            out[n, k, 0] = x_orig
            out[n, k, 1] = y_orig
            out[n, k, 2] = scores[n, k]
    return out


def _run_vitpose_hf_batch(
    vitpose_dict: dict,
    img_rgb: np.ndarray,
    bboxes_xyxy: List[np.ndarray],
) -> Optional[np.ndarray]:
    """HF transformers backend. Uses VitPoseImageProcessor's built-in
    pre/post processing."""
    if not bboxes_xyxy:
        return None

    from PIL import Image as _PIL

    model      = vitpose_dict["model"]
    processor  = vitpose_dict["processor"]
    device     = vitpose_dict["device"]
    is_plus    = vitpose_dict["is_plus"]
    torch_dtype = vitpose_dict.get("torch_dtype", torch.float32)

    # xyxy → xywh per ViTPose's processor convention.
    boxes_xywh = []
    for bb in bboxes_xyxy:
        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        boxes_xywh.append([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)])

    pil_img = _PIL.fromarray(img_rgb)

    try:
        inputs = processor(
            pil_img, boxes=[boxes_xywh], return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if torch_dtype != torch.float32 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)

        with torch.inference_mode():
            forward_kwargs = dict(inputs)
            if is_plus:
                forward_kwargs["dataset_index"] = torch.tensor(
                    [0] * len(boxes_xywh), device=device,
                )
            outputs = model(**forward_kwargs)

        results = processor.post_process_pose_estimation(
            outputs, boxes=[boxes_xywh],
        )
        per_person = results[0]
    except Exception as e:
        _logger.error(
            "ViTPose HF batch failed (%d boxes): %s", len(boxes_xywh), e,
        )
        return None

    if len(per_person) != len(boxes_xywh):
        _logger.warning(
            "ViTPose HF returned %d results for %d boxes — alignment "
            "broken, skipping batch.",
            len(per_person), len(boxes_xywh),
        )
        return None

    # Determine the keypoint count from the first result so the output
    # shape matches whatever variant we loaded (typically 17 for
    # ViTPose-simple body-only, but ViTPose+ multi-task can theoretically
    # emit different counts depending on dataset_index).
    if not per_person:
        return None
    first_kp = np.asarray(per_person[0]["keypoints"], dtype=np.float32)
    K = int(first_kp.shape[0])

    out = np.zeros((len(per_person), K, 3), dtype=np.float32)
    for i, person in enumerate(per_person):
        kp = np.asarray(person["keypoints"], dtype=np.float32)
        sc = np.asarray(person["scores"],    dtype=np.float32)
        n_kp = min(K, kp.shape[0])
        out[i, :n_kp, 0] = kp[:n_kp, 0]
        out[i, :n_kp, 1] = kp[:n_kp, 1]
        out[i, :n_kp, 2] = sc[:n_kp]
    return out


def _run_vitpose_batch(
    vitpose_dict: dict,
    img_rgb: np.ndarray,
    bboxes_xyxy: List[np.ndarray],
) -> Optional[np.ndarray]:
    """Backend-dispatching wrapper. Returns ``(N, 17, 3)`` or None."""
    backend = vitpose_dict.get("backend", "hf")
    if backend == "onnx":
        return _run_vitpose_onnx_batch(vitpose_dict, img_rgb, bboxes_xyxy)
    return _run_vitpose_hf_batch(vitpose_dict, img_rgb, bboxes_xyxy)


def _run_wilor_one_frame(
    wilor_dict: dict,
    img_bgr: np.ndarray,
) -> List[Dict[str, Any]]:
    """Run WiLoR hand detection + pose on one image, return a list of
    per-hand records:

        {
            "bbox_xyxy":   (4,) float in original image coords
            "is_right":    bool — WiLoR's own L/R class
            "joints_2d":   (21, 2) float in original image coords
            "det_conf":    float — YOLO detection confidence
        }

    Empty list if no hands detected.
    """
    # Defensive: LoadWiLoRNode normally adds wilor_lib/ to sys.path, but
    # if the node was loaded from ComfyUI's cache without re-running
    # __init__, the path injection might be missing. Idempotent.
    from ..wilor_lib import ensure_lib_importable as _ensure
    _ensure()

    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    detector = wilor_dict["detector"]
    model = wilor_dict["model"]
    model_cfg = wilor_dict["model_cfg"]
    device = wilor_dict["device"]
    fast = wilor_dict.get("fast", False)
    det_conf = wilor_dict.get("detector_conf", 0.3)

    dets = detector(img_bgr, conf=float(det_conf), verbose=False)[0]
    bboxes = []
    is_right = []
    confs = []
    for det in dets:
        Bbox = det.boxes.data.cpu().detach().squeeze(0).numpy()
        if Bbox.ndim == 0:
            continue
        bboxes.append(Bbox[:4].tolist())
        confs.append(float(Bbox[4]) if len(Bbox) >= 5 else 1.0)
        # YOLO class: 0 = left, 1 = right
        is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))

    if not bboxes:
        return []

    boxes = np.stack(bboxes, axis=0)
    right = np.stack(is_right, axis=0)
    dataset = ViTDetDataset(
        model_cfg, img_bgr, boxes, right, rescale_factor=2.0, fp16=fast,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0,
    )

    results: List[Dict[str, Any]] = []
    cur_idx = 0
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        # Camera params + projection (per WiLoR demo.py)
        multiplier = (2 * batch["right"] - 1)
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size   = batch["box_size"].float()
        img_size   = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * img_size.max()
        )
        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length,
            )
            .detach().cpu().numpy()
        )

        # ``scaled_focal_length`` is a 0-dim tensor (img_size.max() over
        # the batch — i.e. ONE focal scalar shared by every hand crop in
        # this image). It's NOT indexable per-hand. WiLoR's demo.py uses
        # it as a scalar (``focal_length=scaled_focal_length``); we do
        # the same here, after a single .item() conversion outside the
        # per-hand loop so the cost stays out of the inner loop.
        focal_scalar = float(scaled_focal_length.item())

        bsz = batch["img"].shape[0]
        for n in range(bsz):
            joints_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()  # (21, 3)
            is_right_n = float(batch["right"][n].cpu().numpy())
            # WiLoR mirrors left hands to right-hand canonical pose; un-mirror
            # by flipping x axis when this is a left hand.
            joints_3d[:, 0] = (2 * is_right_n - 1) * joints_3d[:, 0]
            cam_t = pred_cam_t_full[n]
            joints_2d = _project_full_img(
                joints_3d, cam_t,
                focal_scalar,
                img_size[n].cpu().numpy(),
            )

            results.append({
                "bbox_xyxy": np.array(boxes[cur_idx], dtype=np.float32),
                "is_right":  bool(is_right_n > 0.5),
                "joints_2d": joints_2d,                  # (21, 2)
                "det_conf":  float(confs[cur_idx]),
            })
            cur_idx += 1

    return results


def _assign_wilor_hands_to_persons(
    wilor_hands: List[Dict[str, Any]],
    person_keypoints_133: List[Optional[np.ndarray]],
    person_bboxes: List[Optional[np.ndarray]],
    wrist_score_thresh: float = 0.3,
    radius_frac_of_diag: float = 0.20,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Match each WiLoR-detected hand to a tracked-person slot via
    wrist proximity. Returns nested dict ``{person_idx: {"left": hand,
    "right": hand}}`` — at most one of each side per person.

    Match logic:
        - For each hand, find the (person, side) wrist closest to the
          hand's bbox center.
        - Distance threshold = ``radius_frac_of_diag × person_bbox_diag``.
        - Greedy 1-to-1: highest-score (lowest-distance) pairs win first.
        - WiLoR's own ``is_right`` flag is **ignored** — geometric
          proximity to the matched wrist decides the side. WiLoR
          mis-classifies L/R fairly often on mirrored / unusual poses.
    """
    candidates = []
    for h_idx, hand in enumerate(wilor_hands):
        x1, y1, x2, y2 = hand["bbox_xyxy"]
        hand_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])

        for p_idx, kp133 in enumerate(person_keypoints_133):
            if kp133 is None:
                continue
            pbb = person_bboxes[p_idx]
            if pbb is None:
                continue
            person_diag = float(
                np.hypot(pbb[2] - pbb[0], pbb[3] - pbb[1])
            )
            radius = person_diag * radius_frac_of_diag

            for wrist_idx, side in [(_LEFT_WRIST_IDX, "left"),
                                     (_RIGHT_WRIST_IDX, "right")]:
                w = kp133[wrist_idx]
                # Note: scores are normalised by caller
                if w[2] < wrist_score_thresh:
                    continue
                dist = float(np.hypot(
                    hand_center[0] - w[0],
                    hand_center[1] - w[1],
                ))
                if dist > radius:
                    continue
                candidates.append({
                    "score":  -dist,
                    "h_idx":  h_idx,
                    "p_idx":  p_idx,
                    "side":   side,
                })

    candidates.sort(key=lambda c: -c["score"])
    used_hands = set()
    used_slots = set()  # (p_idx, side)
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for c in candidates:
        if c["h_idx"] in used_hands:
            continue
        if (c["p_idx"], c["side"]) in used_slots:
            continue
        out.setdefault(c["p_idx"], {})[c["side"]] = wilor_hands[c["h_idx"]]
        used_hands.add(c["h_idx"])
        used_slots.add((c["p_idx"], c["side"]))
    return out


# --------------------------------------------------------------------------
# Debug overlay
# --------------------------------------------------------------------------

# Region color scheme. Each model contributes a different region of the
# 133-keypoint COCO-WB layout, so we use a FIXED color per region (NOT
# per-track). This makes it visually obvious which model's output is
# being looked at.
#
#   green   ← BMP body  (0..16)
#   yellow  ← RTMW feet (17..22)
#   cyan    ← FaRL face (23..90)
#   magenta ← left hand (91..111)   — RTMW or WiLoR
#   orange  ← right hand (112..132) — RTMW or WiLoR
#
# Mask & bbox in the underlying overlay still use one color per track
# (from build_debug_overlay's palette) so person identity is also
# preserved at a glance.
# OKS sigma per body joint, copied from COCO's standard table (used by
# pycocotools.cocoeval). Same sigmas mmpose uses internally for
# evaluation. Distance × 2σ²·s² in the OKS formula — small sigmas
# (eyes, hips) penalise distance more than large ones (feet).
_COCO17_OKS_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,   # nose, eyes, ears
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,   # shoulders, elbows, wrists
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,   # hips, knees, ankles
], dtype=np.float32)


def _body_oks(
    kp_a: np.ndarray, kp_b: np.ndarray,
    bbox_a: Optional[np.ndarray], bbox_b: Optional[np.ndarray],
    score_threshold: float,
) -> float:
    """Object Keypoint Similarity between two (17, 3) body skeletons.

    Returns 0.0 if there are fewer than 5 jointly-visible keypoints
    (insufficient evidence to call them the same person). Otherwise
    follows the COCO OKS formula:
        OKS = mean_i exp(-d_i² / (2 σ_i² s²))
    where ``s²`` = bbox area (we use mean of the two bboxes).
    """
    visible = (kp_a[:17, 2] >= score_threshold) & (kp_b[:17, 2] >= score_threshold)
    if int(visible.sum()) < 5:
        return 0.0

    diff = kp_a[:17, :2] - kp_b[:17, :2]
    d2 = np.sum(diff * diff, axis=1)

    # bbox area used to normalise distances. Average of the two bboxes
    # — both should describe the SAME person if they're a ghost pair,
    # so any discrepancy means the bboxes themselves are off-spec.
    def _area(bb):
        if bb is None:
            return 0.0
        w = max(0.0, bb[2] - bb[0])
        h = max(0.0, bb[3] - bb[1])
        return float(w * h)

    area_a, area_b = _area(bbox_a), _area(bbox_b)
    if area_a > 0 and area_b > 0:
        s2 = 0.5 * (area_a + area_b)
    elif area_a > 0:
        s2 = area_a
    elif area_b > 0:
        s2 = area_b
    else:
        s2 = max(d2.max(), 1.0)  # degenerate: just normalise by max d²

    e = d2 / (2.0 * _COCO17_OKS_SIGMAS ** 2 * s2 + 1e-9)
    contrib = np.exp(-e)
    return float(contrib[visible].mean())


_BODY_COLOR  = (50,  255, 50)    # green
_FOOT_COLOR  = (255, 220, 0)     # yellow
_FACE_COLOR  = (0,   220, 255)   # cyan
_LHAND_COLOR = (255, 0,   200)   # magenta
_RHAND_COLOR = (255, 140, 0)     # orange


def _draw_keypoints(frame_u8: np.ndarray, kp133: np.ndarray,
                     indices, color_rgb: Tuple[int, int, int],
                     conf_thresh: float, radius: int):
    """Helper: draw a subset of keypoints as filled circles."""
    import cv2
    H, W = frame_u8.shape[:2]
    for k in indices:
        if k >= kp133.shape[0]:
            continue
        x, y, c = float(kp133[k, 0]), float(kp133[k, 1]), float(kp133[k, 2])
        if c < conf_thresh:
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(frame_u8, (ix, iy), radius, color_rgb,
                        thickness=-1, lineType=cv2.LINE_AA)


def _draw_edges(frame_u8: np.ndarray, kp133: np.ndarray,
                 edges, base_offset: int,
                 color_rgb: Tuple[int, int, int],
                 conf_thresh: float, thickness: int):
    """Helper: draw a list of (i, j) edges. ``base_offset`` is added
    to the index pair (lets us reuse a 0-relative finger topology
    for both left-hand 91..111 and right-hand 112..132)."""
    import cv2
    H, W = frame_u8.shape[:2]
    for i, j in edges:
        ai, aj = i + base_offset, j + base_offset
        if ai >= kp133.shape[0] or aj >= kp133.shape[0]:
            continue
        ci, cj = float(kp133[ai, 2]), float(kp133[aj, 2])
        if ci < conf_thresh or cj < conf_thresh:
            continue
        p1 = (int(round(float(kp133[ai, 0]))), int(round(float(kp133[ai, 1]))))
        p2 = (int(round(float(kp133[aj, 0]))), int(round(float(kp133[aj, 1]))))
        if not (0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H):
            continue
        cv2.line(frame_u8, p1, p2, color_rgb,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_body(frame_u8, kp133, conf_thresh):
    """COCO-17 body 0..16 — sourced from BMP. Green."""
    _draw_keypoints(frame_u8, kp133, range(17),
                     _BODY_COLOR, conf_thresh, radius=3)
    _draw_edges(frame_u8, kp133, _DEBUG_BODY_EDGES,
                 base_offset=0, color_rgb=_BODY_COLOR,
                 conf_thresh=conf_thresh, thickness=2)


def _draw_feet(frame_u8, kp133, conf_thresh):
    """COCO-WB feet 17..22 — sourced from RTMW. Yellow."""
    _draw_keypoints(frame_u8, kp133, range(17, 23),
                     _FOOT_COLOR, conf_thresh, radius=3)
    _draw_edges(frame_u8, kp133, _DEBUG_FOOT_EDGES,
                 base_offset=0, color_rgb=_FOOT_COLOR,
                 conf_thresh=conf_thresh, thickness=2)


def _draw_face(frame_u8, kp133, conf_thresh):
    """COCO-WB face 23..90 — sourced from FaRL when connected else RTMW.
    Cyan. Drawn as small dots; the 68-point iBUG layout is too dense
    for skeleton edges to read cleanly at preview resolution."""
    _draw_keypoints(frame_u8, kp133, range(23, 91),
                     _FACE_COLOR, conf_thresh, radius=1)


def _draw_hands(frame_u8, kp133, conf_thresh):
    """COCO-WB hands 91..132 — sourced from RTMW or WiLoR.
    Magenta (left) + orange (right). MANO finger-chain skeleton."""
    # Left hand
    _draw_keypoints(frame_u8, kp133, range(91, 112),
                     _LHAND_COLOR, conf_thresh, radius=2)
    for chain in _HAND_FINGER_CHAINS:
        edges = list(zip(chain[:-1], chain[1:]))
        _draw_edges(frame_u8, kp133, edges, base_offset=91,
                     color_rgb=_LHAND_COLOR, conf_thresh=conf_thresh,
                     thickness=1)
    # Right hand
    _draw_keypoints(frame_u8, kp133, range(112, 133),
                     _RHAND_COLOR, conf_thresh, radius=2)
    for chain in _HAND_FINGER_CHAINS:
        edges = list(zip(chain[:-1], chain[1:]))
        _draw_edges(frame_u8, kp133, edges, base_offset=112,
                     color_rgb=_RHAND_COLOR, conf_thresh=conf_thresh,
                     thickness=1)


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class BMPRTMWPoseNode:
    """Compose BMP (tracking + masks) + RTMW (133-pt base) + optional
    WiLoR (high-quality hands) into a unified COCO-WB POSES dict.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":     ("IMAGE",),
                "bmp_masks":  ("MASK",),
                "rtmw":       ("RTMW",),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Minimum normalised keypoint score to mark a "
                            "joint as visible (third column of the (133, "
                            "3) output). Used downstream by PoseRenderer / "
                            "Pose Editor as their visibility cutoff. "
                            "RTMW raw scores get clamped to [0, 3] and "
                            "divided by 3 first, so 0.3 here ≈ raw 0.9."
                        ),
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 240.0,
                        "step": 1.0,
                        "tooltip": (
                            "Output FPS metadata written into the POSES "
                            "dict (used by Save NPZ / Pose Editor). "
                            "Doesn't affect inference, only metadata."
                        ),
                    },
                ),
                "ghost_oks_thresh": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.3,
                        "max": 0.99,
                        "step": 0.05,
                        "tooltip": (
                            "Two slots are considered 'ghost duplicates' "
                            "(same real person split into two BMP mask "
                            "fragments) on a frame when their COCO-17 "
                            "body OKS exceeds this threshold. 0.7 is a "
                            "safe default — distinct people typically "
                            "score < 0.3 even when standing close."
                        ),
                    },
                ),
                "ghost_max_burst_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 60,
                        "step": 1,
                        "tooltip": (
                            "Maximum length (in frames) of a "
                            "transient ghost-overlap burst that gets "
                            "suppressed. Set to 0 to disable suppression "
                            "entirely. Bursts longer than this are left "
                            "alone — they're likely two different "
                            "people, not a SAM mask split."
                        ),
                    },
                ),
                "recovery_max_gap_frames": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 60,
                        "step": 1,
                        "tooltip": (
                            "Track-recovery merge — if a slot goes "
                            "silent and a NEW slot's first detection "
                            "appears within this many frames, AND "
                            "their poses match (OKS ≥ "
                            "recovery_oks_thresh), AND the two slots "
                            "never co-existed in any prior frame, "
                            "the new slot's data is merged back into "
                            "the old one to keep IDs stable across "
                            "brief disappearances. Set to 0 to "
                            "disable. Strict-by-default (10 = ~0.3s "
                            "@ 30fps); raise carefully — large gaps "
                            "make false merges more likely."
                        ),
                    },
                ),
                "recovery_oks_thresh": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.3,
                        "max": 0.99,
                        "step": 0.05,
                        "tooltip": (
                            "OKS threshold for track-recovery merging. "
                            "Same person across a 10-frame gap "
                            "typically scores > 0.7; distinct people "
                            "score < 0.3. 0.65 is a slightly lenient "
                            "default to forgive small pose changes "
                            "during occlusion. Mirrors / matches the "
                            "ghost-suppression OKS threshold so "
                            "behaviour is symmetric."
                        ),
                    },
                ),
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Render mask + body+feet skeleton + hand "
                            "dots overlay on the input frames. CPU "
                            "post-processing — moderate cost. When off, "
                            "debug_overlay is just images passed through."
                        ),
                    },
                ),
            },
            "optional": {
                "bmp_pose":  ("YOLO11POSE",),
                "farl_face": ("FARLFACE",),
                "wilor":     ("WILOR",),
                "vitpose":   ("VITPOSE",),
            },
        }

    RETURN_TYPES = ("POSES", "IMAGE")
    RETURN_NAMES = ("poses", "debug_overlay")
    FUNCTION = "run"
    CATEGORY = "4dhumans"

    def run(self, images, bmp_masks, rtmw,
            score_threshold, fps,
            ghost_oks_thresh, ghost_max_burst_frames,
            recovery_max_gap_frames, recovery_oks_thresh,
            debug_overlay,
            bmp_pose=None, farl_face=None, wilor=None, vitpose=None):
        from mmpose.apis import inference_topdown

        rtmw_model = rtmw["model"]
        rtmw_device = rtmw["device"]

        # ---- Decode shapes ------------------------------------------------
        # images: (B, H, W, 3) RGB float in [0, 1]
        # bmp_masks: (B*N_total, H, W) float in {0, 1}
        B, H, W, _C = images.shape
        total = bmp_masks.shape[0]
        if total == 0 or B == 0:
            _logger.warning(
                "BMPRTMWPose: no input frames or no masks; returning "
                "empty POSES."
            )
            return self._empty_output(images, fps, B, H, W, debug_overlay)
        if total % B != 0:
            raise ValueError(
                f"bmp_masks shape {tuple(bmp_masks.shape)} not divisible "
                f"by frame count {B} — expected (B*N, H, W) layout."
            )
        n_persons = total // B

        # Convert to BGR uint8 once for cv2 / WiLoR / mmpose consumption
        rgb_u8 = (images.clamp(0, 1) * 255).byte().cpu().numpy()
        bgr_u8 = rgb_u8[..., ::-1].copy()

        masks_bool = (bmp_masks.detach().cpu().numpy() > 0.5)
        masks_bool = masks_bool.reshape(B, n_persons, H, W)

        pbar = comfy.utils.ProgressBar(B + 1)
        t0 = time.perf_counter()

        # ---- Phase 1: per-frame per-person bbox ---------------------------
        # bbox_per_frame[t][p_idx] = xyxy or None if mask is empty
        bbox_per_frame: List[List[Optional[np.ndarray]]] = []
        for t in range(B):
            row: List[Optional[np.ndarray]] = []
            for p in range(n_persons):
                bb = _bbox_from_mask(masks_bool[t, p], padding_frac=0.10)
                if bb is not None:
                    bb[0] = max(0, bb[0]); bb[1] = max(0, bb[1])
                    bb[2] = min(W, bb[2]); bb[3] = min(H, bb[3])
                    if bb[2] - bb[0] < 5 or bb[3] - bb[1] < 5:
                        bb = None
                row.append(bb)
            bbox_per_frame.append(row)

        # ---- Phase 2: RTMW-x per (frame, person) → 133 keypoints ----------
        persons_133: List[List[Optional[np.ndarray]]] = [
            [None] * B for _ in range(n_persons)
        ]
        rtmw_t = time.perf_counter()
        for t in range(B):
            bboxes_this_frame = []
            slot_indices = []
            for p in range(n_persons):
                bb = bbox_per_frame[t][p]
                if bb is None:
                    continue
                bboxes_this_frame.append(bb.tolist())
                slot_indices.append(p)
            if not bboxes_this_frame:
                pbar.update(1)
                continue

            try:
                results = inference_topdown(
                    rtmw_model, bgr_u8[t],
                    bboxes=bboxes_this_frame, bbox_format="xyxy",
                )
            except Exception as e:
                _logger.error("RTMW failed on frame %d: %s", t, e)
                pbar.update(1)
                continue

            for r_idx, r in enumerate(results):
                p_idx = slot_indices[r_idx]
                kp = r.pred_instances.keypoints[0]            # (133, 2)
                sc = _extract_kpt_scores(r.pred_instances)    # (133,) in [0, 1]
                kp133 = np.concatenate(
                    [kp.astype(np.float32), sc[:, None].astype(np.float32)],
                    axis=-1,
                )
                persons_133[p_idx][t] = kp133

            pbar.update(1)
        rtmw_time = time.perf_counter() - rtmw_t

        # Per-(slot, frame) body source tracking. None means body 0..16
        # is still RTMW's output (which we DON'T want in the final
        # POSES dict — pipeline policy: body never comes from RTMW).
        # After Phase 2.5/2.7 finish, Phase 2.8 nulls every (p, t)
        # whose body_source is still None — entire kp133 set to None
        # so nothing leaks.
        body_source: List[List[Optional[str]]] = [
            [None] * B for _ in range(n_persons)
        ]

        # Sanity gate: at least ONE body source must be wired up.
        # Otherwise the entire output is empty (per the policy above).
        if bmp_pose is None and vitpose is None:
            _logger.warning(
                "BMPRTMWPose: neither bmp_pose nor vitpose connected. "
                "By design this pipeline never uses RTMW's body output, "
                "so the result will have NO body keypoints (and feet "
                "/ hands tied to body — entire (slot, frame) entries "
                "will be NULL). Connect at least one of bmp_pose or "
                "vitpose."
            )

        # ---- Phase 2.5: BMP body source (0..16) --------------------------
        # When BMPInstanceSegmentation provides track-aligned 17-pt
        # output for a (slot, frame), claim body for it. This is the
        # PRIMARY body source when bmp_pose is connected.
        #
        # Trigger: BMP detection exists for this (p, t) — i.e.
        # ``track_aligned[p][t]`` is not None. We don't gate on
        # confidence: BMP's tracker only emits a non-None entry when
        # it actually has a detection, so existence == "BMP saw a
        # person here".
        body_override_count = 0
        if bmp_pose is not None:
            track_aligned = bmp_pose.get("_track_aligned_kpts", None)
            if track_aligned is None:
                _logger.warning(
                    "BMPRTMWPose: bmp_pose was connected but the dict "
                    "doesn't carry '_track_aligned_kpts' — that field "
                    "was added 2026-04-27. Re-run BMPInstanceSegmentation "
                    "with the latest node code or skip the bmp_pose input."
                )
            else:
                bmp_n = len(track_aligned)
                if bmp_n != n_persons:
                    _logger.warning(
                        "BMPRTMWPose: bmp_pose slots (%d) don't match "
                        "mask slots (%d) — connecting bmp_pose from a "
                        "DIFFERENT BMPInstanceSegmentation than "
                        "bmp_masks? Body override skipped.",
                        bmp_n, n_persons,
                    )
                else:
                    for p_idx in range(n_persons):
                        for t in range(B):
                            kp17 = track_aligned[p_idx][t]
                            kp133 = persons_133[p_idx][t]
                            if kp17 is None or kp133 is None:
                                continue
                            # BMP keypoints are (17, 3) with PMPose scores
                            # already in [0, 1]. Direct slot replacement.
                            kp133[0:17] = kp17.astype(np.float32)
                            body_source[p_idx][t] = "bmp"
                            body_override_count += 1

        # ---- Phase 2.7: ViTPose fallback (body + feet + maybe hands) -----
        # Trigger: any (slot, frame) whose body_source is still None
        # (i.e. BMP wasn't connected, OR BMP didn't detect this person
        # this frame). ViTPose tries; if it can produce a confident
        # body it claims body_source = "vit" and ALSO fills feet+hands.
        #
        # Slice ownership when fallback fires:
        #   0..16   body  ← ViTPose
        #   17..22  feet  ← ViTPose                 (if K >= 23)
        #   23..90  face  ← UNCHANGED — FaRL Phase 2.6 or RTMW base
        #   91..132 hands ← ViTPose                 (only if WiLoR not
        #                                              connected; else
        #                                              Phase 3 will own it)
        #
        # ViTPose body-only models (K=17) fill only body — feet/hands
        # stay at None for this (p, t) and Phase 2.8 cleans them up.
        # Wholebody models (K=133) give us enough to fully populate
        # the slot when WiLoR isn't around.
        #
        # Sanity gate: ViTPose must have body max-conf >= score_threshold;
        # otherwise leave body_source as None (Phase 2.8 will null
        # the entire (p, t)).
        vitpose_fallback_count = 0
        vitpose_feet_filled    = 0
        vitpose_hands_filled   = 0
        if vitpose is not None:
            from collections import defaultdict
            backend = vitpose.get("backend", "hf")
            wilor_will_override = wilor is not None
            _logger.info(
                "ViTPose fallback enabled (backend=%s). Fires for "
                "(slot, frame) pairs without a BMP detection. Slice "
                "ownership: body+feet from ViTPose, face stays with "
                "FaRL/RTMW, hands from %s.",
                backend,
                "WiLoR (Phase 3 override)" if wilor_will_override
                else "ViTPose",
            )
            vt_t = time.perf_counter()

            # Group fallback requests by frame so each ViTPose forward
            # runs all that frame's missing-body bboxes in one shot.
            requests_by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = (
                defaultdict(list)
            )
            for t in range(B):
                for p in range(n_persons):
                    if body_source[p][t] is not None:
                        continue   # Already claimed by BMP
                    kp133 = persons_133[p][t]
                    if kp133 is None:
                        continue
                    bb = bbox_per_frame[t][p]
                    if bb is None:
                        continue
                    requests_by_frame[t].append((p, bb))

            for t, slot_bbox_pairs in requests_by_frame.items():
                slots = [pair[0] for pair in slot_bbox_pairs]
                bboxes_for_vt = [pair[1] for pair in slot_bbox_pairs]

                # _run_vitpose_batch returns (N, K, 3); K is the
                # loaded model's keypoint count (17 body-only or 133
                # wholebody).
                vt_out = _run_vitpose_batch(
                    vitpose, rgb_u8[t], bboxes_for_vt,
                )
                if vt_out is None:
                    continue

                K_returned = vt_out.shape[1]
                has_feet  = K_returned >= 23
                has_hands = K_returned >= 133

                for i, p_idx in enumerate(slots):
                    kp133 = persons_133[p_idx][t]
                    if kp133 is None:
                        continue

                    new_kp = vt_out[i].astype(np.float32)
                    body_new = new_kp[:17]
                    body_new_max = float(body_new[:, 2].max())
                    # Sanity gate: ViTPose's own body confidence must
                    # clear threshold. If it can't see a body either,
                    # body_source stays None → Phase 2.8 nulls the
                    # entire (p, t).
                    if body_new_max < float(score_threshold):
                        continue

                    # Body 0..16
                    kp133[0:17] = body_new
                    body_source[p_idx][t] = "vit"
                    vitpose_fallback_count += 1

                    # Feet 17..22
                    if has_feet:
                        kp133[17:23] = new_kp[17:23]
                        vitpose_feet_filled += 1

                    # Hands 91..132 — only when WiLoR won't override
                    # (saves a copy; otherwise Phase 3 takes over).
                    if has_hands and not wilor_will_override:
                        kp133[91:133] = new_kp[91:133]
                        vitpose_hands_filled += 1

            vitpose_time = time.perf_counter() - vt_t
        else:
            vitpose_time = 0.0

        # ---- Phase 2.8: clear unsourced bodies (RTMW body never leaks) ---
        # Per pipeline policy, body keypoints must come from BMP or
        # ViTPose. Any (slot, frame) still using RTMW's body output
        # gets the entire kp133 nulled — feet/hands tied to body
        # presence, so the whole entry is meaningless without a real
        # body source.
        cleared_no_body = 0
        for p in range(n_persons):
            for t in range(B):
                if body_source[p][t] is None and persons_133[p][t] is not None:
                    persons_133[p][t] = None
                    cleared_no_body += 1
        if cleared_no_body > 0:
            _logger.info(
                "Cleared %d (slot, frame) pairs with no body source "
                "(RTMW body never leaks per pipeline policy).",
                cleared_no_body,
            )

        # ---- Phase 2.6: optional FaRL face override (23..90) -------------
        # When farl_face is connected, regenerate the face slice using
        # BMP's per-person head 5pt → synthesise FaRL 5pt landmarks →
        # face_aligner. This bypasses the RetinaFace + mask matching
        # path entirely (which fails in tight overlap) and instead
        # leverages the per-person face↔body association we already
        # have from BMP.
        farl_time = 0.0
        farl_override_count = 0
        if farl_face is not None:
            farl_t = time.perf_counter()

            # Source for head 5pt: prefer BMP body (clean per-person
            # association by construction), fall back to RTMW body slice
            # if bmp_pose isn't connected.
            head_kp_per_track: List[List[Optional[np.ndarray]]] = [
                [None] * B for _ in range(n_persons)
            ]
            if bmp_pose is not None and bmp_pose.get("_track_aligned_kpts") is not None:
                bmp_kpts = bmp_pose["_track_aligned_kpts"]
                if len(bmp_kpts) == n_persons:
                    for p_idx in range(n_persons):
                        for t in range(B):
                            kp17 = bmp_kpts[p_idx][t]
                            if kp17 is not None:
                                # Head = COCO indices 0..4
                                head_kp_per_track[p_idx][t] = (
                                    kp17[:5].astype(np.float32).copy()
                                )
            else:
                # Fall back to RTMW body slice (also COCO 0..4 layout).
                for p_idx in range(n_persons):
                    for t in range(B):
                        kp133 = persons_133[p_idx][t]
                        if kp133 is None:
                            continue
                        head_kp_per_track[p_idx][t] = (
                            kp133[0:5].astype(np.float32).copy()
                        )

            face_kp_68_timeline, _ = run_farl_face_per_person(
                images_np_u8=rgb_u8,
                head_kp_per_track=head_kp_per_track,
                farl_face_dict=farl_face,
                img_h=H, img_w=W,
                head_conf_thresh=float(score_threshold),
                frame_batch_size=32,
                pbar=None,
            )

            for p_idx in range(n_persons):
                for t in range(B):
                    face68 = face_kp_68_timeline[p_idx][t]
                    kp133 = persons_133[p_idx][t]
                    if face68 is None or kp133 is None:
                        continue
                    # face68 is already (68, 3) with conf=1.0. Drop into
                    # COCO-WB face slot 23..90 (68 iBUG / 300W landmarks).
                    kp133[23:91] = face68.astype(np.float32)
                    farl_override_count += 1

            farl_time = time.perf_counter() - farl_t

        # ---- Phase 3: optional WiLoR hand override ------------------------
        wilor_time = 0.0
        wilor_overrides_count = 0
        if wilor is not None:
            wilor_t = time.perf_counter()
            for t in range(B):
                wilor_hands = _run_wilor_one_frame(wilor, bgr_u8[t])
                if not wilor_hands:
                    continue

                this_frame_kp133 = [persons_133[p][t] for p in range(n_persons)]
                this_frame_bbox = [bbox_per_frame[t][p] for p in range(n_persons)]
                assignments = _assign_wilor_hands_to_persons(
                    wilor_hands, this_frame_kp133, this_frame_bbox,
                    wrist_score_thresh=float(score_threshold),
                    radius_frac_of_diag=0.20,
                )

                for p_idx, sides in assignments.items():
                    kp133 = persons_133[p_idx][t]
                    if kp133 is None:
                        continue
                    for side, hand in sides.items():
                        joints_2d = hand["joints_2d"]                 # (21, 2)
                        # WiLoR doesn't output per-keypoint confidence; use
                        # the YOLO det conf as a single scalar across all 21.
                        conf_col = np.full(
                            (21, 1), float(hand["det_conf"]), dtype=np.float32,
                        )
                        joints_213 = np.concatenate(
                            [joints_2d.astype(np.float32), conf_col], axis=-1,
                        )
                        if side == "left":
                            kp133[91:112] = joints_213
                        else:
                            kp133[112:133] = joints_213
                        wilor_overrides_count += 1
            wilor_time = time.perf_counter() - wilor_t

        pbar.update(1)

        # ---- Phase 3.5: ghost-track suppression ---------------------------
        # In tightly-overlapping multi-person scenes, BMP's mask tracker
        # sometimes splits a single real person into two slots for a
        # short window — think upper body in slot A, lower body in
        # slot B, both fragments getting their own mask-bbox. RTMW
        # then runs on both bboxes and produces nearly-identical
        # skeletons because the underlying ViT crop sees the same body
        # parts. The result is two visually-overlapping skeletons in a
        # 2-5 frame burst, then collapse back to one slot. Worse, the
        # IoU tracker often re-assigns the dominant slot ID after the
        # burst, manifesting as a track-ID swap.
        #
        # Detection: for each ordered pair (a, b) and each frame
        # where BOTH have body keypoints, compute COCO-17 OKS. Frames
        # with OKS > ghost_oks_thresh are "overlap frames". Group
        # consecutive overlap frames into bursts. If a burst is short
        # (≤ ghost_max_burst_frames) we declare the lower-quality slot
        # in that burst (sum of body keypoint scores) the ghost and
        # null its keypoints in the affected frames. Long bursts are
        # left alone — they're more likely two genuinely close people.
        ghosts_suppressed_frames = 0
        ghosts_suppressed_bursts = 0
        ghosts_swaps_applied = 0
        # Burst metadata collected during the suppression pass and
        # consumed by the swap-detection pass below: each entry is
        # ``(slot_a, slot_b, r_start, r_end)`` with slot_a < slot_b
        # (canonical ordering matches the pair-enumeration loop).
        suppressed_bursts: List[Tuple[int, int, int, int]] = []
        if ghost_max_burst_frames > 0 and n_persons >= 2:
            # Per-frame body bboxes (re-derived from kp133 since we may
            # have overridden body via BMP — bbox should track the
            # final keypoints, not the original mask bbox).
            def _body_bbox_from_kp133(kp133, conf_thresh):
                if kp133 is None:
                    return None
                vis = kp133[:17, 2] >= conf_thresh
                if int(vis.sum()) < 3:
                    return None
                pts = kp133[:17][vis][:, :2]
                return np.array([
                    pts[:, 0].min(), pts[:, 1].min(),
                    pts[:, 0].max(), pts[:, 1].max(),
                ], dtype=np.float32)

            def _kp_quality(kp133, conf_thresh):
                if kp133 is None:
                    return -1.0
                vis = kp133[:17, 2] >= conf_thresh
                if not vis.any():
                    return 0.0
                return float(kp133[:17][vis, 2].sum())

            # Pass 1: build overlap matrix flag[(a, b)][t] = True if pair
            # is ghost-overlapping that frame.
            overlap_per_frame: List[Dict[Tuple[int, int], bool]] = [
                {} for _ in range(B)
            ]
            for t in range(B):
                for a in range(n_persons):
                    kpa = persons_133[a][t]
                    if kpa is None:
                        continue
                    for b in range(a + 1, n_persons):
                        kpb = persons_133[b][t]
                        if kpb is None:
                            continue
                        bbox_a = _body_bbox_from_kp133(kpa, score_threshold)
                        bbox_b = _body_bbox_from_kp133(kpb, score_threshold)
                        oks = _body_oks(
                            kpa, kpb, bbox_a, bbox_b,
                            score_threshold,
                        )
                        if oks > float(ghost_oks_thresh):
                            overlap_per_frame[t][(a, b)] = True

            # Pass 2: for each (a, b) pair, find contiguous overlap
            # bursts and suppress short ones.
            checked_pairs = set()
            for a in range(n_persons):
                for b in range(a + 1, n_persons):
                    if (a, b) in checked_pairs:
                        continue
                    checked_pairs.add((a, b))
                    # Find runs of contiguous frames where (a, b) overlap.
                    runs: List[Tuple[int, int]] = []
                    run_start = None
                    for t in range(B):
                        is_overlap = (a, b) in overlap_per_frame[t]
                        if is_overlap and run_start is None:
                            run_start = t
                        elif not is_overlap and run_start is not None:
                            runs.append((run_start, t - 1))
                            run_start = None
                    if run_start is not None:
                        runs.append((run_start, B - 1))

                    for r_start, r_end in runs:
                        burst_len = r_end - r_start + 1
                        if burst_len > int(ghost_max_burst_frames):
                            continue  # likely a real adjacent person, leave it

                        # Per-burst quality comparison: sum body
                        # confidences across the burst. Slot with lower
                        # total is the ghost.
                        qa = sum(
                            _kp_quality(persons_133[a][t], score_threshold)
                            for t in range(r_start, r_end + 1)
                        )
                        qb = sum(
                            _kp_quality(persons_133[b][t], score_threshold)
                            for t in range(r_start, r_end + 1)
                        )
                        ghost_slot = a if qa < qb else b
                        keep_slot  = b if ghost_slot == a else a

                        for t in range(r_start, r_end + 1):
                            if persons_133[ghost_slot][t] is not None:
                                persons_133[ghost_slot][t] = None
                                ghosts_suppressed_frames += 1
                        ghosts_suppressed_bursts += 1
                        suppressed_bursts.append((a, b, r_start, r_end))
                        _logger.info(
                            "Ghost suppression: pair (slot %d, slot %d) "
                            "frames %d..%d (%d frames) — kept slot %d, "
                            "nulled slot %d (quality %.2f vs %.2f).",
                            a, b, r_start, r_end, burst_len,
                            keep_slot, ghost_slot, qa, qb,
                        )

            # ---- Phase 3.6: post-burst slot-swap correction ---------------
            # Suppressing a ghost only blanks the duplicate keypoints
            # WITHIN the burst. It does nothing about BMP's IoU tracker
            # potentially swapping the real person's slot when the
            # mask-fragments collapse back to one mask after the burst.
            #
            # Detect that case here: for each burst (a, b) at frames
            # [r_start, r_end], compare each slot's pose immediately
            # before vs immediately after the burst. If "post-burst
            # slot a" matches "pre-burst slot b" better than itself,
            # the tracker swapped — fix by swapping persons_133[a]
            # with persons_133[b] from r_end+1 to end of video.
            #
            # Process bursts in chronological r_end order so cascading
            # swaps see the corrected state from earlier bursts.
            #
            # Three cases are handled:
            #   (1) Both slots have pre AND post poses → compare
            #       no-swap vs swap OKS sums, swap if margin > 0.2
            #   (2) Slot a active before, slot b active after, the
            #       opposite slots empty around the burst → classic
            #       SAM-split pattern. Swap if cross-OKS > 0.5.
            #   (3) Mirror of (2): slot b active before, slot a after.
            for (sa, sb, sr_start, sr_end) in sorted(
                suppressed_bursts, key=lambda x: (x[3], x[2]),
            ):
                # Find latest non-None pose in slot s before burst start.
                def _pose_before(slot: int, t_burst_start: int):
                    for tt in range(t_burst_start - 1, -1, -1):
                        kp = persons_133[slot][tt]
                        if kp is not None:
                            return kp, tt
                    return None, -1

                # Find earliest non-None pose in slot s after burst end.
                def _pose_after(slot: int, t_burst_end: int):
                    for tt in range(t_burst_end + 1, B):
                        kp = persons_133[slot][tt]
                        if kp is not None:
                            return kp, tt
                    return None, -1

                pre_a,  pre_a_t  = _pose_before(sa, sr_start)
                pre_b,  pre_b_t  = _pose_before(sb, sr_start)
                post_a, post_a_t = _pose_after(sa,  sr_end)
                post_b, post_b_t = _pose_after(sb,  sr_end)

                do_swap = False
                swap_reason = ""

                if (pre_a is not None and pre_b is not None
                        and post_a is not None and post_b is not None):
                    # Case 1: full data — pick the configuration with
                    # higher cumulative OKS by a clear margin.
                    bb_pre_a  = _body_bbox_from_kp133(pre_a,  score_threshold)
                    bb_pre_b  = _body_bbox_from_kp133(pre_b,  score_threshold)
                    bb_post_a = _body_bbox_from_kp133(post_a, score_threshold)
                    bb_post_b = _body_bbox_from_kp133(post_b, score_threshold)

                    oks_no = (
                        _body_oks(pre_a, post_a, bb_pre_a, bb_post_a, score_threshold)
                        + _body_oks(pre_b, post_b, bb_pre_b, bb_post_b, score_threshold)
                    )
                    oks_yes = (
                        _body_oks(pre_a, post_b, bb_pre_a, bb_post_b, score_threshold)
                        + _body_oks(pre_b, post_a, bb_pre_b, bb_post_a, score_threshold)
                    )
                    # Margin of 0.2 to avoid hair-trigger swaps when
                    # the two configurations are nearly tied (which
                    # happens when both slots track genuinely-similar
                    # poses).
                    if oks_yes > oks_no + 0.2:
                        do_swap = True
                        swap_reason = (
                            f"OKS swapped={oks_yes:.3f} vs no-swap={oks_no:.3f}"
                        )

                elif (pre_a is not None and post_b is not None
                      and pre_b is None and post_a is None):
                    # Case 2: classic mask-split swap — slot a was the
                    # active slot before burst, the new track-id (slot
                    # b) carries the person after burst.
                    bb_pre_a  = _body_bbox_from_kp133(pre_a,  score_threshold)
                    bb_post_b = _body_bbox_from_kp133(post_b, score_threshold)
                    oks_x = _body_oks(
                        pre_a, post_b, bb_pre_a, bb_post_b, score_threshold,
                    )
                    if oks_x > 0.5:
                        do_swap = True
                        swap_reason = (
                            f"slot{sa}→{sb} transfer, OKS(pre_a, post_b)"
                            f"={oks_x:.3f}"
                        )

                elif (pre_b is not None and post_a is not None
                      and pre_a is None and post_b is None):
                    # Case 3: mirror of case 2.
                    bb_pre_b  = _body_bbox_from_kp133(pre_b,  score_threshold)
                    bb_post_a = _body_bbox_from_kp133(post_a, score_threshold)
                    oks_x = _body_oks(
                        pre_b, post_a, bb_pre_b, bb_post_a, score_threshold,
                    )
                    if oks_x > 0.5:
                        do_swap = True
                        swap_reason = (
                            f"slot{sb}→{sa} transfer, OKS(pre_b, post_a)"
                            f"={oks_x:.3f}"
                        )

                if do_swap:
                    # In-place swap of the two slot timelines from
                    # ``sr_end + 1`` to the end of the video. Subsequent
                    # bursts processed later will see the corrected
                    # state — that's why we sort by r_end.
                    for tt in range(sr_end + 1, B):
                        persons_133[sa][tt], persons_133[sb][tt] = (
                            persons_133[sb][tt], persons_133[sa][tt]
                        )
                    ghosts_swaps_applied += 1
                    _logger.info(
                        "Slot swap: from frame %d, slot %d ↔ slot %d "
                        "(post-burst remap; %s)",
                        sr_end + 1, sa, sb, swap_reason,
                    )

        # ---- Phase 3.7: track recovery merge (lost-and-back-again) --------
        # When a person briefly leaves the frame / gets fully occluded /
        # has all keypoints drop below conf threshold, BMP's IoU
        # tracker often spawns a new track-id when they reappear (since
        # the old slot's last mask was IoU=0 vs the gap). The result is
        # a fragmented identity: e.g. Alice = slot 0 for frames 0..150,
        # then slot 0 goes empty, then she's slot 4 for frames 161..300.
        #
        # If the gap is short (≤ recovery_max_gap_frames) and the pose
        # at slot 0's last frame matches slot 4's first frame (OKS ≥
        # recovery_oks_thresh) AND the two slots NEVER co-existed in
        # any frame (sanity gate against merging two genuinely
        # different people), we move slot 4's data into slot 0 and
        # null slot 4 for those frames.
        #
        # Strict by default: 10-frame gap (≈0.3s @ 30fps) and OKS 0.65
        # rule out almost all false merges, at the cost of leaving
        # longer disappearances unfixed (user said this is acceptable).
        recovery_merges = 0
        if recovery_max_gap_frames > 0 and n_persons >= 2:
            # Build active intervals per slot — runs of consecutive
            # frames where ``persons_133[s][t]`` is not None.
            intervals: List[Tuple[int, int, int]] = []  # (slot, start, end)
            for s in range(n_persons):
                run_start = None
                for t in range(B):
                    if persons_133[s][t] is not None and run_start is None:
                        run_start = t
                    elif persons_133[s][t] is None and run_start is not None:
                        intervals.append((s, run_start, t - 1))
                        run_start = None
                if run_start is not None:
                    intervals.append((s, run_start, B - 1))

            # Process intervals chronologically. For each interval we
            # try to absorb subsequent intervals (in any other slot)
            # that match by OKS within the recovery gap window.
            intervals.sort(key=lambda x: (x[1], x[2]))
            merged_indices: set = set()

            for j in range(len(intervals)):
                if j in merged_indices:
                    continue
                s_j, st_j, en_j = intervals[j]

                k = j + 1
                while k < len(intervals):
                    if k in merged_indices:
                        k += 1
                        continue
                    s_k, st_k, en_k = intervals[k]

                    if s_k == s_j:
                        # Same slot — already disjoint by interval
                        # construction, nothing to merge.
                        k += 1
                        continue

                    gap = st_k - en_j - 1  # frames of None between intervals
                    if gap > int(recovery_max_gap_frames):
                        # All later intervals are even further away
                        # (sorted by start). Stop scanning.
                        break

                    if st_k <= en_j:
                        # Time-overlap. Two slots active in the same
                        # frame ⇒ different people (ghost handler
                        # would already have collapsed any genuine
                        # ghost). Skip.
                        k += 1
                        continue

                    # Pose continuity check via OKS at the boundary.
                    kp_a = persons_133[s_j][en_j]
                    kp_b = persons_133[s_k][st_k]
                    bbox_a = _body_bbox_from_kp133(kp_a, score_threshold)
                    bbox_b = _body_bbox_from_kp133(kp_b, score_threshold)
                    oks = _body_oks(
                        kp_a, kp_b, bbox_a, bbox_b, score_threshold,
                    )

                    if oks < float(recovery_oks_thresh):
                        k += 1
                        continue

                    # Sanity gate: if these two slots EVER co-existed
                    # in any frame across the whole video, they must
                    # be different people (ghost handler would have
                    # caught a SAM-split co-existence). Refuse to
                    # merge.
                    both_ever = any(
                        (persons_133[s_j][t] is not None
                         and persons_133[s_k][t] is not None)
                        for t in range(B)
                    )
                    if both_ever:
                        k += 1
                        continue

                    # Merge: move slot s_k's interval data into slot s_j,
                    # leaving slot s_k Nulled for those frames.
                    for t in range(st_k, en_k + 1):
                        persons_133[s_j][t] = persons_133[s_k][t]
                        persons_133[s_k][t] = None
                    merged_indices.add(k)
                    recovery_merges += 1
                    _logger.info(
                        "Track recovery: slot %d frames %d..%d → slot %d "
                        "(gap=%d, OKS=%.3f)",
                        s_k, st_k, en_k, s_j, gap, oks,
                    )

                    # The merged data extends slot s_j's interval to
                    # ``en_k``. Continue scanning forward — a later
                    # interval might be reachable now that en_j has
                    # advanced.
                    en_j = en_k
                    k += 1

        # ---- Phase 4: pack POSES dict -------------------------------------
        poses_persons = []
        for p_idx in range(n_persons):
            kp_main = persons_133[p_idx]
            kp_raw  = [kp.copy() if kp is not None else None for kp in kp_main]
            poses_persons.append({
                "visible":       True,
                # 3D fields kept as None placeholders so the existing
                # NPZ schema (Save Pose Data / Load Pose Data) stays
                # backward-compatible without conditional handling.
                "body_joints2d": [None] * B,
                "body_joints":   [None] * B,
                "smpl_j3d":      [None] * B,
                "keypoints":     kp_main,
                "keypoints_raw": kp_raw,
            })

        poses = {
            "n_persons": n_persons,
            "n_frames":  B,
            "img_h":     int(H),
            "img_w":     int(W),
            "fps":       float(fps),
            "persons":   poses_persons,
            # Camera intrinsics not estimated in this 2D-only pipeline —
            # leave None so downstream tools that need them know to fall
            # back (e.g. NLF converter would compute its own).
            "cam_int": [None] * B,
            "scale":   [None] * B,
            "offset":  [None] * B,
        }

        elapsed = time.perf_counter() - t0
        _logger.info(
            "BMPRTMWPose: %d frames, %d tracks | RTMW %.2fs | "
            "BMP body: %d | ViTPose %.2fs (body=%d feet=%d hands=%d) "
            "| cleared %d no-body slots | FaRL face %.2fs (%d) | "
            "WiLoR %.2fs (%d) | ghost: %d bursts (%d nulled, %d "
            "swaps) | recovery: %d merges | total %.2fs",
            B, n_persons, rtmw_time,
            body_override_count,
            vitpose_time, vitpose_fallback_count,
            vitpose_feet_filled, vitpose_hands_filled,
            cleared_no_body,
            farl_time, farl_override_count,
            wilor_time, wilor_overrides_count,
            ghosts_suppressed_bursts, ghosts_suppressed_frames,
            ghosts_swaps_applied,
            recovery_merges,
            elapsed,
        )

        # ---- Phase 5: debug overlay ---------------------------------------
        if not debug_overlay:
            return (poses, images)

        per_frame_items = [
            [(p, pack_mask(masks_bool[t, p])) for p in range(n_persons)
             if masks_bool[t, p].any()]
            for t in range(B)
        ]
        overlay_t, legend = build_debug_overlay(
            images=images, per_frame_items=per_frame_items, H=H, W=W,
        )
        _logger.info("BMPRTMWPose debug overlay legend (mask): %s", legend)
        _logger.info(
            "BMPRTMWPose debug overlay legend (joints): "
            "body=green | feet=yellow | face=cyan | "
            "left-hand=magenta | right-hand=orange"
        )

        arr = (overlay_t.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        # Per-region drawing — each model's contribution gets its own
        # color (body=green, feet=yellow, face=cyan, lhand=magenta,
        # rhand=orange). The mask + bbox layer underneath uses one color
        # per track (from build_debug_overlay's palette), so reading the
        # overlay you get track identity from the mask color and joint
        # source from the keypoint color.
        ct = float(score_threshold)
        for t in range(B):
            for p in range(n_persons):
                kp = persons_133[p][t]
                if kp is None:
                    continue
                _draw_body(arr[t], kp, conf_thresh=ct)
                _draw_feet(arr[t], kp, conf_thresh=ct)
                _draw_face(arr[t], kp, conf_thresh=ct)
                _draw_hands(arr[t], kp, conf_thresh=ct)

        out_overlay = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return (poses, out_overlay)

    @staticmethod
    def _empty_output(images, fps, B, H, W, debug_overlay):
        empty_poses = {
            "n_persons": 0,
            "n_frames":  B,
            "img_h":     int(H),
            "img_w":     int(W),
            "fps":       float(fps),
            "persons":   [],
            "cam_int":   [None] * B,
            "scale":     [None] * B,
            "offset":    [None] * B,
        }
        return (empty_poses, images)
