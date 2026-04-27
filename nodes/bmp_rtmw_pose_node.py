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
                float(scaled_focal_length[n].item()
                      if hasattr(scaled_focal_length, "__getitem__")
                      else scaled_focal_length),
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
            },
        }

    RETURN_TYPES = ("POSES", "IMAGE")
    RETURN_NAMES = ("poses", "debug_overlay")
    FUNCTION = "run"
    CATEGORY = "4dhumans"

    def run(self, images, bmp_masks, rtmw,
            score_threshold, fps, debug_overlay,
            bmp_pose=None, farl_face=None, wilor=None):
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

        # ---- Phase 2.5: optional BMP body override (0..16) ----------------
        # When the user wires in BMPInstanceSegmentation's `bmp_pose` we
        # replace RTMW's body section with BMP's track-aligned 17-pt
        # output. BMP's mask-conditioned PMPose is OCHuman-SOTA whereas
        # RTMW-x is a generic WholeBody estimator — for occluded /
        # tightly-overlapping POV scenes BMP's body wins.
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
                # BMP and our slot count must match — both come from the
                # same IoU-tracked persons via the same MASK output, so
                # they index the same N_total tracks.
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
                            body_override_count += 1

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
            "BMP body overrides: %d | FaRL face %.2fs (%d overrides) | "
            "WiLoR %.2fs (%d hand overrides) | total %.2fs",
            B, n_persons, rtmw_time,
            body_override_count,
            farl_time, farl_override_count,
            wilor_time, wilor_overrides_count,
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
