"""
BMPRTMWPose — composite 2D pose pipeline.

Inputs:
    images       : (B, H, W, 3) float in [0, 1], RGB
    bmp_masks    : (B*N_total, H, W) float — frame-grouped per-tracked-person
                   masks from BMPInstanceSegmentation
    rtmw         : RTMW dict from LoadRTMWNode (133-keypoint base estimator)
    wilor (opt)  : WILOR dict from LoadWiLoRNode — when connected, overrides
                   the hand portion (91..132) of every person's COCO-WB
                   output with WiLoR's MANO-projected 21-point hands

Output:
    poses           : POSES dict (NPZ-compatible 133-keypoint COCO-WB layout)
    debug_overlay   : (B, H, W, 3) float — color-coded mask + skeleton
                      visualization for sanity-checking

Pipeline per frame:
    1. From bmp_masks, derive per-person mask + bbox (the BMP node already
       did the cross-frame tracking for us; we just consume its layout).
    2. Run RTMW-x ``inference_topdown`` with all per-person bboxes ➔ each
       person gets a (133, 2) keypoint + (133,) score array.
    3. Pack into the 133-layout COCO-WB slot directly (RTMW's output
       order matches COCO-WB, no remapping needed).
    4. (optional) Run WiLoR's hand detector on the whole image, project
       its 3D joints to 2D, and match each hand to a BMP-tracked person
       by wrist-proximity. Replace 91..111 (left) / 112..132 (right) of
       that person's keypoints with WiLoR's 21 points.

Score handling: RTMW's SimCC scores are unnormalized (typical range
0.2 - 3+). We normalize to [0, 1] via ``clip(s, 0, 3) / 3`` before
writing into the POSES dict so the existing PoseRenderer / Pose Editor
(which use 0-1 thresholds) work without changes.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import comfy.utils

from ._mask_utils import _DEBUG_PALETTE_RGB, build_debug_overlay, pack_mask, unpack_mask

_logger = logging.getLogger(__name__)


# COCO-17 / COCO-WB body wrist indices used for WiLoR-hand matching.
_LEFT_WRIST_IDX  = 9
_RIGHT_WRIST_IDX = 10

# RTMW SimCC scores are unnormalized; this is the empirical max we clamp
# at when normalising to [0, 1]. Picked so that "very confident" lands
# around 0.7-1.0 rather than always 1.0.
_RTMW_SCORE_NORM = 3.0


# COCO-17-ish skeleton edges for the debug overlay. We only draw body
# 0..16 + a couple of feet links — face / hand have too many points to
# render cleanly at typical preview resolution.
_DEBUG_SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (15, 19), (16, 22),  # ankles → heels (feet)
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


def _normalize_rtmw_score(score: np.ndarray) -> np.ndarray:
    """Map RTMW SimCC scores → [0, 1] via clip + scale."""
    return np.clip(score, 0.0, _RTMW_SCORE_NORM) / _RTMW_SCORE_NORM


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

def _draw_skeleton(frame_u8: np.ndarray, kp133: np.ndarray,
                    color_rgb: Tuple[int, int, int],
                    conf_thresh: float = 0.3,
                    radius: int = 3, thickness: int = 2):
    """Mutate frame_u8 in place: draw body+feet skeleton from a 133-pt array."""
    import cv2
    H, W = frame_u8.shape[:2]
    for k in range(min(23, kp133.shape[0])):  # body + feet only for clarity
        x, y, c = float(kp133[k, 0]), float(kp133[k, 1]), float(kp133[k, 2])
        if c < conf_thresh:
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(frame_u8, (ix, iy), radius, color_rgb,
                        thickness=-1, lineType=cv2.LINE_AA)
    for i, j in _DEBUG_SKELETON_EDGES:
        if i >= kp133.shape[0] or j >= kp133.shape[0]:
            continue
        ci, cj = float(kp133[i, 2]), float(kp133[j, 2])
        if ci < conf_thresh or cj < conf_thresh:
            continue
        p1 = (int(round(float(kp133[i, 0]))), int(round(float(kp133[i, 1]))))
        p2 = (int(round(float(kp133[j, 0]))), int(round(float(kp133[j, 1]))))
        if not (0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H):
            continue
        cv2.line(frame_u8, p1, p2, color_rgb,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_hands(frame_u8: np.ndarray, kp133: np.ndarray,
                 color_rgb: Tuple[int, int, int],
                 conf_thresh: float = 0.3, radius: int = 2):
    """Draw the 42 hand keypoints (91..132) as small dots."""
    import cv2
    H, W = frame_u8.shape[:2]
    for k in range(91, 133):
        if k >= kp133.shape[0]:
            break
        x, y, c = float(kp133[k, 0]), float(kp133[k, 1]), float(kp133[k, 2])
        if c < conf_thresh:
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(frame_u8, (ix, iy), radius, color_rgb,
                        thickness=-1, lineType=cv2.LINE_AA)


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
                "wilor": ("WILOR",),
            },
        }

    RETURN_TYPES = ("POSES", "IMAGE")
    RETURN_NAMES = ("poses", "debug_overlay")
    FUNCTION = "run"
    CATEGORY = "4dhumans"

    def run(self, images, bmp_masks, rtmw,
            score_threshold, fps, debug_overlay,
            wilor=None):
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
                sc = r.pred_instances.keypoint_scores[0]      # (133,)
                sc_norm = _normalize_rtmw_score(sc)
                kp133 = np.concatenate(
                    [kp.astype(np.float32), sc_norm[:, None].astype(np.float32)],
                    axis=-1,
                )
                persons_133[p_idx][t] = kp133

            pbar.update(1)
        rtmw_time = time.perf_counter() - rtmw_t

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
            "WiLoR %.2fs (%d hand overrides) | total %.2fs",
            B, n_persons, rtmw_time, wilor_time,
            wilor_overrides_count, elapsed,
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
        _logger.info("BMPRTMWPose debug overlay legend: %s", legend)

        arr = (overlay_t.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        for t in range(B):
            for p in range(n_persons):
                kp = persons_133[p][t]
                if kp is None:
                    continue
                color = _DEBUG_PALETTE_RGB[p % len(_DEBUG_PALETTE_RGB)]
                color_rgb = tuple(int(c) for c in color.tolist())
                _draw_skeleton(arr[t], kp, color_rgb,
                                conf_thresh=float(score_threshold))
                _draw_hands(arr[t], kp, color_rgb,
                             conf_thresh=float(score_threshold))

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
