"""
RTMPose-Face inference helper (via rtmlib).

Given per-person face bboxes derived from Fast SAM 3D Body's head
keypoints, run the RTMPose-Face ONNX model through rtmlib's
``RTMPose.__call__`` which handles preprocessing (top-down affine crop
→ 256×256), ONNX forward, SimCC argmax decode and back-projection in
one shot.

Output is 68-point face landmarks in the original image's pixel
coordinates (mapped from rtmlib's 106-point LaPa output via the
_LAPA106_TO_300W68 table).

NOTE: this is the "pure rtmlib" baseline — no side-prior / mirror-peak
workaround is applied. We use this specifically to confirm the
alternating-frame bug reproduces with rtmlib's reference
implementation, proving it's a model-level issue, not our code. Once
confirmed, a side-prior decoder wrapper will be layered on top.
"""

import logging
from typing import Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


# =============================================================================
# LaPa 106 → 300W 68 subset mapping (unchanged from previous impl)
# =============================================================================
_LAPA106_TO_300W68 = np.array([
    # --- Jaw 17 pts (300W 0-16) from LaPa 0-32: every-other sample ---
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    # --- Left eyebrow 5 pts (300W 17-21) from LaPa 33-37 ---
    33, 34, 35, 36, 37,
    # --- Right eyebrow 5 pts (300W 22-26) from LaPa 42-46 ---
    42, 43, 44, 45, 46,
    # --- Nose bridge 4 pts (300W 27-30) from LaPa 51-54 ---
    51, 52, 53, 54,
    # --- Nose horizontal / nostrils 5 pts (300W 31-35) from LaPa 55-65 ---
    55, 57, 60, 63, 65,
    # --- Left eye 6 pts (300W 36-41) from LaPa 66-74 ---
    # order: outer_corner, upper_outer, upper_inner, inner_corner,
    #        lower_inner, lower_outer
    66, 67, 68, 70, 73, 71,
    # --- Right eye 6 pts (300W 42-47) from LaPa 75-83 ---
    75, 76, 77, 79, 82, 80,
    # --- Outer lip 12 pts (300W 48-59) from LaPa 84-95 ---
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    # --- Inner lip 8 pts (300W 60-67) from LaPa 96-103 ---
    96, 97, 98, 99, 100, 101, 102, 103,
], dtype=np.int64)


def _get_face_bbox_from_coco_wb_head(
    coco_wb_body_feet: np.ndarray,  # (23, 3) or (23, 2)
    img_h: int,
    img_w: int,
    expand_ratio: float = 1.8,
) -> Optional[Tuple[int, int, int, int]]:
    """Derive a square, padded face bbox from the nose + eyes + ears
    (COCO-WB indices 0-4).

    Returns ``None`` when head keypoints are too sparse to locate a face.
    """
    head = coco_wb_body_feet[:5]  # (5, D)
    if head.shape[1] >= 3:
        conf = head[:, 2]
    else:
        conf = np.ones(5, dtype=np.float32)
    valid = conf > 0.1
    if valid.sum() < 2:
        return None

    pts = head[valid, :2]
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    # Use the max distance between valid head points as the face scale
    if len(pts) >= 2:
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        base = float(dists.max()) + 1.0
    else:
        base = max(img_h, img_w) * 0.05
    half = 0.5 * base * expand_ratio

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    # Clamp to image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


# =============================================================================
# Main entry point: run RTMPose-Face over the whole video via rtmlib
# =============================================================================

def run_rtmpose_face_video(
    images_np_u8: np.ndarray,              # (B, H, W, 3) uint8 RGB
    persons_coco_body_feet_timeline: list, # per-person list:
                                           #   persons_body_feet[p_idx][t] = (23, 3) or None
    rtmpose_face_dict: dict,               # from LoadRTMPoseFace
    img_h: int,
    img_w: int,
    batch_size: int = 16,                  # unused — rtmlib runs one bbox at a time
    pbar=None,
    debug_dump_path: str | None = None,     # if set, save raw 106-pt output
):
    """Run RTMPose-Face on every visible (frame, person) slot and return
    per-person face-68 timelines.

    Returns
    -------
    face_kp_68_timeline : list[list[np.ndarray | None]]
        face_kp_68_timeline[p_idx][t] = (68, 3) with pixel (x, y, conf)
        or None when no valid face bbox.
    time_s : float
    """
    face_est = rtmpose_face_dict["face"]  # rtmlib.RTMPose instance

    n_persons = len(persons_coco_body_feet_timeline)
    B = images_np_u8.shape[0]

    if batch_size != 1:
        _logger.info(
            "RTMPose-Face (rtmlib): batch_size=%d ignored — rtmlib processes "
            "one bbox per forward pass internally.", batch_size,
        )

    # Pre-allocate output
    face_kp_68_timeline = [
        [None] * B for _ in range(n_persons)
    ]

    # Build per-frame bbox / person lists
    per_frame_bboxes = [[] for _ in range(B)]
    per_frame_persons = [[] for _ in range(B)]
    for p_idx in range(n_persons):
        for t in range(B):
            head = persons_coco_body_feet_timeline[p_idx][t]
            if head is None:
                continue
            bbox = _get_face_bbox_from_coco_wb_head(head, img_h, img_w)
            if bbox is None:
                continue
            per_frame_bboxes[t].append(list(bbox))
            per_frame_persons[t].append(p_idx)

    total_faces = sum(len(bs) for bs in per_frame_bboxes)
    if total_faces == 0:
        return face_kp_68_timeline, 0.0

    import time as _time
    t_start = _time.perf_counter()

    # Optional debug: collect raw 106-pt output + bbox for every (person,
    # frame) so we can diagnose frame-to-frame stability issues.
    debug_dump = None
    if debug_dump_path is not None:
        debug_dump = {
            "bboxes": [], "kpts_106": [],
            "frame_idx": [], "person_idx": [],
        }

    for t in range(B):
        bboxes_t = per_frame_bboxes[t]
        if not bboxes_t:
            continue
        frame_rgb = images_np_u8[t]  # (H, W, 3) uint8 RGB

        # rtmlib returns (keypoints, scores) where keypoints is
        # (n_boxes, 106, 2) in ORIGINAL image pixel coords and scores
        # is (n_boxes, 106).
        try:
            keypoints, scores = face_est(frame_rgb, bboxes_t)
        except Exception as e:
            _logger.error(
                "rtmlib RTMPose-Face inference failed on frame %d with "
                "%d bboxes: %s",
                t, len(bboxes_t), e,
            )
            if pbar is not None:
                pbar.update(len(bboxes_t))
            continue

        # Sanity: rtmlib concatenates all bboxes so keypoints.shape[0]
        # must equal len(bboxes_t).
        if keypoints.shape[0] != len(bboxes_t):
            _logger.warning(
                "rtmlib returned %d keypoint sets for %d bboxes on frame "
                "%d — dropping this frame.",
                keypoints.shape[0], len(bboxes_t), t,
            )
            if pbar is not None:
                pbar.update(len(bboxes_t))
            continue

        for i, p_idx in enumerate(per_frame_persons[t]):
            kpts_xy = np.asarray(keypoints[i], dtype=np.float32)   # (106, 2)
            scr     = np.asarray(scores[i],    dtype=np.float32)   # (106,)
            K = kpts_xy.shape[0]

            kpts_xyz = np.concatenate(
                [kpts_xy, scr[:, None]], axis=-1,
            ).astype(np.float32)  # (K, 3)

            if debug_dump is not None:
                debug_dump["bboxes"].append(np.array(bboxes_t[i], dtype=np.float32))
                debug_dump["kpts_106"].append(kpts_xyz)
                debug_dump["frame_idx"].append(int(t))
                debug_dump["person_idx"].append(int(p_idx))

            if K != 106:
                _logger.warning(
                    "rtmlib returned %d keypoints (expected 106 LaPa) on "
                    "frame %d person %d — keeping the first 68 raw pts.",
                    K, t, p_idx,
                )
                face_kp_68_timeline[p_idx][t] = kpts_xyz[:min(K, 68)]
            else:
                # Map 106 → 68 via LaPa → 300W table
                kpts_68 = kpts_xyz[_LAPA106_TO_300W68]  # (68, 3)
                face_kp_68_timeline[p_idx][t] = kpts_68.astype(np.float32)

        if pbar is not None:
            pbar.update(len(bboxes_t))

    # Persist raw debug dump if requested
    if debug_dump is not None and debug_dump["kpts_106"]:
        np.savez(
            debug_dump_path,
            bboxes=np.stack(debug_dump["bboxes"]),
            kpts_106=np.stack(debug_dump["kpts_106"]),
            frame_idx=np.array(debug_dump["frame_idx"], dtype=np.int32),
            person_idx=np.array(debug_dump["person_idx"], dtype=np.int32),
        )
        _logger.info(
            "RTMPose-Face (rtmlib) raw 106-pt debug dump saved to %s  "
            "(%d entries)",
            debug_dump_path, len(debug_dump["kpts_106"]),
        )

    return face_kp_68_timeline, _time.perf_counter() - t_start
