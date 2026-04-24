"""
RTMPose-Face inference helper.

Given per-person face bboxes derived from Fast SAM 3D Body's head
keypoints, run the RTMPose-Face ONNX model batched across (frame, person)
slots and output 68-point face landmarks in the original image's pixel
coordinates.

The public RTMPose-Face Face6 checkpoint outputs **106 landmarks** at
256x256 input.  We map them to the 68-point COCO-WholeBody / 300W
convention via a fixed index table (LaPa 106 → 300W 68).

Pipeline per face:
  1. compute face bbox from MHR nose + eyes + ears, square it, pad 1.5x
  2. warp-resize the image crop to 256x256 via affine transform
     (matching MMPose's top-down preprocessing — preserves aspect)
  3. ImageNet normalise → NCHW float32
  4. ONNX forward returns SimCC heatmaps (x + y)
  5. argmax on each SimCC axis → 106 keypoints in 256x256 space
  6. inverse affine back to original pixel coords
  7. drop to 68 points via the LaPa106→300W68 map
"""

import logging
from typing import Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


# =============================================================================
# LaPa 106 → 300W 68 subset mapping
#
# The Face6 model's 106-landmark convention follows LaPa/InsightFace
# order:
#   0-32   : jawline / face contour (33 points)
#   33-42  : left eyebrow (10)
#   43-52  : right eyebrow (10)
#   53-57  : nose bridge (5)
#   58-62  : nose tip horizontal (5)
#   63-74  : left eye contour (12 incl. pupil)
#   75-86  : right eye contour (12 incl. pupil)
#   87-105 : mouth outer + inner (20)
#   (plus 2 pupil centres inside the eye indices)
#
# 300W 68 layout:
#   0-16   : jaw  (17)
#   17-21  : left brow (5)
#   22-26  : right brow (5)
#   27-30  : nose bridge (4)
#   31-35  : nose horizontal (5)
#   36-41  : left eye (6)
#   42-47  : right eye (6)
#   48-59  : outer lip (12)
#   60-67  : inner lip (8)
#
# The mapping below is the widely used community convention — pick one
# representative 106-index for each 68-slot.  Adjust if the model's
# exact definition differs (documented per-slot so it's easy to fix).
# =============================================================================

# LaPa 106-point region boundaries inferred from MMPose's lapa.py swap
# table (pairs like 0↔32, 33↔46, 66↔79, etc.):
#
#   0-32   : face contour (33 pts, 16 is chin center)
#   33-41  : LEFT eyebrow (9 pts; 33-37 pair with right 42-46)
#   42-50  : RIGHT eyebrow (9 pts)
#   51-54  : nose bridge (4 pts, no swap = center line)
#   55-65  : nose bottom / nostrils (11 pts; 55↔65, 60 center)
#   66-74  : LEFT eye (9 pts; 66-70 pair with right 75-79)
#   75-83  : RIGHT eye (9 pts)
#   84-95  : outer lip (12 pts; 84↔90, 87 center top, 93 center bottom)
#   96-105 : inner lip (10 pts; 96↔100, 98 center)
#
# Within each region the best-fit 300W pick is a judgement call on which
# LaPa points are on the silhouette the 300W 68 convention expects.  A
# final tune-up may still be needed after visual inspection of raw 106
# output — enable `debug_dump_rtmface_106` on the node to save it.
_LAPA106_TO_300W68 = np.array([
    # --- Jaw 17 pts (300W 0-16) from LaPa 0-32 ---
    # Every-other sampling gives 17 evenly spaced contour points.
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,

    # --- Left eyebrow 5 pts (300W 17-21) from LaPa 33-41 ---
    # 33-37 = top-edge pair with right brow's 42-46 (swap confirms).
    # Use those 5 consecutive indices.
    33, 34, 35, 36, 37,

    # --- Right eyebrow 5 pts (300W 22-26) from LaPa 42-50 ---
    42, 43, 44, 45, 46,

    # --- Nose bridge 4 pts (300W 27-30) from LaPa 51-54 ---
    # Direct 4→4 mapping on the center line.
    51, 52, 53, 54,

    # --- Nose horizontal / nostrils 5 pts (300W 31-35) from LaPa 55-65 ---
    # LaPa 55↔65 endpoints, 60 center. Evenly sample 5 points.
    55, 57, 60, 63, 65,

    # --- Left eye 6 pts (300W 36-41) from LaPa 66-74 ---
    # 66-70 upper-edge (5 pts pair with right 75-79),
    # 71-74 lower-edge (4 pts).
    # 300W 36-41 order: outer_corner, upper_outer, upper_inner,
    #                   inner_corner, lower_inner, lower_outer
    66, 67, 68, 70, 73, 71,

    # --- Right eye 6 pts (300W 42-47) from LaPa 75-83 ---
    # Mirror of left-eye pattern.
    75, 76, 77, 79, 82, 80,

    # --- Outer lip 12 pts (300W 48-59) from LaPa 84-95 ---
    # LaPa outer lip has 12 pts directly (swap: 84↔90, 87 center top,
    # 91↔95, 93 center bottom).  Direct 12→12 map.
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,

    # --- Inner lip 8 pts (300W 60-67) from LaPa 96-103 ---
    # LaPa inner lip has 10 pts.  Skip 104-105 which appear to be
    # lip-corner commissures not in 300W's inner contour.
    96, 97, 98, 99, 100, 101, 102, 103,
], dtype=np.int64)


# =============================================================================
# Preprocessing / postprocessing (MMPose top-down SimCC convention)
# =============================================================================

_IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMG_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)


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


def _affine_preproc(
    img_np: np.ndarray,        # (H, W, 3) uint8 RGB
    bbox: Tuple[int, int, int, int],
    out_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop + pad to out_size×out_size preserving aspect.

    Returns
    -------
    tensor_chw : (3, out_size, out_size) float32, ImageNet normalised
    inv_xy     : (2,) (scale_x, scale_y) + (2,) (pad_x, pad_y) packed
                 as a 4-tuple [sx, sy, ox, oy] for back-projection.
    """
    import cv2
    x1, y1, x2, y2 = bbox
    crop = img_np[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = out_size / max(h, w)
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to square
    padded = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    pad_x = (out_size - new_w) // 2
    pad_y = (out_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Normalise
    f = padded.astype(np.float32)
    f = (f - _IMG_MEAN) / _IMG_STD
    chw = f.transpose(2, 0, 1)  # (3, out, out)
    # Back-projection parameters
    inv_params = np.array([
        (x2 - x1) / new_w,   # sx: model x -> crop x
        (y2 - y1) / new_h,   # sy
        x1 - pad_x * (x2 - x1) / new_w,  # ox: pixel_x_model_0 in original
        y1 - pad_y * (y2 - y1) / new_h,
    ], dtype=np.float32)
    return chw, inv_params


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray,
                  simcc_split: float = 2.0) -> np.ndarray:
    """Decode RTMPose's SimCC output to (N, K, 3): (x, y, score).

    simcc_x : (N, K, W_x) = (N, K, in_w * simcc_split)
    simcc_y : (N, K, W_y) = (N, K, in_h * simcc_split)

    Returns model-space pixel coords in [0, in_size) and confidence.
    """
    N, K, Wx = simcc_x.shape
    _, _, Wy = simcc_y.shape
    x_idx = simcc_x.argmax(axis=-1)  # (N, K)
    y_idx = simcc_y.argmax(axis=-1)
    x_score = simcc_x.max(axis=-1)
    y_score = simcc_y.max(axis=-1)
    # Min of x/y scores — matches MMPose's decoder convention
    conf = np.minimum(x_score, y_score)
    # Convert bin indices to pixel coords
    x_px = x_idx.astype(np.float32) / float(simcc_split)
    y_px = y_idx.astype(np.float32) / float(simcc_split)
    return np.stack([x_px, y_px, conf], axis=-1)  # (N, K, 3)


def _unmap_to_original(
    kpts_model: np.ndarray,  # (N, K, 3)  model-space x, y, conf
    inv_params_batch: np.ndarray,  # (N, 4)  [sx, sy, ox, oy]
) -> np.ndarray:
    """Back-project model-space keypoints to original image pixel space."""
    out = kpts_model.copy()
    out[..., 0] = kpts_model[..., 0] * inv_params_batch[:, 0:1] + inv_params_batch[:, 2:3]
    out[..., 1] = kpts_model[..., 1] * inv_params_batch[:, 1:2] + inv_params_batch[:, 3:4]
    return out


# =============================================================================
# Main entry point: run RTMPose-Face over the whole video
# =============================================================================

def run_rtmpose_face_video(
    images_np_u8: np.ndarray,              # (B, H, W, 3) uint8 RGB
    persons_coco_body_feet_timeline: list, # per-person list:
                                           #   persons_body_feet[p_idx][t] = (23, 3)
                                           #   or None
    rtmpose_face_dict: dict,               # from LoadRTMPoseFace
    img_h: int,
    img_w: int,
    batch_size: int = 16,
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
    session     = rtmpose_face_dict["session"]
    input_name  = rtmpose_face_dict["input_name"]
    output_names = rtmpose_face_dict["output_names"]
    input_shape = rtmpose_face_dict.get("input_shape", [])

    n_persons = len(persons_coco_body_feet_timeline)
    B = images_np_u8.shape[0]

    # IMPORTANT: MMPose's RTMPose deployment ONNX exports are tuned for
    # single-frame inference (often with a fixed batch dim, sometimes
    # with a partially-baked pairing that breaks slots 1+). We observed
    # repeatedly that calling session.run with a multi-frame batch
    # gives a correct result only on slot 0 (in the batch=1 case) or
    # correct-on-even-garbage-on-odd pattern (in the batch=2 case).
    #
    # Stay on the safe side unconditionally: run ONE frame per
    # session.run call. A single 256×256 SimCC forward is a few
    # milliseconds, so for typical video lengths (100-500 frames) this
    # adds only a fraction of a second.
    if batch_size != 1:
        _logger.info(
            "RTMPose-Face: forcing runtime batch_size=1 (ignoring "
            "requested %d) to avoid a known multi-slot output bug in "
            "MMPose RTMPose-Face ONNX deployment exports.",
            batch_size,
        )
        batch_size = 1

    # Pre-allocate output
    face_kp_68_timeline = [
        [None] * B for _ in range(n_persons)
    ]

    # Build the request list: (p_idx, t, bbox)
    requests = []
    for p_idx in range(n_persons):
        for t in range(B):
            head = persons_coco_body_feet_timeline[p_idx][t]
            if head is None:
                continue
            bbox = _get_face_bbox_from_coco_wb_head(head, img_h, img_w)
            if bbox is None:
                continue
            requests.append((p_idx, t, bbox))

    if not requests:
        return face_kp_68_timeline, 0.0

    import time as _time
    t_start = _time.perf_counter()

    # Optional debug: collect raw 106-pt output from the first frame of
    # each person so we can manually calibrate the LaPa106 → 300W68 map.
    debug_dump = None
    if debug_dump_path is not None:
        debug_dump = {"bboxes": [], "kpts_106": [], "frame_idx": [], "person_idx": []}

    # Process in batches
    for chunk_start in range(0, len(requests), batch_size):
        chunk = requests[chunk_start:chunk_start + batch_size]
        tensors = []
        inv_params_list = []
        for p_idx, t, bbox in chunk:
            chw, inv = _affine_preproc(images_np_u8[t], bbox, out_size=256)
            tensors.append(chw)
            inv_params_list.append(inv)
        batch = np.stack(tensors, axis=0)      # (N, 3, 256, 256)
        inv_params_batch = np.stack(inv_params_list, axis=0)  # (N, 4)

        # ONNX forward — expects (N, 3, 256, 256) float32
        outputs = session.run(output_names, {input_name: batch})
        # RTMPose SDK ONNX outputs: simcc_x (N, K, 512), simcc_y (N, K, 512)
        # (256 * simcc_split=2.0 = 512). If export was different, adjust.
        if len(outputs) >= 2 and outputs[0].ndim == 3:
            simcc_x, simcc_y = outputs[0], outputs[1]
            kpts_model = _decode_simcc(simcc_x, simcc_y, simcc_split=2.0)
        elif len(outputs) == 1 and outputs[0].ndim == 4:
            # Unlikely but handle: single heatmap output (K, H, W) per person
            raise NotImplementedError(
                "RTMPose-Face ONNX returned a single 4D output — expected "
                "SimCC x+y. Please re-export with SimCC head."
            )
        else:
            raise RuntimeError(
                f"Unexpected RTMPose-Face outputs: shapes="
                f"{[o.shape for o in outputs]}"
            )

        kpts_orig = _unmap_to_original(kpts_model, inv_params_batch)

        # Debug dump: save raw 106 points + bbox for the first frame of
        # each person so the user can help calibrate the LaPa→300W map.
        if debug_dump is not None:
            for i, (p_idx, t, bbox) in enumerate(chunk):
                if t == 0 or (p_idx not in debug_dump["person_idx"]):
                    debug_dump["bboxes"].append(np.array(bbox, dtype=np.float32))
                    debug_dump["kpts_106"].append(kpts_orig[i].astype(np.float32))
                    debug_dump["frame_idx"].append(int(t))
                    debug_dump["person_idx"].append(int(p_idx))

        # Map 106 → 68 and scatter back
        K = kpts_orig.shape[1]
        if K != 106:
            _logger.warning(
                "RTMPose-Face returned %d keypoints, expected 106 (Face6). "
                "Keeping raw output and dropping 68-point mapping.", K
            )
            # Fall back: put raw keypoints in; downstream will be misaligned
            for i, (p_idx, t, _bbox) in enumerate(chunk):
                face_kp_68_timeline[p_idx][t] = kpts_orig[i, :min(K, 68)]
        else:
            kpts_68 = kpts_orig[:, _LAPA106_TO_300W68]  # (N, 68, 3)
            for i, (p_idx, t, _bbox) in enumerate(chunk):
                # Post-process: the RTMPose-Face SimCC decoder occasionally
                # outputs a bilateral-mirrored jaw contour on symmetric
                # face points. Detect via jaw endpoint ordering and flip
                # back. This eliminates the 30-85 px per-frame jitter on
                # COCO-WB indices 23..39 we observed when face was not
                # rotating between frames.
                f68 = kpts_68[i].astype(np.float32)
                # 300W jaw: 17 points from image-left ear to image-right ear.
                # In a non-flipped, upright face: jaw[0].x < jaw[16].x.
                if f68[0, 0] > f68[16, 0]:
                    f68[0:17] = f68[0:17][::-1]
                face_kp_68_timeline[p_idx][t] = f68

        if pbar is not None:
            for _ in range(len(chunk)):
                pbar.update(1)

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
            "RTMPose-Face raw 106-pt debug dump saved to %s  (%d entries)",
            debug_dump_path, len(debug_dump["kpts_106"]),
        )

    return face_kp_68_timeline, _time.perf_counter() - t_start
