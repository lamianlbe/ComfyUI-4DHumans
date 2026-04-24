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


# =============================================================================
# LaPa-106 side prior for SimCC decoding
#
# Diagnosis: on nearly identical input frames (1-2 px shift), ONNX outputs
# jaw[0]/jaw[32] (and other symmetric landmarks) at the bilaterally MIRRORED
# position on roughly half the frames. The raw ONNX output is deterministic
# on identical input (verified via diagnose_rtmpose_face_determinism.py
# scenario 1), so this is the model itself producing a bimodal SimCC
# heatmap — one peak at the true location and a ghost at its mirror — for
# symmetric landmarks. Numeric noise from a 1-2 px input shift tips argmax
# from the true peak to the ghost.
#
# Fix: each LaPa landmark has a well-defined LEFT/CENTER/RIGHT position on
# the face in a frontal crop. Since our face bbox is square and centered
# on the head keypoints, the face center sits at x=128 in the 256×256
# model input, i.e. SimCC bin 256 out of 512. For LEFT landmarks we
# argmax over simcc_x[:, :256]; for RIGHT we argmax over simcc_x[:, 256:];
# for CENTER we keep the global argmax. The mirror peak is thus
# unreachable for side-constrained landmarks.
#
# Classification derived from the actual predicted positions of a
# known-good frame (f0 in rtmface_106_debug.npz) — see the debug analysis
# in commit message for the decision per index. Near-center landmarks
# (|x - face_cx| <= 3 px) are marked CENTER.
# =============================================================================

# -1 = LEFT half, 0 = center (no restriction), +1 = RIGHT half
_LAPA_SIDE = np.zeros(106, dtype=np.int8)

# Jaw contour: 0..15 left, 16 chin-bottom (center), 17..32 right
_LAPA_SIDE[0:16]  = -1
_LAPA_SIDE[16]    = 0
_LAPA_SIDE[17:33] = +1

# Eyebrows: left 33..41, right 42..50
_LAPA_SIDE[33:42] = -1
_LAPA_SIDE[42:51] = +1

# Nose bridge 51..54 sits on the vertical centerline
_LAPA_SIDE[51:55] = 0

# Nose bottom 55..65: alternating L/R layout around centre idx 60 and 61
_LAPA_SIDE[55] = -1; _LAPA_SIDE[56] = +1; _LAPA_SIDE[57] = -1
_LAPA_SIDE[58] = +1; _LAPA_SIDE[59] = -1; _LAPA_SIDE[60] = 0
_LAPA_SIDE[61] = 0   # within ±3 of centre on f0, keep unrestricted
_LAPA_SIDE[62] = -1; _LAPA_SIDE[63] = +1; _LAPA_SIDE[64] = -1
_LAPA_SIDE[65] = +1

# Left eye 66..74
_LAPA_SIDE[66:75] = -1
# Right eye 75..83
_LAPA_SIDE[75:84] = +1

# Outer lip 84..95
_LAPA_SIDE[84:87] = -1  # top-left → top-centre-left
_LAPA_SIDE[87]    = 0   # top centre
_LAPA_SIDE[88:91] = +1  # top-centre-right → top-right
_LAPA_SIDE[91:93] = +1  # bottom-right
_LAPA_SIDE[93]    = 0   # bottom centre
_LAPA_SIDE[94:96] = -1  # bottom-left

# Inner lip 96..103 + eye pupils 104..105
_LAPA_SIDE[96:98]   = -1  # 96, 97 inner top-left
_LAPA_SIDE[98]      = 0
_LAPA_SIDE[99:101]  = +1  # 99, 100 inner top-right
_LAPA_SIDE[101]     = +1  # NOTE: LaPa pair [101, 103] has 101 on RIGHT
_LAPA_SIDE[102]     = 0   # bottom centre
_LAPA_SIDE[103]     = -1  # inner bottom-left
_LAPA_SIDE[104]     = -1  # left pupil
_LAPA_SIDE[105]     = +1  # right pupil


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
    # IMPORTANT: np.transpose returns a NON-contiguous view. onnxruntime
    # usually handles that fine, but we have seen MMPose RTMPose face
    # ONNX return garbled output on non-first frames unless the input
    # is C-contiguous. Force a copy so every frame's input is clean.
    chw = np.ascontiguousarray(f.transpose(2, 0, 1))  # (3, out, out)
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
    """Decode RTMPose's SimCC output to (N, K, 3): (x, y, score) using the
    per-landmark LEFT / CENTER / RIGHT side prior in ``_LAPA_SIDE``.

    simcc_x : (N, K, W_x) = (N, K, in_w * simcc_split)   -- 512 bins for 256 px
    simcc_y : (N, K, W_y) = (N, K, in_h * simcc_split)

    For LEFT/RIGHT landmarks we argmax over the correct half of simcc_x
    only, which makes the mirror peak unreachable. Y axis is never
    constrained — bimodal ghosts are horizontal, not vertical.

    Returns model-space pixel coords in [0, in_size) and confidence.
    """
    N, K, Wx = simcc_x.shape
    _, _, Wy = simcc_y.shape
    cx_sim = Wx // 2  # 256 for Wx=512 (face horizontal centre in SimCC space)

    # Default: global argmax for all keypoints (fallback / CENTER landmarks)
    x_idx = simcc_x.argmax(axis=-1).astype(np.int64)  # (N, K)
    x_score = simcc_x.max(axis=-1).astype(simcc_x.dtype)

    if K == _LAPA_SIDE.size:
        # Override LEFT landmarks: argmax restricted to [0, cx_sim)
        left_k = np.where(_LAPA_SIDE == -1)[0]
        if len(left_k) > 0:
            left_slice = simcc_x[:, left_k, :cx_sim]  # (N, K_L, cx_sim)
            x_idx[:, left_k] = left_slice.argmax(axis=-1)
            x_score[:, left_k] = left_slice.max(axis=-1)
        # Override RIGHT landmarks: argmax restricted to [cx_sim, Wx)
        right_k = np.where(_LAPA_SIDE == +1)[0]
        if len(right_k) > 0:
            right_slice = simcc_x[:, right_k, cx_sim:]  # (N, K_R, Wx-cx_sim)
            x_idx[:, right_k] = right_slice.argmax(axis=-1) + cx_sim
            x_score[:, right_k] = right_slice.max(axis=-1)
        # CENTER landmarks (_LAPA_SIDE == 0) keep the global argmax above.

    y_idx = simcc_y.argmax(axis=-1)
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
    # Also dump the actual 256×256 tensor fed to ONNX (reconstructed as
    # uint8 RGB) so we can visually verify the preprocessing.
    debug_dump = None
    if debug_dump_path is not None:
        debug_dump = {
            "bboxes": [], "kpts_106": [],
            "frame_idx": [], "person_idx": [],
            "preproc_u8": [],  # (256, 256, 3) uint8, what ONNX sees
        }

    # Process in batches
    for chunk_start in range(0, len(requests), batch_size):
        chunk = requests[chunk_start:chunk_start + batch_size]
        tensors = []
        inv_params_list = []
        for p_idx, t, bbox in chunk:
            chw, inv = _affine_preproc(images_np_u8[t], bbox, out_size=256)
            tensors.append(chw)
            inv_params_list.append(inv)
        batch = np.ascontiguousarray(np.stack(tensors, axis=0))   # (N, 3, 256, 256)
        inv_params_batch = np.stack(inv_params_list, axis=0)       # (N, 4)

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

        # Debug dump: save raw 106 points + bbox + preproc image for
        # EVERY (person, frame) pair so we can diagnose frame-to-frame
        # stability issues.
        if debug_dump is not None:
            for i, (p_idx, t, bbox) in enumerate(chunk):
                debug_dump["bboxes"].append(np.array(bbox, dtype=np.float32))
                debug_dump["kpts_106"].append(kpts_orig[i].astype(np.float32))
                debug_dump["frame_idx"].append(int(t))
                debug_dump["person_idx"].append(int(p_idx))
                # Reverse the ImageNet normalisation so what we save is
                # literally "the pixels ONNX got to see".
                chw = batch[i]  # (3, 256, 256) float32 normalised
                hwc = chw.transpose(1, 2, 0)
                rgb = (hwc * _IMG_STD + _IMG_MEAN)
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                debug_dump["preproc_u8"].append(rgb)

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
                # The side-prior SimCC decoder (_decode_simcc) already
                # prevents the bilateral-mirror ambiguity per landmark,
                # so no post-flip correction is needed here.
                face_kp_68_timeline[p_idx][t] = kpts_68[i].astype(np.float32)

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
            preproc_u8=np.stack(debug_dump["preproc_u8"]),  # (N, 256, 256, 3)
        )
        _logger.info(
            "RTMPose-Face raw 106-pt debug dump saved to %s  (%d entries, "
            "with preproc_u8 tensor)",
            debug_dump_path, len(debug_dump["kpts_106"]),
        )

    return face_kp_68_timeline, _time.perf_counter() - t_start
