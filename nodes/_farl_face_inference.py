"""
RetinaFace → FaRL face inference helper.

Replaces the earlier "synthesize 5 landmarks from MHR head keypoints"
approach. That one had to fabricate the 2 mouth corners from canonical
ratios, which only held for perfectly front-facing upright faces — any
head roll / profile / tilt silently corrupted the similarity matrix
FaRL depends on, producing 68-pt output that didn't match the real
face at all.

This version follows pyfacer's own reference pipeline
(``samples/face_alignment.ipynb``): RetinaFace gives us real 5-point
landmarks in the exact order the FaRL aligner expects, so there's zero
convention massaging and zero pose assumption.

Per chunk of frames:

  1. Build (M, 3, H, W) uint8 RGB tensor on device.
  2. RetinaFace detector → ``{image_ids, rects, points, scores}`` for
     every face in the chunk.
  3. Assign each detected face to a tracked person by testing the face
     bbox center against each person's segmentation mask. Sort faces
     by detector score DESC first so when two detections land inside
     the same mask (rare, e.g. tight overlap) the higher-confidence
     one wins. Detections with no matching mask are dropped.
  4. Feed the matched ``(image_ids, points)`` subset back into the
     FaRL aligner — same image tensor, one forward pass.
  5. Scatter the (K, 68, 2) output into per-(person, frame) slots.

FaRL returns landmarks already in ORIGINAL image pixel coords, so no
back-projection is needed.

Notes
-----
* Running RetinaFace per-frame is cheap (MobileNet-0.25, <5 ms/frame).
* We do NOT fall back to the MHR-head synthesis when RetinaFace misses
  a face — the synthesis was actively harmful in most non-frontal
  cases, so a clean "None" for that (person, frame) is strictly better
  than a wrong result.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


# RetinaFace MobileNet-0.25 doesn't benefit from >640 px input for
# typical human-sized faces, and its anchor grid / NMS cost is
# O(H*W). Downscale the detector input (NOT the aligner input — FaRL
# still runs on the full-res frame so 68-pt accuracy is preserved).
_DETECTOR_MAX_SIDE = 640


# --------------------------------------------------------------------------
# Head-keypoint → RetinaFace 5pt synthesis (per-person, no-mask path)
# --------------------------------------------------------------------------
#
# Given per-person COCO-17 head keypoints (nose 0, L_eye 1, R_eye 2,
# L_ear 3, R_ear 4 — subject-anatomy naming), synthesize the 5 RetinaFace
# landmarks FaRL's face_aligner needs. Used when face↔person
# association is already known (each tracked person has its own head
# keypoints from BMP / Sapiens / similar) so we don't need to run
# RetinaFace + mask matching.
#
# Why this used to be unreliable and is now OK: the original failure
# mode was that pyfacer's similarity transform CAN'T mirror, so
# feeding eyes in subject-anatomy order (subject's L_eye at FaRL
# position 0) caused FaRL's solver to compensate via 180° rotation,
# producing a flipped/upside-down aligned face and wrong 68-pt output.
# The fix is the explicit subject→image-space eye swap below.

def coco_head_to_retinaface_5_points(
    head_kp: np.ndarray,            # (5, 2) or (5, 3) — COCO indices 0..4
    conf_thresh: float = 0.1,
) -> Optional[np.ndarray]:
    """Synthesize FaRL face_aligner's 5-point RetinaFace input from
    COCO-17 head keypoints (nose / L_eye / R_eye / L_ear / R_ear).

    pyfacer's FaRL face_aligner expects 5 points in IMAGE-space L/R
    ordering:

        pos 0 : image-LEFT  eye   (= subject's RIGHT eye for a
                                    front-facing subject)
        pos 1 : image-RIGHT eye   (= subject's LEFT eye)
        pos 2 : nose
        pos 3 : image-LEFT  mouth corner
        pos 4 : image-RIGHT mouth corner

    Mouth corners are synthesized from the eye-nose triangle's image-
    plane geometry, using ratios pulled from pyfacer's canonical
    template (``facer/transform.py::_standard_face_pts``):

        eye_distance = 120 px in the 256-px basis
        mouth-below-nose = 74.4 / 120 = 0.62 × eye_distance
        mouth-half-width = 36.0 / 120 = 0.30 × eye_distance

    Returns ``None`` when there aren't enough confident head kps to
    produce a stable alignment (caller should leave that
    (person, frame)'s face slot empty).

    Note: back-view / hallucinated-face filtering happens DOWNSTREAM
    at render time via ``_pose_utils.is_face_visible()`` — that uses
    3D body normal + RTMW face conf + 2D geometry, which is strictly
    more discriminative than what we could do with just the 5 head
    keypoints here. This function only enforces basic sanity (kpts
    present + confidence + non-degenerate eye distance).
    """
    if head_kp.shape[0] < 5:
        return None
    if head_kp.shape[1] >= 3:
        conf = head_kp[:, 2]
    else:
        conf = np.ones(5, dtype=np.float32)

    # Need nose + both eyes to estimate face geometry reliably.
    if conf[0] < conf_thresh or conf[1] < conf_thresh or conf[2] < conf_thresh:
        return None

    nose       = head_kp[0, :2].astype(np.float32)
    subj_l_eye = head_kp[1, :2].astype(np.float32)  # COCO: subject's left eye
    subj_r_eye = head_kp[2, :2].astype(np.float32)  # COCO: subject's right eye

    eye_dist = float(np.linalg.norm(subj_r_eye - subj_l_eye))
    if eye_dist < 5.0:
        return None  # Face too small / eyes degenerate

    eye_mid = (subj_l_eye + subj_r_eye) * 0.5
    down = nose - eye_mid
    down_norm = float(np.linalg.norm(down))
    if down_norm < 1e-3:
        # Nose coincides with eye midpoint — assume upright face.
        down_unit = np.array([0.0, 1.0], dtype=np.float32)
    else:
        down_unit = down / down_norm

    # "Image-right" (face's image-right direction) = 90° rotation of
    # the down axis in image coords (y down convention):
    #   (dx, dy) -> (dy, -dx).  For an upright face (down ≈ (0, 1))
    # this gives (1, 0), i.e. image +x.
    right_unit = np.array([down_unit[1], -down_unit[0]], dtype=np.float32)

    mouth_center = nose + down_unit * eye_dist * 0.62
    half_mouth   = right_unit * eye_dist * 0.30
    img_l_mouth = mouth_center - half_mouth   # image-LEFT mouth corner
    img_r_mouth = mouth_center + half_mouth   # image-RIGHT mouth corner

    # Subject-anatomy → image-space eye remapping. For a front-facing
    # subject: image-LEFT eye = subject's RIGHT eye and vice versa.
    img_l_eye = subj_r_eye
    img_r_eye = subj_l_eye

    return np.stack(
        [img_l_eye, img_r_eye, nose, img_l_mouth, img_r_mouth],
        axis=0,
    ).astype(np.float32)


def run_farl_face_per_person(
    images_np_u8: np.ndarray,                # (B, H, W, 3) uint8 RGB
    head_kp_per_track: List[List[Optional[np.ndarray]]],
                                              # [n_tracks][B] of (5, 3) head
                                              # COCO-17 indices 0..4 or None
    farl_face_dict: dict,                    # from LoadFaRLFace
    img_h: int,
    img_w: int,
    head_conf_thresh: float = 0.1,
    frame_batch_size: int = 32,
    pbar=None,
) -> Tuple[List[List[Optional[np.ndarray]]], float]:
    """Run FaRL face_aligner per-tracked-person using head keypoints
    instead of RetinaFace + mask matching.

    Per-person flow:
        for each (person, frame) where head 5pt is available:
            synthesize RetinaFace 5pt via coco_head_to_retinaface_5_points
            stash in a flat list with (frame_local, track_idx) metadata
        batch all of them by frame into a single FaRL forward
        scatter (K, 68, 2) output back into face_kp_68_timeline[t][p]

    Returns ``face_kp_68_timeline`` shaped ``[n_tracks][B]`` of
    (68, 3) arrays (with confidence column = 1.0) or None.

    Use this in pipelines where face-to-person association is already
    known (BMP, Sapiens, etc. each output per-tracked-person head
    keypoints). For pipelines without per-person head info, see
    ``run_farl_face_video`` which uses RetinaFace + mask matching.
    """
    aligner = farl_face_dict["aligner"]
    device  = farl_face_dict["device"]

    n_tracks = len(head_kp_per_track)
    B = int(images_np_u8.shape[0])
    face_kp_68_timeline: List[List[Optional[np.ndarray]]] = [
        [None] * B for _ in range(n_tracks)
    ]

    if n_tracks == 0 or B == 0:
        return face_kp_68_timeline, 0.0

    # Group requests by frame so we can run FaRL in chunked frames.
    # Each entry: (frame_local, track_idx, retinaface_5pt)
    per_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for p_idx in range(n_tracks):
        for t in range(B):
            head = head_kp_per_track[p_idx][t]
            if head is None:
                continue
            pts5 = coco_head_to_retinaface_5_points(
                head, conf_thresh=head_conf_thresh,
            )
            if pts5 is None:
                continue
            per_frame.setdefault(t, []).append((p_idx, pts5))

    if not per_frame:
        return face_kp_68_timeline, 0.0

    frame_idx_sorted = sorted(per_frame.keys())
    t_start = time.perf_counter()

    for chunk_start in range(0, len(frame_idx_sorted), frame_batch_size):
        chunk_frames = frame_idx_sorted[
            chunk_start: chunk_start + frame_batch_size
        ]

        chunk_imgs = images_np_u8[chunk_frames]
        images_bchw = (
            torch.from_numpy(chunk_imgs)
                  .permute(0, 3, 1, 2)
                  .contiguous()
                  .to(device)
        )

        pts5_list, image_ids_list, meta = [], [], []
        for local_b, t_global in enumerate(chunk_frames):
            for p_idx, pts5 in per_frame[t_global]:
                pts5_list.append(pts5)
                image_ids_list.append(local_b)
                meta.append((t_global, p_idx))

        if not pts5_list:
            if pbar is not None:
                pbar.update(len(chunk_frames))
            continue

        points_t    = torch.tensor(
            np.stack(pts5_list, axis=0), dtype=torch.float32, device=device,
        )
        image_ids_t = torch.tensor(image_ids_list, dtype=torch.long, device=device)

        try:
            with torch.inference_mode():
                faces_out = aligner(images_bchw, {
                    "image_ids": image_ids_t,
                    "points":    points_t,
                })
        except Exception as e:
            _logger.error(
                "FaRL aligner failed on frames %d..%d (%d faces): %s",
                chunk_frames[0], chunk_frames[-1], len(pts5_list), e,
            )
            if pbar is not None:
                pbar.update(len(chunk_frames))
            continue

        alignment = (
            faces_out["alignment"].detach().cpu().numpy().astype(np.float32)
        )
        if alignment.ndim != 3 or alignment.shape[1] != 68:
            _logger.warning(
                "FaRL returned alignment shape %s on chunk starting at "
                "frame %d — expected (N, 68, *). Skipping chunk.",
                alignment.shape, chunk_frames[0],
            )
            if pbar is not None:
                pbar.update(len(chunk_frames))
            continue

        if alignment.shape[2] == 2:
            conf_col = np.ones(
                (alignment.shape[0], alignment.shape[1], 1), dtype=np.float32,
            )
            kpts_xyc = np.concatenate([alignment, conf_col], axis=-1)
        else:
            kpts_xyc = alignment

        for i, (t_global, p_idx) in enumerate(meta):
            face_kp_68_timeline[p_idx][t_global] = kpts_xyc[i]

        if pbar is not None:
            pbar.update(len(chunk_frames))

    return face_kp_68_timeline, time.perf_counter() - t_start


def _match_faces_to_persons(
    image_ids: np.ndarray,       # (K,) local batch index
    rects:     np.ndarray,       # (K, 4) xyxy in original image pixels
    scores:    np.ndarray,       # (K,)
    chunk_frames: List[int],     # maps local_b -> global frame index
    masks_np:  np.ndarray,       # (B, n_persons, H, W) bool
    n_persons: int,
    img_h: int,
    img_w: int,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Greedy face→person assignment by mask membership.

    For each detection (sorted by score DESC), test the bbox center
    against every unassigned person's mask on that frame. First mask
    that contains the point wins, and that (frame, person) slot is
    locked so subsequent lower-confidence faces can't steal it.

    Returns
    -------
    matched_image_ids : list[int]
        Local batch indices for faces that got assigned.
    matched_det_idx : list[int]
        Index into the original (K,) arrays so the caller can
        rebuild the ``points`` subset.
    matched_meta : list[(t_global, p_idx)]
        One-to-one with the other two lists.
    """
    K = len(image_ids)
    if K == 0:
        return [], [], []

    order = np.argsort(-scores)  # DESC by score

    matched_image_ids: List[int] = []
    matched_det_idx:   List[int] = []
    matched_meta: List[Tuple[int, int]] = []
    assigned: set = set()  # (t_global, p_idx) already filled

    for k in order:
        local_b = int(image_ids[k])
        if local_b < 0 or local_b >= len(chunk_frames):
            continue
        t_global = chunk_frames[local_b]

        x1, y1, x2, y2 = rects[k]
        cx = 0.5 * (float(x1) + float(x2))
        cy = 0.5 * (float(y1) + float(y2))
        ix = int(round(cx))
        iy = int(round(cy))
        if not (0 <= iy < img_h and 0 <= ix < img_w):
            continue

        for p_idx in range(n_persons):
            if (t_global, p_idx) in assigned:
                continue
            if bool(masks_np[t_global, p_idx, iy, ix]):
                assigned.add((t_global, p_idx))
                matched_image_ids.append(local_b)
                matched_det_idx.append(int(k))
                matched_meta.append((t_global, p_idx))
                break

    return matched_image_ids, matched_det_idx, matched_meta


def run_farl_face_video(
    images_np_u8: np.ndarray,      # (B, H, W, 3) uint8 RGB
    masks_np: np.ndarray,          # (B, n_persons, H, W) bool
    farl_face_dict: dict,          # from LoadFaRLFace (detector + aligner)
    n_persons: int,
    img_h: int,
    img_w: int,
    frame_batch_size: int = 32,
    pbar=None,
):
    """Run RetinaFace + FaRL across the full video.

    Parameters
    ----------
    images_np_u8 : (B, H, W, 3) uint8 RGB
        pyfacer's own reader yields RGB (``facer/io.py::read_hwc``), so
        we keep RGB end-to-end. Matches the detector/aligner
        expectations without extra channel swaps.
    masks_np : (B, n_persons, H, W) bool
        Used only for face→person assignment via mask membership.
    farl_face_dict : dict
        Must carry ``"detector"`` (RetinaFace), ``"aligner"`` (FaRL),
        and ``"device"``.
    n_persons : int
        Number of tracked persons in the ``masks_np`` axis 1.
    frame_batch_size : int
        Per-chunk size for both the detector and aligner forward passes.
        Larger = faster but more VRAM. 32 is safe for 512×960 frames
        on a 24 GB GPU.

    Returns
    -------
    face_kp_68_timeline : list[list[np.ndarray | None]]
        ``face_kp_68_timeline[p_idx][t]`` is (68, 3) with (x, y, 1.0)
        or ``None`` when no face was detected / matched for that slot.
    time_s : float
        Wall time for the whole pass (detector + matching + aligner).
    """
    detector = farl_face_dict["detector"]
    aligner  = farl_face_dict["aligner"]
    device   = farl_face_dict["device"]

    B = images_np_u8.shape[0]
    face_kp_68_timeline: List[List[np.ndarray]] = [
        [None] * B for _ in range(n_persons)
    ]

    if B == 0 or n_persons == 0:
        return face_kp_68_timeline, 0.0

    # --- Pre-filter: drop frames that have no tracked persons ----------
    # RetinaFace + FaRL are the biggest per-frame cost in this node.
    # Running them on frames where every mask is empty was wasted work
    # (the earlier pipeline already filtered by MHR head-kp availability,
    # which effectively gated on mask presence upstream).
    if masks_np.size > 0:
        active_any = np.any(masks_np.reshape(B, -1), axis=1)
        active_frames_all = np.where(active_any)[0].tolist()
    else:
        active_frames_all = []

    if pbar is not None and len(active_frames_all) < B:
        # Advance the progress bar for the frames we're skipping so the
        # caller's `ProgressBar(B * 2)` budget stays accurate.
        pbar.update(B - len(active_frames_all))

    if not active_frames_all:
        return face_kp_68_timeline, 0.0

    # --- Detector downscale factor -------------------------------------
    # RetinaFace MobileNet-0.25's anchor + NMS cost is O(H*W). A 640-
    # side cap drops a 1080p frame's detector cost by ~9× and a 4K
    # frame's by ~36× with no meaningful recall hit for human-sized
    # faces. The FaRL aligner still runs on the FULL-resolution frame
    # (landmark accuracy depends on real pixel fidelity), so this
    # optimization is detector-only.
    det_scale = min(
        float(_DETECTOR_MAX_SIDE) / float(max(img_h, img_w)), 1.0,
    )
    det_h = max(1, int(round(img_h * det_scale)))
    det_w = max(1, int(round(img_w * det_scale)))

    _logger.info(
        "FaRL: processing %d/%d active frames; detector input "
        "%dx%d (scale %.3f), aligner input %dx%d",
        len(active_frames_all), B, det_h, det_w, det_scale, img_h, img_w,
    )

    t_start = time.perf_counter()

    for chunk_start in range(0, len(active_frames_all), frame_batch_size):
        chunk_frames = active_frames_all[chunk_start:chunk_start + frame_batch_size]
        M = len(chunk_frames)

        # (M, H, W, 3) uint8 RGB -> (M, 3, H, W) uint8 RGB on device.
        # The aligner wants uint8 at the ORIGINAL resolution; the
        # detector gets a downscaled float copy below (when applicable).
        chunk_imgs = images_np_u8[chunk_frames]
        images_bchw = torch.from_numpy(chunk_imgs).permute(0, 3, 1, 2).contiguous()
        images_bchw = images_bchw.to(device)

        # --- (Optional) downscale for detector -------------------------------
        if det_scale < 1.0:
            # F.interpolate doesn't accept uint8; go through float once.
            # The detector's batch_detect does `images.float()` anyway, so
            # feeding float here is a no-op on its side.
            det_input = F.interpolate(
                images_bchw.float(),
                size=(det_h, det_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            det_input = images_bchw

        # --- RetinaFace forward ----------------------------------------------
        try:
            with torch.inference_mode():
                faces_det: Dict[str, torch.Tensor] = detector(det_input)
        except Exception as e:
            _logger.error(
                "RetinaFace failed on frames %d..%d: %s",
                chunk_frames[0], chunk_frames[-1], e,
            )
            if pbar is not None:
                pbar.update(M)
            continue

        # Empty result → no faces in this chunk, move on.
        if (
            "image_ids" not in faces_det
            or faces_det["image_ids"].numel() == 0
        ):
            if pbar is not None:
                pbar.update(M)
            continue

        all_image_ids = faces_det["image_ids"].detach().cpu().numpy().astype(np.int64)
        all_rects     = faces_det["rects"].detach().cpu().numpy().astype(np.float32)
        all_points    = faces_det["points"].detach().cpu().numpy().astype(np.float32)
        all_scores    = faces_det["scores"].detach().cpu().numpy().astype(np.float32)
        if det_scale < 1.0:
            inv = 1.0 / det_scale
            all_rects  = all_rects * inv
            all_points = all_points * inv

        # --- Match detections to persons ------------------------------------
        m_image_ids, m_det_idx, m_meta = _match_faces_to_persons(
            image_ids=all_image_ids,
            rects=all_rects,
            scores=all_scores,
            chunk_frames=chunk_frames,
            masks_np=masks_np,
            n_persons=n_persons,
            img_h=img_h,
            img_w=img_w,
        )

        if not m_image_ids:
            if pbar is not None:
                pbar.update(M)
            continue

        # --- FaRL aligner on matched subset ---------------------------------
        matched_points = all_points[np.asarray(m_det_idx, dtype=np.int64)]  # (K', 5, 2)
        aligner_in = {
            "image_ids": torch.as_tensor(
                m_image_ids, dtype=torch.long, device=device,
            ),
            "points": torch.as_tensor(
                matched_points, dtype=torch.float32, device=device,
            ),
        }

        try:
            with torch.inference_mode():
                aligner_out = aligner(images_bchw, aligner_in)
        except Exception as e:
            _logger.error(
                "FaRL aligner failed on frames %d..%d (%d faces): %s",
                chunk_frames[0], chunk_frames[-1], len(m_image_ids), e,
            )
            if pbar is not None:
                pbar.update(M)
            continue

        alignment = (
            aligner_out["alignment"].detach().cpu().numpy().astype(np.float32)
        )
        if alignment.ndim != 3 or alignment.shape[1] != 68:
            _logger.warning(
                "FaRL returned alignment shape %s on chunk starting at "
                "frame %d — expected (N, 68, *). Skipping chunk.",
                alignment.shape, chunk_frames[0],
            )
            if pbar is not None:
                pbar.update(M)
            continue

        # Add a dummy confidence col if pyfacer returned (N, 68, 2). The
        # downstream COCO-WB packing expects (x, y, c).
        if alignment.shape[2] == 2:
            conf_col = np.ones(
                (alignment.shape[0], alignment.shape[1], 1), dtype=np.float32,
            )
            kpts_xyz = np.concatenate([alignment, conf_col], axis=-1)
        else:
            kpts_xyz = alignment

        for i, (t_global, p_idx) in enumerate(m_meta):
            face_kp_68_timeline[p_idx][t_global] = kpts_xyz[i]

        if pbar is not None:
            pbar.update(M)

    return face_kp_68_timeline, time.perf_counter() - t_start
