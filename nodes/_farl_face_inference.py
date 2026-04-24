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
from typing import Dict, List, Tuple

import numpy as np
import torch

_logger = logging.getLogger(__name__)


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

    t_start = time.perf_counter()

    for chunk_start in range(0, B, frame_batch_size):
        chunk_end = min(chunk_start + frame_batch_size, B)
        chunk_frames = list(range(chunk_start, chunk_end))
        M = len(chunk_frames)

        # (M, H, W, 3) uint8 RGB -> (M, 3, H, W) uint8 RGB on device.
        # pyfacer's detector does ``images.float()`` internally and
        # subtracts the [104, 117, 123] mean; its own samples feed RGB
        # (read via PIL → RGB), so we follow the same convention.
        # ``images.clone()`` inside RetinaFaceDetector.forward keeps
        # subsequent aligner calls from seeing a mutated tensor, but we
        # also avoid any in-place surprises by keeping our reference
        # separate.
        chunk_imgs = images_np_u8[chunk_frames]
        images_bchw = torch.from_numpy(chunk_imgs).permute(0, 3, 1, 2).contiguous()
        images_bchw = images_bchw.to(device)

        # --- RetinaFace -----------------------------------------------------
        try:
            with torch.inference_mode():
                faces_det: Dict[str, torch.Tensor] = detector(images_bchw)
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

        # --- Scatter to per-person timeline ---------------------------------
        for i, (t_global, p_idx) in enumerate(m_meta):
            face_kp_68_timeline[p_idx][t_global] = kpts_xyz[i]

        if pbar is not None:
            pbar.update(M)

    return face_kp_68_timeline, time.perf_counter() - t_start
