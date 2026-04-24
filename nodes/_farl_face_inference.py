"""
FaRL face inference helper.

Given per-person face bboxes derived from Fast SAM 3D Body's head
keypoints, run FaRL (via pyfacer) to get 68-point face landmarks in
the 300W / iBUG convention — exactly what COCO-WholeBody indices
23..90 expect.

FaRL returns landmarks already in ORIGINAL image pixel coords, so no
back-projection is needed.

Pipeline per frame-chunk:
  1. collect all (frame_idx, person_idx, bbox) slots
  2. build a (M, 3, H, W) uint8 image tensor for M frames in the chunk
  3. build rects (N, 4) + image_ids (N,) mapping each face to its frame
  4. one call to ``face_aligner(images, faces)`` processes them all
  5. scatter the (N, 68, 2) output back into per-(person, frame) slots
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch

_logger = logging.getLogger(__name__)


def _get_face_bbox_from_coco_wb_head(
    coco_wb_body_feet: np.ndarray,  # (23, 3) or (23, 2)
    img_h: int,
    img_w: int,
    expand_ratio: float = 1.8,
) -> Optional[Tuple[int, int, int, int]]:
    """Derive a square face bbox from the 5 head keypoints (nose + eyes
    + ears, COCO-WB 0..4).

    Returns ``None`` when head keypoints are too sparse to locate a face.
    """
    head = coco_wb_body_feet[:5]
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
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


def run_farl_face_video(
    images_np_u8: np.ndarray,              # (B, H, W, 3) uint8 RGB
    persons_coco_body_feet_timeline: list, # per-person list:
                                           #   persons_body_feet[p_idx][t] = (23, 3) or None
    farl_face_dict: dict,                  # from LoadFaRLFace
    img_h: int,
    img_w: int,
    frame_batch_size: int = 32,
    pbar=None,
):
    """Run FaRL across the full video.

    Returns
    -------
    face_kp_68_timeline : list[list[np.ndarray | None]]
        face_kp_68_timeline[p_idx][t] = (68, 3) with (x, y, 1.0)
        or None when no valid face bbox.
    time_s : float
    """
    aligner = farl_face_dict["aligner"]
    device  = farl_face_dict["device"]

    n_persons = len(persons_coco_body_feet_timeline)
    B = images_np_u8.shape[0]

    # Pre-allocate output
    face_kp_68_timeline = [[None] * B for _ in range(n_persons)]

    # Build request list grouped by frame so we can chunk across frames
    # for batched GPU inference.
    per_frame = {}  # t -> list[(p_idx, bbox)]
    for p_idx in range(n_persons):
        for t in range(B):
            head = persons_coco_body_feet_timeline[p_idx][t]
            if head is None:
                continue
            bbox = _get_face_bbox_from_coco_wb_head(head, img_h, img_w)
            if bbox is None:
                continue
            per_frame.setdefault(t, []).append((p_idx, bbox))

    if not per_frame:
        return face_kp_68_timeline, 0.0

    frame_idx_sorted = sorted(per_frame.keys())

    t_start = time.perf_counter()

    for chunk_start in range(0, len(frame_idx_sorted), frame_batch_size):
        chunk_frames = frame_idx_sorted[chunk_start:chunk_start + frame_batch_size]

        # Assemble image batch (M, 3, H, W) uint8 on device
        chunk_imgs = images_np_u8[chunk_frames]  # (M, H, W, 3) uint8 RGB
        images_bchw = torch.from_numpy(chunk_imgs).permute(0, 3, 1, 2).contiguous()
        images_bchw = images_bchw.to(device)

        # Assemble rects + image_ids + metadata (person & global frame idx)
        rects_list = []
        image_ids_list = []
        face_meta = []  # aligned with rects: (t_global, p_idx)
        for local_b, t_global in enumerate(chunk_frames):
            for p_idx, bbox in per_frame[t_global]:
                rects_list.append(list(bbox))
                image_ids_list.append(local_b)
                face_meta.append((t_global, p_idx))

        if not rects_list:
            # Shouldn't happen — we filtered to frames that have faces.
            if pbar is not None:
                pbar.update(len(chunk_frames))
            continue

        rects_t     = torch.tensor(rects_list,     dtype=torch.float32, device=device)
        image_ids_t = torch.tensor(image_ids_list, dtype=torch.long,    device=device)

        faces_in = {
            "rects":     rects_t,
            "image_ids": image_ids_t,
        }

        try:
            with torch.inference_mode():
                faces_out = aligner(images_bchw, faces_in)
        except Exception as e:
            _logger.error(
                "FaRL face_aligner failed on frames %d..%d (%d faces): %s",
                chunk_frames[0], chunk_frames[-1], len(rects_list), e,
            )
            if pbar is not None:
                pbar.update(len(chunk_frames))
            continue

        # FaRL returns 'alignment' shape (N, 68, 2) in ORIGINAL image pixel
        # coords. Some pyfacer versions may instead return (N, 68, 3) with
        # a confidence column — handle both.
        alignment = faces_out["alignment"].detach().cpu().numpy().astype(np.float32)
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
            # Add a dummy confidence column of 1.0 to match the (x, y, c)
            # format downstream COCO-WB 133-pt packing expects.
            conf_col = np.ones(
                (alignment.shape[0], alignment.shape[1], 1), dtype=np.float32,
            )
            kpts_xyz = np.concatenate([alignment, conf_col], axis=-1)
        else:
            kpts_xyz = alignment  # already (N, 68, 3)

        for i, (t_global, p_idx) in enumerate(face_meta):
            face_kp_68_timeline[p_idx][t_global] = kpts_xyz[i]

        if pbar is not None:
            pbar.update(len(chunk_frames))

    return face_kp_68_timeline, time.perf_counter() - t_start
