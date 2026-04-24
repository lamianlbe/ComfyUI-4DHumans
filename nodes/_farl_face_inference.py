"""
FaRL face inference helper.

Given per-person head keypoints from Fast SAM 3D Body (COCO-WB
indices 0..4: nose, L_eye, R_eye, L_ear, R_ear), run FaRL (via
pyfacer) to get 68-point face landmarks in the 300W / iBUG
convention — exactly what COCO-WholeBody indices 23..90 expect.

IMPORTANT: pyfacer's FaRLFaceAlignment is a REFINEMENT model, not a
bbox-based detector. It requires ``faces['points']`` with shape
(N, 5, 2) in RetinaFace order
``[L_eye, R_eye, nose, L_mouth, R_mouth]`` (where L/R are IMAGE
left/right, not subject's own); it uses them to compute a similarity
alignment matrix to the canonical 448×448 template before running
the ViT. Our upstream only gives us nose + eyes + ears, so we
synthesize the 2 mouth corners from facial geometry (see
``_mhr_head_to_retinaface_5_points`` — ratios read off the canonical
template in ``facer/transform.py::_standard_face_pts``:
nose-below-eyes = 0.50*eye_dist, mouth-below-nose = 0.62*eye_dist,
mouth-half-width = 0.30*eye_dist).

FaRL returns landmarks already in ORIGINAL image pixel coords, so no
back-projection is needed.

Pipeline per frame-chunk:
  1. collect all (frame_idx, person_idx, 5-point array) slots
  2. build (M, 3, H, W) uint8 image tensor for M frames in the chunk
  3. build points (N, 5, 2) + image_ids (N,) mapping each face to its frame
  4. one call to ``face_aligner(images, faces)`` processes them all
  5. scatter the (N, 68, 2) output back into per-(person, frame) slots
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch

_logger = logging.getLogger(__name__)


def _mhr_head_to_retinaface_5_points(
    coco_wb_body_feet: np.ndarray,  # (23, 3) or (23, 2)
) -> Optional[np.ndarray]:
    """Build a (5, 2) RetinaFace-order landmark array from the COCO-WB
    head keypoints we have from MHR (nose, L_eye, R_eye, L_ear, R_ear).

    pyfacer's FaRL face_aligner needs 5 landmarks in the order
    ``[L_eye, R_eye, nose, L_mouth, R_mouth]`` to compute its
    similarity alignment matrix. MHR gives us 3 real points (nose +
    both eyes) and two that aren't useful (ears). We synthesize the
    mouth corners from facial geometry so the alignment transform
    gets real rotation + scale info from the actual eye/nose triangle
    rather than canonical guesses inside a bbox.

    Ratios taken from pyfacer's canonical template (in
    ``facer/transform.py::_standard_face_pts``), which in 256×256
    pixel basis is:
        L_eye (196, 226), R_eye (316, 226)   -> eye_dist = 120
        nose  (256, 286)                     -> nose below eyes by 60 px
        L_mouth (220, 360.4), R_mouth (292, 360.4)
                                              -> mouth below nose by 74.4 px
                                              -> mouth half-width   = 36 px
    Relative to ``eye_dist``:
        mouth-below-nose   = 74.4 / 120 = 0.62
        mouth-half-width   = 36.0 / 120 = 0.30
    The mouth-center is placed along the face-midline (extension of
    the eye-midpoint → nose vector) so the synthesis respects real
    head tilt/roll, not just image-axis assumptions.

    Returns ``None`` when we don't have enough valid head kps to
    produce a stable alignment.
    """
    head = coco_wb_body_feet[:5]
    if head.shape[1] >= 3:
        conf = head[:, 2]
    else:
        conf = np.ones(5, dtype=np.float32)

    # We need nose + both eyes to estimate face geometry reliably.
    if conf[0] < 0.1 or conf[1] < 0.1 or conf[2] < 0.1:
        return None

    nose  = head[0, :2].astype(np.float32)
    l_eye = head[1, :2].astype(np.float32)
    r_eye = head[2, :2].astype(np.float32)

    eye_dist = float(np.linalg.norm(r_eye - l_eye))
    if eye_dist < 5.0:
        # Face too small / eye points degenerate — skip.
        return None

    eye_mid = (l_eye + r_eye) * 0.5
    down    = nose - eye_mid
    down_norm = float(np.linalg.norm(down))
    if down_norm < 1e-3:
        # Nose coincides with eye midpoint — use a default downward
        # axis aligned with image y (upright face assumption).
        down_unit = np.array([0.0, 1.0], dtype=np.float32)
    else:
        down_unit = down / down_norm

    # Face-right is a 90° CW rotation of the down axis in image coords
    # (y increases downward). (dx, dy) -> (dy, -dx).
    right_unit = np.array([down_unit[1], -down_unit[0]], dtype=np.float32)

    mouth_center = nose + down_unit * eye_dist * 0.62
    half_mouth   = right_unit * eye_dist * 0.30
    l_mouth = mouth_center - half_mouth   # image-left mouth corner
    r_mouth = mouth_center + half_mouth   # image-right mouth corner

    return np.stack(
        [l_eye, r_eye, nose, l_mouth, r_mouth], axis=0,
    ).astype(np.float32)


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
        or None when no valid head keypoints for alignment.
    time_s : float
    """
    aligner = farl_face_dict["aligner"]
    device  = farl_face_dict["device"]

    n_persons = len(persons_coco_body_feet_timeline)
    B = images_np_u8.shape[0]

    # Pre-allocate output
    face_kp_68_timeline = [[None] * B for _ in range(n_persons)]

    # Build request list grouped by frame so we can chunk across frames
    # for batched GPU inference. Each request carries a (5, 2) RetinaFace
    # order landmark array used by FaRL to compute the alignment matrix.
    per_frame = {}  # t -> list[(p_idx, pts_5)]
    for p_idx in range(n_persons):
        for t in range(B):
            head = persons_coco_body_feet_timeline[p_idx][t]
            if head is None:
                continue
            pts_5 = _mhr_head_to_retinaface_5_points(head)
            if pts_5 is None:
                continue
            per_frame.setdefault(t, []).append((p_idx, pts_5))

    if not per_frame:
        _logger.warning(
            "FaRL: no valid face alignment input on any frame — "
            "all head keypoints have confidence < 0.1 or are degenerate."
        )
        return face_kp_68_timeline, 0.0

    frame_idx_sorted = sorted(per_frame.keys())

    t_start = time.perf_counter()

    for chunk_start in range(0, len(frame_idx_sorted), frame_batch_size):
        chunk_frames = frame_idx_sorted[chunk_start:chunk_start + frame_batch_size]

        # Assemble image batch (M, 3, H, W) uint8 on device. pyfacer's
        # FaRLFaceAlignment.forward does ``images = images.float() / 255.0``
        # internally, so uint8 RGB is the expected input format.
        chunk_imgs = images_np_u8[chunk_frames]  # (M, H, W, 3) uint8 RGB
        images_bchw = torch.from_numpy(chunk_imgs).permute(0, 3, 1, 2).contiguous()
        images_bchw = images_bchw.to(device)

        # Assemble points (N, 5, 2) + image_ids (N,) + metadata
        pts5_list = []
        image_ids_list = []
        face_meta = []  # aligned with pts5_list: (t_global, p_idx)
        for local_b, t_global in enumerate(chunk_frames):
            for p_idx, pts_5 in per_frame[t_global]:
                pts5_list.append(pts_5)
                image_ids_list.append(local_b)
                face_meta.append((t_global, p_idx))

        if not pts5_list:
            # Shouldn't happen — we filtered to frames that have faces.
            if pbar is not None:
                pbar.update(len(chunk_frames))
            continue

        points_t    = torch.tensor(
            np.stack(pts5_list, axis=0), dtype=torch.float32, device=device,
        )  # (N, 5, 2)
        image_ids_t = torch.tensor(image_ids_list, dtype=torch.long, device=device)

        # pyfacer FaRLFaceAlignment.forward consumes:
        #   data['image_ids']: (N,) long indexing into images batch
        #   data['points']:    (N, 5, 2) float RetinaFace-order landmarks
        #                      (L_eye, R_eye, nose, L_mouth, R_mouth)
        # It IGNORES 'rects' entirely — passing them was the bug that
        # kept the face region of the POSES dict empty in the previous
        # revision of this module.
        faces_in = {
            "image_ids": image_ids_t,
            "points":    points_t,
        }

        try:
            with torch.inference_mode():
                faces_out = aligner(images_bchw, faces_in)
        except Exception as e:
            _logger.error(
                "FaRL face_aligner failed on frames %d..%d (%d faces): %s",
                chunk_frames[0], chunk_frames[-1], len(pts5_list), e,
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
