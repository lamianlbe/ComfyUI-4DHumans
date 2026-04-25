"""
Sapiens2InstancePose — pose-first crowd pipeline tuned for POV / mid-
close-up footage with heavy occlusion.

Replaces RTMDet (the weak link in BMP) with SAM3-image-segmentation as
the instance proposer, then runs Sapiens2-1B (308-keypoint, ICCV-grade
SOTA on OCHuman) on each instance group. To handle two people who
SAM3 splits into masks but who occlude each other so badly Sapiens2's
top-down crop would miss one, we apply BMP's iterative trick: run
once, find masks that didn't get attributed any keypoints, black out
the people we already covered, and re-run on the residual.

Core design choices, justified by the user's evidence:

  * SAM3 image (per-frame, no temporal tracking) for masks. Their POV
    footage breaks SAM3 video's tracker, so we don't rely on it —
    cross-frame association happens later via OKS-based pose tracking.
  * Bbox merging via union-find on IoU > 0 → groups of overlapping
    SAM3 instances share one Sapiens2 forward; each forward returns
    one strongest skeleton, which we attribute to whichever mask has
    the most visible keypoints inside it.
  * Two iterations max (matches BMP). One is sufficient on most
    frames; iteration 2 is the safety net for two-person tight
    overlap.
  * OKS-based cross-frame tracking with a 10-frame buffer for lost
    tracks. Resilient to brief occlusion / partial-out-of-frame.

Output (this first cut, for debug viz):
  debug_overlay : IMAGE — color-coded mask + bbox + 308-pt skeleton
                  per track, alpha-blended on the input frames.
  raw_keypoints : POSE_TIMELINE — per-track 308-pt + score timeline
                  (B, N, 308, 3) packed dict, useful for sanity
                  checks but not yet wired into the POSES dict
                  consumed by FaRL / FastSAM3DBody. That bridge ships
                  in Phase 5.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import comfy.utils

from ._mask_utils import (
    _DEBUG_PALETTE_RGB,
    build_debug_overlay,
    pack_mask,
    unpack_mask,
)

_logger = logging.getLogger(__name__)


# COCO-WB-like indices used for OKS sigma weighting. Sapiens2's 308-
# point set extends COCO-WholeBody, so the first 17 are body joints.
# We use a uniform sigma below across all 308 points — this is a
# simplification but adequate for tracking; for benchmarking we'd use
# per-keypoint sigmas like upstream OKS does.
_OKS_SIGMA_UNIFORM = 0.05


def _iou_packed(a_packed: np.ndarray, b_packed: np.ndarray) -> float:
    """Mask Jaccard IoU on bit-packed arrays (from pack_mask)."""
    if a_packed.size == 0 or b_packed.size == 0:
        return 0.0
    inter = int(np.unpackbits(a_packed & b_packed).sum())
    if inter == 0:
        return 0.0
    union = int(np.unpackbits(a_packed | b_packed).sum())
    return inter / union if union > 0 else 0.0


def _bbox_iou(a, b) -> float:
    """xyxy bbox IoU."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _bbox_from_mask(mask_bool: np.ndarray, padding_frac: float = 0.0) -> Optional[np.ndarray]:
    """Tight xyxy bbox of a bool mask + optional fractional padding."""
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    if padding_frac > 0:
        w, h = x2 - x1, y2 - y1
        px, py = int(round(w * padding_frac)), int(round(h * padding_frac))
        x1 -= px; y1 -= py; x2 += px; y2 += py
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _union_bboxes(bboxes: List[np.ndarray]) -> np.ndarray:
    """Smallest xyxy that contains all input boxes."""
    arr = np.stack(bboxes, axis=0)
    return np.array([
        arr[:, 0].min(), arr[:, 1].min(),
        arr[:, 2].max(), arr[:, 3].max(),
    ], dtype=np.float32)


def _union_find_groups(bboxes: List[np.ndarray], iou_thresh: float) -> List[List[int]]:
    """Group bbox indices by overlap (any IoU > thresh joins them)."""
    n = len(bboxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_iou(bboxes[i], bboxes[j]) > iou_thresh:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _count_keypoints_in_mask(
    visible_xy: np.ndarray,        # (V, 2) already-filtered visible keypoints
    mask: np.ndarray,              # (H, W) bool
    H: int, W: int,
    early_stop: Optional[int] = None,
) -> int:
    """Count how many of ``visible_xy`` fall inside ``mask``. Optional
    short-circuit when the count reaches ``early_stop`` — useful when
    the caller only needs to know "≥ N", not the exact count."""
    hits = 0
    for x, y in visible_xy:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < H and 0 <= ix < W and mask[iy, ix]:
            hits += 1
            if early_stop is not None and hits >= early_stop:
                return hits
    return hits


def _attribute_pose_to_mask(
    pose_xy: np.ndarray,           # (K, 2)
    pose_scores: np.ndarray,       # (K,)
    masks_in_group: List[np.ndarray],   # list of bool (H, W)
    H: int, W: int,
    conf_thresh: float = 0.3,
) -> Optional[int]:
    """Return the index (within ``masks_in_group``) whose mask contains
    the most visible keypoints of ``pose_xy``. None means no mask wins
    (all keypoints fall outside every mask — should be rare)."""
    visible = pose_scores >= conf_thresh
    if not visible.any():
        return None
    visible_xy = pose_xy[visible]

    counts = np.zeros(len(masks_in_group), dtype=np.int32)
    for k in range(len(masks_in_group)):
        m = masks_in_group[k]
        for x, y in visible_xy:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < H and 0 <= ix < W and m[iy, ix]:
                counts[k] += 1
    if counts.max() == 0:
        return None
    return int(counts.argmax())


def _oks(
    kps_a: np.ndarray, sc_a: np.ndarray,
    kps_b: np.ndarray, sc_b: np.ndarray,
    bbox_area: float,
    sigma: float = _OKS_SIGMA_UNIFORM,
    conf_thresh: float = 0.3,
) -> float:
    """Object Keypoint Similarity between two pose detections.

    Uses uniform sigma across all keypoints (good enough for tracking,
    not for benchmarking). Only joints visible in BOTH detections
    contribute; if there are fewer than 3 such joints, returns 0.
    """
    visible = (sc_a >= conf_thresh) & (sc_b >= conf_thresh)
    if visible.sum() < 3:
        return 0.0
    diff = kps_a - kps_b                  # (K, 2)
    d2 = (diff ** 2).sum(axis=-1)         # (K,)
    e = d2 / (2 * (sigma ** 2) * (bbox_area + 1e-6))
    contrib = np.exp(-e)
    return float(contrib[visible].sum() / visible.sum())


def _track_by_oks(
    per_frame_detections: List[List[dict]],
    iou_thresh: float = 0.3,
    buffer_frames: int = 10,
) -> List[List[Tuple[int, dict]]]:
    """Greedy OKS-based cross-frame tracker with lost-track buffer.

    Returns per-frame list of ``(track_id, detection_dict)`` tuples.
    """
    next_tid = 0
    # active_tracks[tid] = (last_seen_frame, last_detection)
    active_tracks: Dict[int, Tuple[int, dict]] = {}
    out_per_frame: List[List[Tuple[int, dict]]] = []

    for t, dets in enumerate(per_frame_detections):
        # Drop tracks lost for too long
        active_tracks = {
            tid: v for tid, v in active_tracks.items()
            if t - v[0] <= buffer_frames
        }

        if not dets:
            out_per_frame.append([])
            continue

        # Build OKS similarity matrix between current dets and all
        # still-active (recently-seen) tracks.
        active_tids = list(active_tracks.keys())
        oks_mat = np.zeros((len(active_tids), len(dets)), dtype=np.float32)
        for i, tid in enumerate(active_tids):
            _t_seen, prev_det = active_tracks[tid]
            for j, d in enumerate(dets):
                # Use prev bbox area for normalization
                pa = prev_det["bbox"]
                bbox_area = max(1.0, (pa[2] - pa[0]) * (pa[3] - pa[1]))
                oks_mat[i, j] = _oks(
                    prev_det["kpts"][:, :2], prev_det["kpts"][:, 2],
                    d["kpts"][:, :2],        d["kpts"][:, 2],
                    bbox_area=bbox_area,
                )

        assigned_t = set()
        assigned_d = set()
        frame_out: List[Tuple[int, dict]] = [None] * len(dets)

        # Greedy: highest OKS pair first
        flat = [
            (oks_mat[i, j], i, j)
            for i in range(len(active_tids))
            for j in range(len(dets))
            if oks_mat[i, j] >= iou_thresh
        ]
        flat.sort(reverse=True)
        for s, i, j in flat:
            if i in assigned_t or j in assigned_d:
                continue
            tid = active_tids[i]
            frame_out[j] = (tid, dets[j])
            active_tracks[tid] = (t, dets[j])
            assigned_t.add(i)
            assigned_d.add(j)

        # Unmatched dets → spawn new tracks
        for j in range(len(dets)):
            if frame_out[j] is None:
                frame_out[j] = (next_tid, dets[j])
                active_tracks[next_tid] = (t, dets[j])
                next_tid += 1

        out_per_frame.append(frame_out)

    return out_per_frame


def _draw_skeleton_on_frame(
    frame: np.ndarray,           # (H, W, 3) uint8 IN-PLACE EDIT
    kpts: np.ndarray,            # (K, 3)  x, y, score
    color_rgb: Tuple[int, int, int],
    skeleton_links: List[Tuple[int, int]],
    conf_thresh: float = 0.3,
    radius: int = 3,
    thickness: int = 2,
):
    import cv2
    H, W = frame.shape[:2]

    # Joints
    for k in range(kpts.shape[0]):
        x, y, c = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
        if c < conf_thresh:
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(frame, (ix, iy), radius, color_rgb,
                        thickness=-1, lineType=cv2.LINE_AA)

    # Bones (only available if skeleton links provided)
    for i, j in skeleton_links:
        if i >= kpts.shape[0] or j >= kpts.shape[0]:
            continue
        ci, cj = float(kpts[i, 2]), float(kpts[j, 2])
        if ci < conf_thresh or cj < conf_thresh:
            continue
        p1 = (int(round(float(kpts[i, 0]))), int(round(float(kpts[i, 1]))))
        p2 = (int(round(float(kpts[j, 0]))), int(round(float(kpts[j, 1]))))
        if not (0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H):
            continue
        cv2.line(frame, p1, p2, color_rgb, thickness=thickness,
                  lineType=cv2.LINE_AA)


def _coco17_skeleton_links() -> List[Tuple[int, int]]:
    """The body-only subset of links that's safe to draw — assumes
    Sapiens2's first 17 keypoints are COCO-17 ordered (which matches
    its 308-point definition's body section)."""
    return [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6),
    ]


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class Sapiens2InstancePoseNode:
    """SAM3 masks → bbox-merge groups → Sapiens2 → iterate-on-residual
    → OKS cross-frame tracker → debug-overlay video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":     ("IMAGE",),
                "sam3_masks": ("MASK",),    # (B*N, H, W) from SAM3ImageSegmentation
                "sapiens2":   ("SAPIENS2",),
                "bbox_iou_merge_thresh": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "tooltip": (
                            "If two SAM3 instances' bboxes overlap by "
                            "more than this IoU, merge them into one "
                            "group that gets a single Sapiens2 forward. "
                            "0.0 = merge any touching bboxes (typical "
                            "for tightly overlapping people)."
                        ),
                    },
                ),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Per-keypoint visibility threshold. Used "
                            "for: (a) attributing a pose to a mask "
                            "(only keypoints above this count), "
                            "(b) deciding which masks are 'covered' "
                            "after iter 1 → triggers iter 2 on the "
                            "residual."
                        ),
                    },
                ),
                "extra_claim_keypoints": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": (
                            "After a pose's main mask wins, ALSO mark "
                            "as covered any other uncovered mask whose "
                            "interior catches at least this many "
                            "visible keypoints from the same pose. "
                            "Catches the case where SAM3 splits one "
                            "person into multiple disjoint masks "
                            "(torso + arm + leg). Set to 0 to disable "
                            "(only winner mask covered, like BMP). "
                            "Tighter (≥10) = more conservative, may "
                            "leave fragments uncovered for iter 2 to "
                            "spuriously re-detect. Looser (≤3) = may "
                            "incorrectly absorb a touching second "
                            "person's mask whose few stray pixels "
                            "caught a couple of edge keypoints."
                        ),
                    },
                ),
                "max_iters": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": (
                            "Iteration cap for the BMP-style 'black "
                            "out covered, re-run' loop. 2 matches BMP "
                            "and is sufficient for two-person POV "
                            "occlusion. Going higher is rarely "
                            "useful and roughly doubles cost."
                        ),
                    },
                ),
                "oks_track_thresh": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Minimum OKS to call two cross-frame "
                            "detections the same person. Lower = "
                            "more lenient re-ID; higher = stricter, "
                            "more new tracks on fast motion."
                        ),
                    },
                ),
                "track_buffer_frames": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                        "tooltip": (
                            "How many frames a track can be 'lost' "
                            "(missing from detections) before we "
                            "drop it. POV footage has frequent brief "
                            "occlusion, so 10 frames is a reasonable "
                            "default."
                        ),
                    },
                ),
                "sapiens2_batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Per-forward batch on the Sapiens2 ViT. "
                            "Each crop is 1024×768; 4 fits comfortably "
                            "on Blackwell at fp32, push higher if "
                            "you're using bf16/fp16."
                        ),
                    },
                ),
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Render mask + bbox + 308-pt skeleton "
                            "overlay on the input frames. CPU "
                            "post-processing pass — not free, but "
                            "free relative to Sapiens2 inference. "
                            "Off → output passes input images "
                            "unchanged."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debug_overlay",)
    FUNCTION = "run"
    CATEGORY = "4dhumans"

    def run(self, images, sam3_masks, sapiens2,
            bbox_iou_merge_thresh, score_threshold,
            extra_claim_keypoints, max_iters,
            oks_track_thresh, track_buffer_frames, sapiens2_batch_size,
            debug_overlay):
        pipe = sapiens2["pipeline"]

        # ---- Decode shapes ------------------------------------------------
        # images: (B, H, W, 3) float [0, 1] RGB
        # sam3_masks: (B * N_total, H, W) float in {0, 1}
        B, H, W, _C = images.shape
        total = sam3_masks.shape[0]
        if total % B != 0:
            raise ValueError(
                f"sam3_masks shape {tuple(sam3_masks.shape)} not divisible "
                f"by frame count {B} — expected (B*N, H, W) layout."
            )
        n_persons = total // B

        # Convert to BGR uint8 for cv2-style downstream ops (Sapiens2's
        # data preprocessor handles BGR→RGB internally).
        rgb_u8 = (images.clamp(0, 1) * 255).byte().cpu().numpy()
        bgr_u8 = rgb_u8[..., ::-1].copy()

        masks_np = sam3_masks.detach().cpu().numpy().astype(bool)
        masks_np = masks_np.reshape(B, n_persons, H, W)

        skeleton_links = _coco17_skeleton_links()
        pbar = comfy.utils.ProgressBar(B + 1)
        t_start = time.perf_counter()

        # ---- Per-frame detection list ------------------------------------
        # Each entry is a list of dicts: {kpts: (308,3), bbox: (4,), mask_packed}
        per_frame_detections: List[List[dict]] = []

        for t in range(B):
            frame_dets: List[dict] = []
            frame_image = bgr_u8[t].copy()           # we mutate per iteration
            frame_masks_per_slot = [masks_np[t, p] for p in range(n_persons)]

            # mask_idx → SAM3 slot p (preserved through iterations because
            # we look up by integer slot, not by mask)
            uncovered_slots = [
                p for p in range(n_persons)
                if frame_masks_per_slot[p].any()     # skip empty mask slots
            ]

            for it in range(max_iters):
                if not uncovered_slots:
                    break

                # Compute per-slot bbox (from each remaining mask)
                slot_bboxes = []
                kept_slots = []
                for p in uncovered_slots:
                    bb = _bbox_from_mask(frame_masks_per_slot[p],
                                          padding_frac=0.10)
                    if bb is None:
                        continue
                    # Clip to image
                    bb[0] = max(0, bb[0]); bb[1] = max(0, bb[1])
                    bb[2] = min(W, bb[2]); bb[3] = min(H, bb[3])
                    if bb[2] - bb[0] < 5 or bb[3] - bb[1] < 5:
                        continue
                    slot_bboxes.append(bb)
                    kept_slots.append(p)

                if not slot_bboxes:
                    break

                # Group by bbox IoU; each group → one Sapiens2 forward
                groups = _union_find_groups(slot_bboxes, bbox_iou_merge_thresh)
                merged_bboxes = np.stack(
                    [_union_bboxes([slot_bboxes[i] for i in g]) for g in groups],
                    axis=0,
                )

                # ---- Sapiens2 forward on the merged bboxes --------------
                # Mask out covered regions in the (already mutated)
                # `frame_image` so iter 2+ doesn't re-detect iter-1 people
                # — but only if it's iter ≥ 2. iter 1 sees the full frame.
                kpts_n308x2, scores_n308 = pipe.predict(
                    frame_image,
                    merged_bboxes,
                    batch_size=sapiens2_batch_size,
                )

                # ---- Attribute poses to masks ---------------------------
                # We process poses in DESCENDING quality order so that:
                #   (1) higher-quality poses claim their masks first
                #   (2) duplicate / split-mask poses arriving later
                #       see their target mask already covered → skipped
                # Quality = sum of visible keypoint confidences.
                pose_quality = []
                for g_idx in range(len(groups)):
                    sc = scores_n308[g_idx]
                    visible = sc >= score_threshold
                    pose_quality.append(
                        float(sc[visible].sum()) if visible.any() else -1.0
                    )
                g_order = sorted(range(len(groups)),
                                  key=lambda i: -pose_quality[i])

                newly_covered_slots = set()
                n_extra_claims_this_iter = 0
                n_dupes_skipped_this_iter = 0

                for g_idx in g_order:
                    if pose_quality[g_idx] < 0:
                        # Pose has no visible keypoints — drop.
                        continue

                    g = groups[g_idx]
                    pose_xy = kpts_n308x2[g_idx]      # (308, 2)
                    pose_sc = scores_n308[g_idx]      # (308,)

                    # Step 1: in-group attribution, but ONLY among masks
                    # not already covered by an earlier-processed pose
                    # this iteration.
                    in_group_uncovered = [
                        i for i in g
                        if kept_slots[i] not in newly_covered_slots
                    ]
                    if not in_group_uncovered:
                        # Every mask in this group was claimed by a
                        # higher-quality pose's main + extra-claim. This
                        # pose is a duplicate → drop without stashing.
                        n_dupes_skipped_this_iter += 1
                        continue

                    masks_for_g = [
                        frame_masks_per_slot[kept_slots[i]]
                        for i in in_group_uncovered
                    ]
                    winner_local = _attribute_pose_to_mask(
                        pose_xy, pose_sc, masks_for_g, H, W,
                        conf_thresh=score_threshold,
                    )
                    if winner_local is None:
                        # Pose has visible keypoints but none fall in
                        # any uncovered mask of its own group. This is
                        # the symptom the user observed earlier — drop
                        # rather than attribute to the wrong mask.
                        continue
                    winner_slot = kept_slots[in_group_uncovered[winner_local]]
                    newly_covered_slots.add(winner_slot)

                    # Stash the detection
                    kpts_full = np.concatenate(
                        [pose_xy, pose_sc[:, None]], axis=-1,
                    ).astype(np.float32)              # (308, 3)
                    frame_dets.append({
                        "kpts":        kpts_full,
                        "bbox":        merged_bboxes[g_idx].astype(np.float32),
                        "mask_packed": pack_mask(frame_masks_per_slot[winner_slot]),
                        "slot":        winner_slot,
                        "iter":        it,
                    })

                    # Step 2: extra-claim. The winner's pose can also
                    # cover OTHER uncovered masks (across the whole
                    # frame, not just this group) if those masks catch
                    # at least `extra_claim_keypoints` visible
                    # keypoints from this pose. Catches SAM3-fragmented
                    # masks of the same person.
                    if extra_claim_keypoints > 0:
                        visible_mask = pose_sc >= score_threshold
                        if visible_mask.any():
                            visible_xy = pose_xy[visible_mask]
                            for other_slot in uncovered_slots:
                                if other_slot in newly_covered_slots:
                                    continue
                                hits = _count_keypoints_in_mask(
                                    visible_xy,
                                    frame_masks_per_slot[other_slot],
                                    H, W,
                                    early_stop=extra_claim_keypoints,
                                )
                                if hits >= extra_claim_keypoints:
                                    newly_covered_slots.add(other_slot)
                                    n_extra_claims_this_iter += 1

                if n_dupes_skipped_this_iter or n_extra_claims_this_iter:
                    _logger.debug(
                        "Sapiens2 frame %d iter %d: %d poses kept, %d "
                        "dupes skipped (mask already taken), %d extra-"
                        "claimed masks (SAM3 fragments absorbed).",
                        t, it,
                        len(newly_covered_slots) - n_extra_claims_this_iter,
                        n_dupes_skipped_this_iter,
                        n_extra_claims_this_iter,
                    )

                if not newly_covered_slots:
                    # Nothing got attributed this iteration; further
                    # iterations will just produce the same result.
                    break

                # Black out the covered slots' pixels in frame_image so
                # iter 2 doesn't see those people. SAM3 mask gives us
                # exact silhouette — set mask region to 0 (matches BMP's
                # mask_out behavior).
                for p in newly_covered_slots:
                    frame_image[frame_masks_per_slot[p]] = 0

                uncovered_slots = [p for p in uncovered_slots
                                    if p not in newly_covered_slots]

            per_frame_detections.append(frame_dets)
            pbar.update(1)

        # ---- Cross-frame tracking via OKS --------------------------------
        per_frame_tracks = _track_by_oks(
            per_frame_detections,
            iou_thresh=oks_track_thresh,
            buffer_frames=track_buffer_frames,
        )

        # Build a stable slot index for color assignment in the overlay
        all_tids = sorted({
            tid for ft in per_frame_tracks for tid, _ in ft
        })
        tid_to_slot = {tid: i for i, tid in enumerate(all_tids)}

        pbar.update(1)
        elapsed = time.perf_counter() - t_start

        if not all_tids:
            _logger.warning(
                "Sapiens2InstancePose: no detections survived. Returning "
                "input images as-is."
            )
            return (images,)

        _logger.info(
            "Sapiens2InstancePose: %d frames, %d unique tracks, "
            "%.2fs (%.2fs/frame)",
            B, len(all_tids), elapsed, elapsed / max(B, 1),
        )

        # ---- Debug overlay ----------------------------------------------
        if not debug_overlay:
            return (images,)

        # Stage 1: shared mask overlay (color per track slot)
        per_frame_items = [
            [(tid, det["mask_packed"]) for tid, det in ft]
            for ft in per_frame_tracks
        ]
        overlay_t, legend = build_debug_overlay(
            images=images,
            per_frame_items=per_frame_items,
            H=H, W=W,
        )
        _logger.info("Sapiens2 debug overlay legend: %s", legend)

        # Stage 2: draw 308-pt skeletons with matching colors
        arr = (overlay_t.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        for t, frame_tracks in enumerate(per_frame_tracks):
            for tid, det in frame_tracks:
                slot = tid_to_slot[tid]
                color = _DEBUG_PALETTE_RGB[slot % len(_DEBUG_PALETTE_RGB)]
                _draw_skeleton_on_frame(
                    arr[t],
                    det["kpts"],
                    color_rgb=tuple(int(c) for c in color.tolist()),
                    skeleton_links=skeleton_links,
                    conf_thresh=score_threshold,
                )

        out_overlay = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return (out_overlay,)
