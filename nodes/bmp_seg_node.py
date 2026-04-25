"""
BMPInstanceSegmentation — per-frame BMP inference + cross-frame
IoU tracking + hot-swap keypoint output for Fast SAM 3D Body.

Three bbox-source modes selectable via the ``bbox_source`` input:

  * ``default`` (BMP stock behaviour). RTMDet runs every iteration on
    the original / blacked-out image. This is what BMP's predict()
    does when ``bboxes=None`` is passed.

  * ``full_image_iter0``. iter 0 uses a single (0, 0, W, H) bbox so
    Sapiens2-style "let the top-down argmax pick the strongest
    person" handles the first detection. iter 1+ falls back to
    RTMDet on the blacked-out image. Useful when RTMDet is mostly
    fine but occasionally merges two close people in iter 0.

  * ``full_image_all_iters``. Every iteration uses (0, 0, W, H). We
    drive the loop ourselves: call ``bmp.predict(num_bmp_iters=1,
    bboxes=full)`` per iter, black out using the returned masks,
    repeat. RTMDet is bypassed completely. Same idea as
    Sapiens2InstancePose's ``start_with_full_image=True`` mode but
    using BMP's PMPose + SAM-pose2seg components.

For the third mode we briefly mutate ``bmp.config.num_bmp_iters = 1``
between iterations and restore it after — no vendoring of BMP
source needed.

BMP is single-image (no temporal tracking), so we wrap it with a
greedy mask-IoU matcher identical to SAM3ImageSegmentation /
CrowdSAMInstanceSegmentation. Result: frame-grouped
``(B*N, H, W)`` masks and a per-person keypoint timeline aligned
to those same track IDs.

The keypoint output (``bmp_pose``) is a ``YOLO11POSE``-compatible
dict, meaning you can plug it straight into the ``yolo11_pose``
input slot of ``FastSAM3DBodyFaRLPose`` and BMP's keypoints will
drive Fast SAM 3D Body's hand-bbox decoder in place of the
YOLO11m-Pose detector. Rationale: the user observed BMP's 17-pt
skeleton is visibly more accurate than YOLO11m-Pose on occluded
frames (and per the paper's OCHuman numbers that tracks —
49.2 AP 2D pose is SOTA).

The BMP_POSE wrapper is a "replay model": it caches per-frame
results at BMPInstanceSegmentation-time and serves them back when
FastSAM3DBody later calls ``.predict(source=chunk)``. Works because
FastSAM3DBody always iterates frames in order with chunk-aligned
slicing.

Outputs
-------
masks        : MASK, (B*N, H, W) float — same layout as SAM3 /
               YOLO-seg / CrowdSAM. N = total unique track IDs.
bmp_pose     : YOLO11POSE — plug into FastSAM3DBody's yolo11_pose
               input. Replaces the YOLO11m-Pose detector pass.
debug_overlay: IMAGE, (B, H, W, 3) float — color-coded masks +
               bbox outlines + tid labels + skeleton overlay.
"""

import logging
import time
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import torch

import comfy.utils

from ._mask_utils import build_debug_overlay, pack_mask, unpack_mask

_logger = logging.getLogger(__name__)


# COCO-17 skeleton edges (for debug viz). Index pairs into the 17
# keypoints in standard COCO ordering.
_COCO17_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
    (5, 6), (5, 11), (6, 12), (11, 12),     # torso
    (0, 1), (0, 2), (1, 3), (2, 4),         # head
    (0, 5), (0, 6),                         # neck→shoulders
]


def _iou_packed(a_packed: np.ndarray, b_packed: np.ndarray) -> float:
    """Jaccard IoU between two bit-packed bool masks (from pack_mask).

    Works directly on uint8 packed bytes via popcount — no unpack to
    full H×W needed for each comparison, which keeps cross-frame
    matching cheap even on long videos.
    """
    if a_packed.size == 0 or b_packed.size == 0:
        return 0.0
    inter = int(np.unpackbits(a_packed & b_packed).sum())
    if inter == 0:
        return 0.0
    union = int(np.unpackbits(a_packed | b_packed).sum())
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# Replay "model" for the YOLO11POSE output slot
# --------------------------------------------------------------------------
#
# FastSAM3DBody's YOLO11m-Pose consumer (see _fastsam3db_inference.py:434-469)
# treats yolo11_pose["model"] as an Ultralytics YOLO instance, calling:
#
#     yolo_model.to("cuda"/"cpu")
#     results = list(yolo_model.predict(source=chunk, ...))
#
# where `chunk` is a list of BGR uint8 frames.  Each Result must expose
# `.boxes.xyxy` + `.keypoints.data` as torch tensors so the downstream
# aligner can IoU-match YOLO detections to our mask bboxes.
#
# Our replay model pre-computes BMP results for all B frames once, then
# serves them sequentially as FastSAM3DBody iterates chunks.  It's a
# thin shim: just enough duck-typing to keep FastSAM3DBody unchanged.

class _BMPResult:
    """Ultralytics-Result-compatible shim for a single frame."""
    __slots__ = ("boxes", "keypoints")

    def __init__(self, bboxes_xyxy, scores, keypoints_17x3):
        if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
            self.boxes = None
            self.keypoints = None
            return
        xyxy_t = torch.as_tensor(np.asarray(bboxes_xyxy), dtype=torch.float32)
        conf_t = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
        kp_t   = torch.as_tensor(np.asarray(keypoints_17x3), dtype=torch.float32)
        # boxes.id is what ultralytics' tracker fills — leave as None so
        # FastSAM3DBody falls back to its own greedy IoU matching.
        self.boxes = SimpleNamespace(
            xyxy=xyxy_t,
            conf=conf_t,
            id=None,
            data=torch.cat([xyxy_t, conf_t.unsqueeze(-1)], dim=-1),
        )
        self.keypoints = SimpleNamespace(data=kp_t)


class _BMPReplayModel:
    """Stateful replay of pre-computed BMP results as if it were an
    Ultralytics YOLO-Pose instance.

    Safe assumptions (validated against _fastsam3db_inference.py):
      - caller iterates frames IN ORDER
      - chunks are contiguous and non-overlapping
      - caller wraps its loop in one continuous .predict() sequence,
        so we can maintain an internal cursor
    """

    def __init__(self, per_frame_results: List[Optional[_BMPResult]]):
        self._cache = per_frame_results
        self._cursor = 0

    def to(self, device):
        # No-op: BMP was already placed on its device at load time and
        # there's no tensor state on THIS wrapper to move.
        return self

    def predict(self, source=None, **kwargs):
        """Return cached results for the next ``len(source)`` frames.

        Returns a list (not a generator) because FastSAM3DBody's
        consumer immediately wraps in ``list(...)`` anyway.
        """
        n = len(source) if source is not None else 1
        start = self._cursor
        end = min(start + n, len(self._cache))
        out = self._cache[start:end]
        self._cursor = end
        return out

    def reset(self):
        """Allow the same replay model to be re-used for a second pass
        (e.g. if FastSAM3DBody is invoked twice with the same node output).
        """
        self._cursor = 0


# --------------------------------------------------------------------------
# Debug overlay: extend the shared mask overlay with skeletons
# --------------------------------------------------------------------------

def _draw_skeletons_on_overlay(overlay_float, per_frame_tids_with_slot,
                                H, W, conf_thresh=0.3):
    """Draw COCO-17 skeletons on top of the color-coded mask overlay.

    Parameters
    ----------
    overlay_float : torch.Tensor
        (B, H, W, 3) float in [0, 1], the mask overlay output from
        build_debug_overlay that we draw skeletons on top of.
    per_frame_tids_with_slot : list[B] of list[tuple]
        per-frame list of ``(tid, slot, kpts_17x3)`` triples. The slot
        index is looked up from ``id_to_slot`` so the skeleton color
        matches the mask color for the same track.

    Uses the same palette as build_debug_overlay so a person's mask
    color matches their skeleton color. Returns a new tensor; input
    is not modified.
    """
    import cv2
    from ._mask_utils import _DEBUG_PALETTE_RGB

    arr = (overlay_float.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)

    for t, frame_entries in enumerate(per_frame_tids_with_slot):
        frame = arr[t]
        for tid, slot, kpts in frame_entries:
            color = _DEBUG_PALETTE_RGB[slot % len(_DEBUG_PALETTE_RGB)]
            color_list = [int(c) for c in color.tolist()]

            # Draw joints as small filled circles
            for k in range(kpts.shape[0]):
                x, y, c = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
                if c < conf_thresh:
                    continue
                if 0 <= int(round(x)) < W and 0 <= int(round(y)) < H:
                    cv2.circle(
                        frame, (int(round(x)), int(round(y))),
                        radius=3, color=color_list, thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

            # Draw bones between confident endpoints
            for i, j in _COCO17_EDGES:
                if i >= kpts.shape[0] or j >= kpts.shape[0]:
                    continue
                ci = float(kpts[i, 2])
                cj = float(kpts[j, 2])
                if ci < conf_thresh or cj < conf_thresh:
                    continue
                p1 = (int(round(float(kpts[i, 0]))), int(round(float(kpts[i, 1]))))
                p2 = (int(round(float(kpts[j, 0]))), int(round(float(kpts[j, 1]))))
                if not all(0 <= v < dim for v, dim in
                           [(p1[0], W), (p1[1], H), (p2[0], W), (p2[1], H)]):
                    continue
                cv2.line(frame, p1, p2, color=color_list,
                         thickness=2, lineType=cv2.LINE_AA)

    return torch.from_numpy(arr.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class BMPInstanceSegmentationNode:
    """Run BMP per-frame, match across frames by mask IoU, emit
    seg-node-compatible masks + YOLO11POSE-compatible pose."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":         ("IMAGE",),
                "bmp":            ("BMP",),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Per-detection score filter. BMP already "
                            "filters internally via RTMDet's conf — "
                            "this is a belt-and-suspenders gate on "
                            "top. Scores here come from the mean of "
                            "the 17 keypoint confidences."
                        ),
                    },
                ),
                "iou_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Minimum mask IoU with the previous frame "
                            "to call it the SAME person. Lower = more "
                            "lenient re-ID; higher = stricter, more "
                            "likely to spawn new tracks when people "
                            "move fast or briefly occlude."
                        ),
                    },
                ),
                "bbox_source": (
                    [
                        "default",
                        "full_image_iter0",
                        "full_image_all_iters",
                    ],
                    {
                        "default": "default",
                        "tooltip": (
                            "How BMP gets bboxes for each iteration:\n"
                            "  • default — RTMDet runs every iter "
                            "    (BMP stock). Best when RTMDet works.\n"
                            "  • full_image_iter0 — iter 0 uses "
                            "    (0,0,W,H) so PMPose's top-down argmax "
                            "    picks the strongest person; iter 1+ "
                            "    still RTMDet on the residual. Use "
                            "    when RTMDet only sometimes merges "
                            "    close people in the first pass.\n"
                            "  • full_image_all_iters — every iter "
                            "    uses (0,0,W,H). Bypasses RTMDet "
                            "    entirely, mirrors Sapiens2InstancePose's "
                            "    start_with_full_image=True. Each iter "
                            "    yields one pose; max number of "
                            "    detections per frame = max_iters_full_"
                            "    image."
                        ),
                    },
                ),
                "max_iters_full_image": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Only used when bbox_source = "
                            "full_image_all_iters. Each iter produces "
                            "exactly one pose, so this is the upper "
                            "bound on # of people detected per frame. "
                            "Set to N+1 where N is the max headcount "
                            "you expect."
                        ),
                    },
                ),
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Output a second IMAGE with each tracked "
                            "person's mask + skeleton color-coded on "
                            "the original frames (same color palette "
                            "as the other seg nodes). Off by default "
                            "(CPU pass per frame). When off, the "
                            "debug_overlay output passes through the "
                            "input images unchanged."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "YOLO11POSE", "IMAGE")
    RETURN_NAMES = ("masks", "bmp_pose", "debug_overlay")
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    @staticmethod
    def _predict_full_image_all_iters(
        bmp_model,
        image_bgr: np.ndarray,
        full_image_bbox: np.ndarray,
        max_iters: int,
        H: int, W: int,
    ) -> dict:
        """Orchestrate BMP iter-by-iter ourselves with a fixed full-
        image bbox each time, manually blacking out the returned
        masks between iterations.

        Side-effect: mutates ``bmp_model.config.num_bmp_iters`` to 1
        for the duration of this call, restored afterwards in a
        finally. Safe because predict() reads num_bmp_iters at the
        top of each call and we run synchronously per-frame.
        """
        original_iters = bmp_model.config.num_bmp_iters
        bmp_model.config.num_bmp_iters = 1

        masked = image_bgr.copy()
        agg_bboxes = []
        agg_masks = []
        agg_keypoints = []
        agg_presence = []
        agg_visibility = []

        try:
            for it in range(max_iters):
                res = bmp_model.predict(
                    image=masked,
                    bboxes=full_image_bbox,
                    return_intermediates=False,
                )

                # res is a dict with bboxes / masks / keypoints / etc.
                # In single-iter mode with one bbox we expect 0 or 1
                # detections per call (PMPose's top-down picks the
                # strongest visible person; SAM2 generates one mask
                # for that pose).
                bboxes_it = np.asarray(
                    res.get("bboxes", np.zeros((0, 4))), dtype=np.float32,
                )
                if bboxes_it.shape[0] == 0:
                    # Nothing left to find — no point iterating more.
                    break

                masks_it     = np.asarray(res.get("masks", np.zeros((0, H, W), dtype=np.uint8)))
                kpts_it      = np.asarray(res.get("keypoints", np.zeros((0, 17, 3))))
                presence_it  = np.asarray(res.get("presence", np.zeros((0, 17, 1))))
                visibility_it = np.asarray(res.get("visibility", np.zeros((0, 17, 1))))

                agg_bboxes.append(bboxes_it)
                agg_masks.append(masks_it)
                agg_keypoints.append(kpts_it)
                agg_presence.append(presence_it)
                agg_visibility.append(visibility_it)

                # Black out using SAM2-refined per-instance masks. Same
                # behaviour as BMP's internal _mask_out_image, just on
                # the image we own.
                for m in masks_it:
                    m_bool = m.astype(bool)
                    if m_bool.shape == (H, W) and m_bool.any():
                        masked[m_bool] = 0
        finally:
            bmp_model.config.num_bmp_iters = original_iters

        if not agg_bboxes:
            return {
                "bboxes":     np.zeros((0, 4), dtype=np.float32),
                "masks":      np.zeros((0, H, W), dtype=np.uint8),
                "keypoints":  np.zeros((0, 17, 3), dtype=np.float32),
                "presence":   np.zeros((0, 17, 1), dtype=np.float32),
                "visibility": np.zeros((0, 17, 1), dtype=np.float32),
            }

        return {
            "bboxes":     np.concatenate(agg_bboxes, axis=0),
            "masks":      np.concatenate(agg_masks, axis=0),
            "keypoints":  np.concatenate(agg_keypoints, axis=0),
            "presence":   np.concatenate(agg_presence, axis=0),
            "visibility": np.concatenate(agg_visibility, axis=0),
        }

    def segment(self, images, bmp, score_threshold, iou_threshold,
                bbox_source="default", max_iters_full_image=3,
                debug_overlay=False):
        bmp_model = bmp["bmp"]

        # images: (B, H, W, 3) float [0, 1], RGB
        B, H, W, _C = images.shape

        # BMP wants BGR uint8 per-frame — its internal OpenCV load path
        # expects cv2 convention (BGR). Pre-convert the whole batch on
        # CPU; cheap since we're already uint8.
        rgb_u8 = (images.clamp(0, 1) * 255).byte().cpu().numpy()     # (B, H, W, 3) RGB
        bgr_u8 = rgb_u8[..., ::-1]                                    # (B, H, W, 3) BGR

        pbar = comfy.utils.ProgressBar(B + 1)

        _logger.info(
            "BMP bbox_source=%s%s",
            bbox_source,
            f" (max_iters_full_image={max_iters_full_image})"
                if bbox_source == "full_image_all_iters" else "",
        )

        # ------------------------------------------------------------------
        # Phase 1: per-frame BMP inference
        # ------------------------------------------------------------------
        t_infer = time.perf_counter()

        per_frame_detections = [[] for _ in range(B)]  # list of dicts per frame
        replay_cache: List[Optional[_BMPResult]] = [None] * B
        logged_once = False

        # Pre-compute the (1, 4) full-image bbox used by both
        # full_image_* modes; constant per video so build it once.
        full_image_bbox = np.array(
            [[0.0, 0.0, float(W), float(H)]], dtype=np.float32,
        )

        for t in range(B):
            try:
                if bbox_source == "default":
                    # BMP stock: RTMDet drives every iteration.
                    result = bmp_model.predict(
                        image=bgr_u8[t].copy(),
                        bboxes=None,
                        return_intermediates=False,
                    )
                elif bbox_source == "full_image_iter0":
                    # iter 0 uses full-image bbox (PMPose top-down
                    # argmax picks strongest person), iter 1+ falls
                    # back to RTMDet on the blacked-out residual —
                    # this is exactly what BMP's predict() already
                    # does when given external bboxes.
                    result = bmp_model.predict(
                        image=bgr_u8[t].copy(),
                        bboxes=full_image_bbox,
                        return_intermediates=False,
                    )
                elif bbox_source == "full_image_all_iters":
                    # Every iter uses full-image bbox. We orchestrate
                    # the loop ourselves: call BMP with num_bmp_iters=1
                    # repeatedly, blacking out the returned masks
                    # between calls. RTMDet is bypassed entirely.
                    result = self._predict_full_image_all_iters(
                        bmp_model,
                        bgr_u8[t].copy(),
                        full_image_bbox,
                        max_iters=max_iters_full_image,
                        H=H, W=W,
                    )
                else:
                    raise ValueError(f"Unknown bbox_source: {bbox_source}")
            except Exception as e:
                _logger.error("BMP failed on frame %d: %s", t, e)
                pbar.update(1)
                continue

            bboxes    = np.asarray(result.get("bboxes",    np.zeros((0, 4))), dtype=np.float32)
            masks     = np.asarray(result.get("masks",     np.zeros((0, H, W), dtype=np.uint8)))
            keypoints = np.asarray(result.get("keypoints", np.zeros((0, 17, 3))), dtype=np.float32)

            # Defensive: BMP internally uses (N, 17, 3) with the 3rd
            # column being per-keypoint score. Some variants return
            # float64; cast to float32 for consistency.
            if keypoints.ndim == 3 and keypoints.shape[-1] >= 3:
                kp17 = keypoints[:, :17, :3].astype(np.float32)
            elif keypoints.ndim == 3 and keypoints.shape[-1] == 2:
                kp17 = np.concatenate(
                    [keypoints[:, :17, :], np.ones((keypoints.shape[0], 17, 1), dtype=np.float32)],
                    axis=-1,
                ).astype(np.float32)
            else:
                kp17 = np.zeros((bboxes.shape[0], 17, 3), dtype=np.float32)

            # Per-instance detection score = mean kp-confidence over the
            # 17 joints. BMP doesn't expose an aggregate "person score"
            # field directly, and keypoint-conf mean is a reasonable
            # proxy for "how confident is this detection as a whole".
            if kp17.shape[0] > 0:
                person_scores = kp17[..., 2].mean(axis=-1)
            else:
                person_scores = np.zeros(0, dtype=np.float32)

            if not logged_once:
                _logger.info(
                    "BMP first-frame: %d detections, masks shape %s "
                    "(dtype %s, range [%d, %d]), person_score range "
                    "[%.2f, %.2f]",
                    bboxes.shape[0], tuple(masks.shape), masks.dtype,
                    int(masks.min()) if masks.size > 0 else 0,
                    int(masks.max()) if masks.size > 0 else 0,
                    float(person_scores.min()) if person_scores.size > 0 else 0.0,
                    float(person_scores.max()) if person_scores.size > 0 else 0.0,
                )
                logged_once = True

            # Stash raw-per-frame-result for the YOLO11POSE replay model.
            # Keep ALL detections in the cache (even those below the
            # score_threshold below) — FastSAM3DBody does its own conf
            # filter with `conf=0.25`, so giving it the full candidate
            # set lets its aligner pick the best matches to mask bboxes.
            if bboxes.shape[0] > 0:
                replay_cache[t] = _BMPResult(bboxes, person_scores, kp17)

            # Apply our own score_threshold only for the MASK + tracking
            # pipeline. The YOLO11POSE cache above stays unfiltered.
            keep = person_scores >= float(score_threshold)
            if not keep.any():
                pbar.update(1)
                continue

            for i in np.where(keep)[0]:
                mask_bool = masks[i].astype(bool)
                if mask_bool.shape != (H, W):
                    continue
                if not mask_bool.any():
                    continue
                per_frame_detections[t].append({
                    "mask_packed": pack_mask(mask_bool),
                    "bbox":        bboxes[i].copy(),
                    "score":       float(person_scores[i]),
                    "kpts":        kp17[i].copy(),
                })

            pbar.update(1)

        infer_time = time.perf_counter() - t_infer

        # ------------------------------------------------------------------
        # Phase 2: greedy mask-IoU cross-frame tracking
        # ------------------------------------------------------------------
        # Same algorithm as SAM3ImageSegmentation / CrowdSAMInstance — match
        # current-frame masks against previous-frame tracks by IoU; any
        # unmatched current detection spawns a new track.
        next_track_id = 0
        per_frame_tracks = []  # per-frame list of (tid, packed_mask, score, kpts)
        prev_tracks: List = []

        for t in range(B):
            dets = per_frame_detections[t]
            if not dets:
                per_frame_tracks.append([])
                continue

            if not prev_tracks:
                frame_tracks = []
                for d in dets:
                    frame_tracks.append((
                        next_track_id, d["mask_packed"], d["score"], d["kpts"],
                    ))
                    next_track_id += 1
                per_frame_tracks.append(frame_tracks)
                prev_tracks = frame_tracks
                continue

            n_prev, n_curr = len(prev_tracks), len(dets)
            iou_matrix = np.zeros((n_prev, n_curr), dtype=np.float32)
            for i in range(n_prev):
                pb = prev_tracks[i][1]
                for j in range(n_curr):
                    iou_matrix[i, j] = _iou_packed(pb, dets[j]["mask_packed"])

            assigned_prev, assigned_curr = set(), set()
            frame_tracks = [None] * n_curr
            pairs = []
            for i in range(n_prev):
                for j in range(n_curr):
                    if iou_matrix[i, j] >= iou_threshold:
                        pairs.append((iou_matrix[i, j], i, j))
            pairs.sort(reverse=True)
            for _, i, j in pairs:
                if i in assigned_prev or j in assigned_curr:
                    continue
                tid = prev_tracks[i][0]
                frame_tracks[j] = (
                    tid, dets[j]["mask_packed"], dets[j]["score"], dets[j]["kpts"],
                )
                assigned_prev.add(i)
                assigned_curr.add(j)

            for j in range(n_curr):
                if frame_tracks[j] is None:
                    frame_tracks[j] = (
                        next_track_id,
                        dets[j]["mask_packed"],
                        dets[j]["score"],
                        dets[j]["kpts"],
                    )
                    next_track_id += 1

            per_frame_tracks.append(frame_tracks)
            prev_tracks = frame_tracks

        pbar.update(1)

        # ------------------------------------------------------------------
        # Phase 3: assemble output tensor
        # ------------------------------------------------------------------
        all_track_ids = sorted({
            tid for ft in per_frame_tracks for tid, _, _, _ in ft
        })
        n_total = len(all_track_ids)

        if n_total == 0:
            _logger.warning(
                "BMP: no detections survived score_threshold=%.2f. "
                "Returning empty masks + empty pose replay.",
                score_threshold,
            )
            # Return zero masks and a no-op pose replay — downstream
            # FastSAM3DBody will see no yolo_pose hits and fall back
            # to its body_decoder path.
            empty_replay = _BMPReplayModel([None] * B)
            return (
                torch.zeros(B, H, W),
                {"model": empty_replay, "_bmp_cache": replay_cache},
                images,
            )

        id_to_slot = {tid: i for i, tid in enumerate(all_track_ids)}

        masks_out = torch.zeros(B * n_total, H, W, dtype=torch.float32)
        for t in range(B):
            for tid, packed, _score, _kpts in per_frame_tracks[t]:
                slot = id_to_slot[tid]
                mask_bool = unpack_mask(packed, H, W)
                masks_out[t * n_total + slot] = torch.from_numpy(
                    mask_bool.astype(np.float32)
                )

        # YOLO11POSE-compatible output: a replay model over the raw
        # per-frame BMP results (unfiltered). FastSAM3DBody plugs this
        # into yolo_model = yolo11pose_dict["model"] and iterates.
        replay_model = _BMPReplayModel(replay_cache)
        bmp_pose_out = {
            "model":             replay_model,
            "checkpoint_path":   "<BMP replay - see LoadBMPNode>",
            "_bmp_cache":        replay_cache,
            "_n_tracks":         n_total,
        }

        # Debug overlay: masks + bboxes + skeletons
        if debug_overlay:
            per_frame_items_for_overlay = [
                [(tid, packed) for tid, packed, _s, _k in per_frame_tracks[t]]
                for t in range(B)
            ]
            overlay_out, legend = build_debug_overlay(
                images=images,
                per_frame_items=per_frame_items_for_overlay,
                H=H, W=W,
            )
            _logger.info("BMP debug overlay legend: %s", legend)

            # Add skeletons on top with matching colors
            per_frame_tids_with_slot = [
                [(tid, id_to_slot[tid], kpts)
                 for tid, _p, _s, kpts in per_frame_tracks[t]]
                for t in range(B)
            ]
            overlay_out = _draw_skeletons_on_overlay(
                overlay_out, per_frame_tids_with_slot, H, W,
            )
        else:
            overlay_out = images

        _logger.info(
            "BMP segmentation: %d frames, %d total tracks, inference "
            "%.2fs (%.2fs/frame), output masks %s, pose cache filled "
            "(%d frames with detections)",
            B, n_total, infer_time, infer_time / max(B, 1),
            tuple(masks_out.shape),
            sum(1 for r in replay_cache if r is not None),
        )

        return (masks_out, bmp_pose_out, overlay_out)
