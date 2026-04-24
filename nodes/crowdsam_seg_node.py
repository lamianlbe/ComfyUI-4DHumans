"""
CrowdSAM Instance Segmentation node.

CrowdSAM is a single-image segmenter — no temporal tracking. We run
it per-frame and then stitch detections into persistent track IDs via
greedy IoU matching against the previous frame (same algorithm used
by ``SAM3ImageSegmentation``). Output layout matches the other
segmentation nodes (``(B*N, H, W)`` frame-grouped), so downstream
pose / 3D-body nodes don't care which segmenter produced the masks.

We pick this over YOLO-seg / SAM3 when the scene has **heavy
occlusion / dense crowds** — CrowdSAM is trained exactly for that
regime (CrowdHuman benchmark, 78.4% AP with just 10 support images)
and its DINOv2-conditioned point prompts tend to separate touching
people better than class-agnostic mask decoders.
"""

import logging
import time

import numpy as np
import torch

import comfy.utils

from ._mask_utils import build_debug_overlay, pack_mask, unpack_mask

_logger = logging.getLogger(__name__)


def _coco_decode_rle_np(encoded_rle, h, w):
    """Decode a CrowdSAM RLE into an (H, W) bool numpy array.

    CrowdSAM returns ``{'counts': str, 'size': [h, w]}`` — the utf-8
    counts string comes from pycocotools' ``encode``, so we use
    ``pycocotools.mask.decode`` to get back a ``(H, W)`` uint8 array.
    """
    from pycocotools import mask as _cocomask  # noqa

    rle = {
        "counts": encoded_rle["counts"].encode("utf-8")
        if isinstance(encoded_rle["counts"], str)
        else encoded_rle["counts"],
        "size": encoded_rle["size"],
    }
    m = _cocomask.decode(rle).astype(bool)
    # pycocotools can return shape (H, W) or (H, W, 1); collapse.
    if m.ndim == 3:
        m = m[..., 0]
    return m


def _iou_packed(a_packed: np.ndarray, b_packed: np.ndarray) -> float:
    """IoU between two bit-packed bool masks (as returned by
    ``pack_mask``). Works on the packed uint8 arrays directly via
    popcount, so we don't have to unpack full H×W buffers just to
    compare — saves both time and memory."""
    if a_packed.size == 0 or b_packed.size == 0:
        return 0.0
    # np.unpackbits is vectorised and much faster than per-bit loops.
    inter = int(np.unpackbits(a_packed & b_packed).sum())
    if inter == 0:
        return 0.0
    union = int(np.unpackbits(a_packed | b_packed).sum())
    return inter / union if union > 0 else 0.0


class CrowdSAMInstanceSegmentationNode:
    """Run CrowdSAM per-frame + greedy IoU cross-frame tracking."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":   ("IMAGE",),
                "crowdsam": ("CROWDSAM",),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Final per-detection score filter. "
                            "CrowdSAM's upstream demo uses 0.3; raise "
                            "for cleaner output, lower to rescue "
                            "weak detections in very dense crowds."
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
                            "Minimum IoU with the previous frame to "
                            "consider the SAME person. Lower = more "
                            "lenient re-identification; higher = "
                            "stricter, more likely to spawn new track "
                            "IDs when a person moves fast."
                        ),
                    },
                ),
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Output a second IMAGE where each tracked "
                            "person's mask is color-coded and alpha-"
                            "blended onto the original frames, with a "
                            "matching bbox outline + tid text label. "
                            "Off by default (adds a CPU pass per "
                            "frame). When off, debug_overlay is the "
                            "input images passed through unchanged."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "debug_overlay")
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    def segment(self, images, crowdsam, score_threshold, iou_threshold,
                debug_overlay=False):
        model = crowdsam["model"]

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # ------------------------------------------------------------------
        # Per-frame inference
        # ------------------------------------------------------------------
        # CrowdSAM's generate() wants a (H, W, 3) uint8 RGB numpy array.
        # Pre-convert the whole batch once on CPU — it's uint8 so cost
        # is modest (B * H * W * 3 bytes), and keeping the model's
        # input pipeline outside our loop makes the timing log clean.
        rgb_u8 = (images.clamp(0, 1) * 255).byte().cpu().numpy()  # (B, H, W, 3) RGB

        pbar = comfy.utils.ProgressBar(B + 1)
        t_infer = time.perf_counter()
        per_frame_detections = [[] for _ in range(B)]
        logged_shape = False

        for t in range(B):
            frame = rgb_u8[t]  # (H, W, 3) uint8 RGB
            try:
                result = model.generate(frame)
            except Exception as e:
                _logger.error("CrowdSAM failed on frame %d: %s", t, e)
                pbar.update(1)
                continue

            # ``MaskData`` implements __getitem__ + items() but no .get();
            # flatten to a plain dict so the defaults below work even when
            # upstream didn't populate a key at all.
            result_dict = {k: v for k, v in result.items()}
            boxes  = np.asarray(result_dict.get("boxes",  np.zeros((0, 4))), dtype=np.float32)
            scores = np.asarray(result_dict.get("scores", np.zeros((0,))),   dtype=np.float32)
            rles   = list(result_dict.get("rles", []))

            if not logged_shape and len(rles) > 0:
                _logger.info(
                    "CrowdSAM first-frame output: %d detections before "
                    "score filter (score range %.2f..%.2f)",
                    len(rles),
                    float(scores.min()) if scores.size > 0 else 0.0,
                    float(scores.max()) if scores.size > 0 else 0.0,
                )
                logged_shape = True

            # Filter by score threshold before decoding masks (saves
            # time since RLE decode is non-trivial for big masks).
            keep = scores >= float(score_threshold)
            if not keep.any():
                pbar.update(1)
                continue

            kept_rles   = [rles[i]   for i in np.where(keep)[0]]
            kept_scores = scores[keep]
            kept_boxes  = boxes[keep] if boxes.shape[0] == len(rles) else None

            for i, rle in enumerate(kept_rles):
                mask_bool = _coco_decode_rle_np(rle, H, W)
                if mask_bool.shape != (H, W):
                    # CrowdSAM should return masks at the original image
                    # size, but guard against config mismatches.
                    continue
                per_frame_detections[t].append({
                    "mask_packed": pack_mask(mask_bool),
                    "score":       float(kept_scores[i]),
                    "box":         kept_boxes[i] if kept_boxes is not None else None,
                })

            pbar.update(1)

        infer_time = time.perf_counter() - t_infer

        # ------------------------------------------------------------------
        # Greedy IoU cross-frame tracking (same algorithm as SAM3 image
        # seg). One strong advantage here: CrowdSAM gives clean per-
        # instance masks, so IoU matching at the MASK level (not bbox)
        # is very discriminative even for overlapping people.
        # ------------------------------------------------------------------
        next_track_id = 0
        per_frame_tracks: list = []   # list[t] of list[(tid, packed_mask, score)]
        prev_tracks: list = []

        for t in range(B):
            detections = per_frame_detections[t]
            if not detections:
                per_frame_tracks.append([])
                continue

            if not prev_tracks:
                frame_tracks = []
                for d in detections:
                    frame_tracks.append(
                        (next_track_id, d["mask_packed"], d["score"])
                    )
                    next_track_id += 1
                per_frame_tracks.append(frame_tracks)
                prev_tracks = frame_tracks
                continue

            n_prev = len(prev_tracks)
            n_curr = len(detections)
            iou_matrix = np.zeros((n_prev, n_curr), dtype=np.float32)
            for i in range(n_prev):
                pb = prev_tracks[i][1]
                for j in range(n_curr):
                    iou_matrix[i, j] = _iou_packed(pb, detections[j]["mask_packed"])

            # Greedy: highest IoU pairs first, locked 1-to-1.
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
                track_id = prev_tracks[i][0]
                frame_tracks[j] = (
                    track_id, detections[j]["mask_packed"], detections[j]["score"]
                )
                assigned_prev.add(i)
                assigned_curr.add(j)

            # Unmatched current detections → new track IDs
            for j in range(n_curr):
                if frame_tracks[j] is None:
                    frame_tracks[j] = (
                        next_track_id,
                        detections[j]["mask_packed"],
                        detections[j]["score"],
                    )
                    next_track_id += 1

            per_frame_tracks.append(frame_tracks)
            prev_tracks = frame_tracks

        pbar.update(1)

        # ------------------------------------------------------------------
        # Assemble output tensor (B * n_total, H, W) float32
        # ------------------------------------------------------------------
        all_track_ids = sorted({
            tid for ft in per_frame_tracks for tid, _, _ in ft
        })
        n_total = len(all_track_ids)

        if n_total == 0:
            _logger.warning(
                "CrowdSAM: no detections passed score_threshold=%.2f. "
                "Returning single empty mask per frame.",
                score_threshold,
            )
            return (torch.zeros(B, H, W), images)

        id_to_slot = {tid: i for i, tid in enumerate(all_track_ids)}

        masks_out = torch.zeros(B * n_total, H, W, dtype=torch.float32)
        for t in range(B):
            for tid, packed, _ in per_frame_tracks[t]:
                slot = id_to_slot[tid]
                mask_bool = unpack_mask(packed, H, W)
                masks_out[t * n_total + slot] = torch.from_numpy(
                    mask_bool.astype(np.float32)
                )

        # Debug overlay — bundle (tid, packed) into the format the
        # shared helper expects, then reuse the same palette / bbox /
        # text renderer the other seg nodes use.
        if debug_overlay:
            per_frame_items = [
                [(tid, packed) for tid, packed, _ in per_frame_tracks[t]]
                for t in range(B)
            ]
            overlay_out, legend = build_debug_overlay(
                images=images,
                per_frame_items=per_frame_items,
                H=H, W=W,
            )
            _logger.info("CrowdSAM debug overlay legend: %s", legend)
        else:
            overlay_out = images

        _logger.info(
            "CrowdSAM segmentation: %d frames, %d total tracks, "
            "inference %.2fs (%.2fs/frame), output shape %s",
            B, n_total, infer_time, infer_time / max(B, 1),
            tuple(masks_out.shape),
        )

        return (masks_out, overlay_out)
