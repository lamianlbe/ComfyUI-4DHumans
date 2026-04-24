"""
SAM3 Image Segmentation node (Ultralytics backend).

Uses SAM3's image model for per-frame text-prompted segmentation with
IoU-based cross-frame tracking for consistent person/object identity.
The image model can be more accurate than the video model for certain
concepts (e.g. less common or specific text prompts) since it does a
fresh detection on every frame instead of relying on memory-based
tracking.

Output masks have shape (B * N, H, W) where B = number of frames and
N = total number of tracked objects.  Every frame has exactly N mask
slots; frames where an object is not visible get a zero mask.
"""

import logging

import numpy as np
import torch

import comfy.utils

from ._mask_utils import build_debug_overlay, pack_mask, unpack_mask

_logger = logging.getLogger(__name__)


def _get_or_build_image_predictor(sam3_dict):
    """Lazily build and cache a SAM3SemanticPredictor on the sam3 dict."""
    if sam3_dict.get("_image_predictor") is not None:
        return sam3_dict["_image_predictor"]

    try:
        from ultralytics.models.sam.predict import SAM3SemanticPredictor
    except ImportError as e:
        raise ImportError(
            "Ultralytics is required for SAM3. Install with:\n"
            "  pip install -U ultralytics"
        ) from e

    overrides = dict(
        conf=sam3_dict.get("conf", 0.25),
        task="segment",
        mode="predict",
        imgsz=sam3_dict.get("imgsz", 640),
        model=sam3_dict["checkpoint_path"],
        half=sam3_dict.get("half", True),
        save=False,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    sam3_dict["_image_predictor"] = predictor
    _logger.info(
        "Built SAM3 image predictor (imgsz=%d, half=%s)",
        overrides["imgsz"], overrides["half"],
    )
    return predictor


def _offload_sam3_to_cpu(sam3_dict):
    """Move SAM3 predictors to CPU and empty CUDA cache to free VRAM."""
    for key in ("predictor", "_image_predictor"):
        pred = sam3_dict.get(key)
        if pred is not None and hasattr(pred, "model") and pred.model is not None:
            try:
                pred.model.to("cpu")
            except Exception as e:
                _logger.warning(
                    "SAM3 VRAM offload failed for %s: %s", key, e
                )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _restore_predictor_to_cuda(predictor):
    """Move a SAM3 predictor's model back to CUDA before inference.

    We offload to CPU after each run; this puts it back before the next run.
    No-op when no CUDA is available.
    """
    if not torch.cuda.is_available():
        return
    try:
        if hasattr(predictor, "model") and predictor.model is not None:
            predictor.model.to("cuda")
    except Exception as e:
        _logger.warning("SAM3 restore-to-CUDA failed: %s", e)


class SAM3ImageSegmentationNode:
    """Run SAM3 image-model segmentation with cross-frame IoU tracking.

    For each frame we run the image model independently, then greedily
    match detections across frames via IoU to assign consistent track
    IDs.  Compared to SAM3 video model: usually higher accuracy per
    frame (no memory/tracker trade-offs) but no temporal consistency
    baked in (we rely on IoU).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sam3": ("SAM3",),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "person",
                        "tooltip": (
                            "Text description of objects to segment. "
                            "Use comma to segment multiple types "
                            "(e.g. 'person, dog')."
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
                            "Minimum IoU between frames to consider the "
                            "same object.  Lower = more lenient tracking."
                        ),
                    },
                ),
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Output a second IMAGE where each tracked "
                            "object's mask is color-coded and alpha-"
                            "blended onto the original frames, with a "
                            "matching bbox outline + tid text label. "
                            "Off by default (adds a CPU pass per "
                            "frame). When off, the debug_overlay "
                            "output is just the input images passed "
                            "through unchanged."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "debug_overlay")
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    def segment(self, images, sam3, text_prompt, iou_threshold,
                debug_overlay=False):
        predictor = _get_or_build_image_predictor(sam3)

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # Split comma-separated prompts
        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        if not prompts:
            prompts = ["person"]

        # Convert to HWC BGR uint8 numpy on CPU.  We deliberately do
        # NOT use torch tensor input:
        #   - ultralytics' LoadTensor skips letterbox, sending raw
        #     non-square tensors into SAM3's ViT backbone.
        #   - SAM3's RoPE freqs_cis is precomputed for square
        #     imgsz×imgsz input. Non-square trips a shape assert.
        # Additionally SAM's pre_transform asserts batch size == 1,
        # so we must feed one frame per predictor() call.
        rgb_u8 = (images.clamp(0, 1) * 255).byte().cpu().numpy()  # (B, H, W, 3) RGB
        frames_bgr = rgb_u8[..., ::-1].copy()                      # BGR

        pbar = comfy.utils.ProgressBar(B + 1)

        # How often (in frames) to call torch.cuda.empty_cache(). SAM3's
        # ViT-L activations can pile up otherwise; 16 is a conservative
        # default for 720p+ on a ~100 GB card.
        CACHE_FLUSH_EVERY = 16

        # ------------------------------------------------------------------
        # Per-frame detections
        # ------------------------------------------------------------------
        # per_frame_detections[t] = list of {mask, score, prompt_idx}
        per_frame_detections = [[] for _ in range(B)]

        try:
            # Ensure model is on GPU (previous run offloaded to CPU)
            _restore_predictor_to_cuda(predictor)

            for t in range(B):
                # Reset the backbone-feature cache on the predictor so
                # ultralytics recomputes features for this frame
                # (otherwise a stale RoPE-sized tensor from the previous
                # frame will be reused and trip the shape assert when
                # image sizes differ).
                if hasattr(predictor, "features"):
                    predictor.features = None

                # Pass a single-frame list; SAM3 requires batch size 1
                # and takes the LoadPilAndNumpy code path which runs the
                # full preprocess (letterbox + normalise + to-tensor).
                results_iter = predictor(
                    source=[frames_bgr[t]],
                    text=prompts,
                    stream=True,
                )

                for r in results_iter:
                    if r.masks is None or r.boxes is None:
                        del r
                        break  # only one frame per iter anyway

                    masks_data = r.masks.data  # (N, H', W')
                    scores = r.boxes.conf      # (N,)
                    classes = r.boxes.cls      # (N,)

                    if masks_data.shape[-2:] != (H, W):
                        masks_data = torch.nn.functional.interpolate(
                            masks_data.unsqueeze(1).float(),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(1)

                    # Bit-pack masks to cap RAM use: each (H, W) bool
                    # becomes ceil(H*W/8) bytes — 1/32 the float32 cost
                    # and 1/8 the plain-bool cost. Critical on long HD
                    # videos with many tracked persons.
                    masks_np = (masks_data > 0.5).cpu().numpy()   # bool
                    scores_np = scores.cpu().numpy().astype(np.float32)
                    classes_np = classes.cpu().numpy().astype(int)

                    order = np.argsort(-scores_np)
                    for idx in order:
                        per_frame_detections[t].append({
                            "mask_packed": pack_mask(masks_np[idx]),
                            "score": float(scores_np[idx]),
                            "prompt_idx": int(classes_np[idx]),
                        })

                    del masks_data, scores, classes, r
                    break  # SAM3 gives one Results per single-frame call

                del results_iter
                pbar.update(1)

                if torch.cuda.is_available() and ((t + 1) % CACHE_FLUSH_EVERY == 0):
                    torch.cuda.empty_cache()
        finally:
            # Free VRAM so downstream nodes (PromptHMR + Sapiens) don't OOM
            _offload_sam3_to_cpu(sam3)

        # ------------------------------------------------------------------
        # Cross-frame IoU tracking (greedy)
        # ------------------------------------------------------------------
        next_track_id = 0
        per_frame_tracks = []  # per_frame_tracks[t] = [(track_id, mask), ...]
        prev_tracks = []

        for t in range(B):
            detections = per_frame_detections[t]
            if not detections:
                per_frame_tracks.append([])
                continue

            current_packed = [d["mask_packed"] for d in detections]
            current_prompts = [d["prompt_idx"] for d in detections]

            if not prev_tracks:
                # First frame with detections: assign new track IDs
                frame_tracks = []
                for i, packed in enumerate(current_packed):
                    frame_tracks.append(
                        (next_track_id, packed, current_prompts[i])
                    )
                    next_track_id += 1
                per_frame_tracks.append(frame_tracks)
                prev_tracks = frame_tracks
                continue

            # Compute IoU matrix between prev_tracks and current detections.
            # IoU works directly on packed uint8 arrays:
            #   popcount(A & B) / popcount(A | B)
            # No need to unpack into full H×W buffers — saves both
            # memory and time.
            n_prev = len(prev_tracks)
            n_curr = len(current_packed)
            iou_matrix = np.zeros((n_prev, n_curr), dtype=np.float32)

            def _popcount_u8(a):
                return np.unpackbits(a).sum()

            for i in range(n_prev):
                pb = prev_tracks[i][1]
                for j in range(n_curr):
                    if prev_tracks[i][2] != current_prompts[j]:
                        continue
                    cb = current_packed[j]
                    inter = _popcount_u8(pb & cb)
                    union = _popcount_u8(pb | cb)
                    iou_matrix[i, j] = (
                        inter / union if union > 0 else 0.0
                    )

            # Greedy matching: highest IoU pairs first
            assigned_prev = set()
            assigned_curr = set()
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
                    track_id, current_packed[j], current_prompts[j]
                )
                assigned_prev.add(i)
                assigned_curr.add(j)

            # Unmatched detections get new track IDs
            for j in range(n_curr):
                if frame_tracks[j] is None:
                    frame_tracks[j] = (
                        next_track_id,
                        current_packed[j],
                        current_prompts[j],
                    )
                    next_track_id += 1

            per_frame_tracks.append(frame_tracks)
            prev_tracks = frame_tracks

        pbar.update(1)

        # ------------------------------------------------------------------
        # Build aligned output
        # ------------------------------------------------------------------
        all_track_ids = sorted({
            tid
            for frame_tracks in per_frame_tracks
            for tid, _, _ in frame_tracks
        })
        n_total = len(all_track_ids)

        if n_total == 0:
            _logger.warning(
                "SAM3 Image Segmentation: no objects detected for %s. "
                "Returning single empty mask per frame.",
                prompts,
            )
            # Pass-through for the debug output so downstream graph
            # wiring stays valid when nothing is detected.
            return (torch.zeros(B, H, W), images)

        id_to_slot = {tid: i for i, tid in enumerate(all_track_ids)}

        # Allocate the final float32 tensor once and fill in place by
        # unpacking each mask. Avoids holding a bool and a float32
        # copy simultaneously. All packed masks released right after
        # unpacking so peak RAM ≈ sizeof(masks_out).
        masks_out = torch.zeros(B * n_total, H, W, dtype=torch.float32)
        for t in range(B):
            for track_id, packed, _ in per_frame_tracks[t]:
                slot = id_to_slot[track_id]
                mask_bool = unpack_mask(packed, H, W)
                masks_out[t * n_total + slot] = torch.from_numpy(
                    mask_bool.astype(np.float32)
                )

        # Build the colored-mask overlay before dropping per_frame_tracks.
        # Flatten each frame's ``(tid, packed, prompt_idx)`` triples down
        # to ``(tid, packed)`` pairs for the shared helper. When the
        # toggle is off, pass the input images through unchanged.
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
            _logger.info("SAM3 image debug overlay legend: %s", legend)
        else:
            overlay_out = images

        del per_frame_tracks, per_frame_detections

        _logger.info(
            "SAM3 Image Segmentation (ultralytics): %d frames, %d prompts, "
            "%d total tracks, output shape %s",
            B, len(prompts), n_total, tuple(masks_out.shape),
        )

        return (masks_out, overlay_out)
