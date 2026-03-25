"""
SAM3 Image Segmentation node.

Uses SAM3's image model (same as RMBG) for per-frame text-prompted
segmentation with IoU-based cross-frame tracking for consistent
person/object identity.

Output masks have shape (B * N, H, W) where B = number of frames and
N = total number of tracked objects.  Every frame has exactly N mask
slots; frames where an object is not visible get a zero mask.
"""

import logging

import numpy as np
import torch
from PIL import Image
from contextlib import nullcontext

import comfy.model_management
import comfy.utils

_logger = logging.getLogger(__name__)


class SAM3ImageSegmentationNode:
    """Run SAM3 image-model segmentation with cross-frame tracking.

    Uses the same image model as RMBG's SAM3 node for better text
    understanding.  Cross-frame identity is maintained via IoU matching.
    Supports comma-separated prompts.
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
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.05,
                        "max": 0.95,
                        "step": 0.01,
                        "tooltip": "Minimum confidence to keep a detection.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    def segment(self, images, sam3, text_prompt, confidence_threshold):
        from sam3.model.sam3_image_processor import Sam3Processor

        predictor = sam3["predictor"]
        # The predictor wraps a video model; we need the underlying model
        # to build an image-model processor.  However build_sam3_image_model
        # creates a *different* architecture.  Instead, we use the same
        # model_builder that RMBG uses.
        #
        # Since LoadSAM3 stores the full predictor, we need to build an
        # image model separately.  We cache it on the predictor object.
        if not hasattr(predictor, "_image_model"):
            from sam3.model_builder import build_sam3_image_model
            import os, sys
            from pathlib import Path

            _REPO_ROOT = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            _SAM3_PKG = os.path.join(_REPO_ROOT, "sam3")
            bpe_path = str(
                Path(_SAM3_PKG) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            )

            # Find the checkpoint path from the video predictor's state
            # We need to rebuild the image model from the same checkpoint
            from folder_paths import models_dir
            import glob

            ckpt_dir = os.path.join(models_dir, "sam3")
            ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
            if not ckpts:
                raise FileNotFoundError(
                    f"No SAM3 checkpoint found in {ckpt_dir}"
                )
            ckpt_path = ckpts[0]

            _logger.info("Building SAM3 image model from %s", ckpt_path)
            image_model = build_sam3_image_model(
                bpe_path=bpe_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                eval_mode=True,
                checkpoint_path=ckpt_path,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            predictor._image_model = image_model

        image_model = predictor._image_model
        device = next(image_model.parameters()).device
        device_str = "cuda" if device.type == "cuda" else "cpu"
        processor = Sam3Processor(
            image_model, device=device_str,
            confidence_threshold=confidence_threshold,
        )

        autocast_ctx = (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # Split comma-separated prompts
        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        if not prompts:
            prompts = ["person"]

        pbar = comfy.utils.ProgressBar(B * len(prompts) + 1)

        # ------------------------------------------------------------------
        # Per-frame segmentation
        # ------------------------------------------------------------------
        # per_frame[t] = list of (mask_np, prompt_idx) for each detection
        per_frame_detections = []

        for t in range(B):
            frame_np = (images[t] * 255).byte().cpu().numpy()
            pil_img = Image.fromarray(frame_np)

            frame_masks = []
            for pi, prompt_text in enumerate(prompts):
                with autocast_ctx:
                    state = processor.set_image(pil_img)
                    processor.reset_all_prompts(state)
                    processor.set_confidence_threshold(
                        confidence_threshold, state
                    )
                    state = processor.set_text_prompt(prompt_text, state)

                masks = state.get("masks")
                scores = state.get("scores")

                if masks is not None and masks.numel() > 0:
                    masks_np = masks.float().cpu().numpy()
                    if masks_np.ndim == 4:
                        masks_np = masks_np.squeeze(1)  # (N, H, W)
                    scores_np = (
                        scores.float().cpu().numpy()
                        if scores is not None
                        else np.ones(masks_np.shape[0])
                    )

                    # Sort by score descending
                    order = np.argsort(-scores_np)
                    for idx in order:
                        frame_masks.append({
                            "mask": masks_np[idx],
                            "prompt_idx": pi,
                            "score": float(scores_np[idx]),
                        })

                pbar.update(1)

            per_frame_detections.append(frame_masks)

        # ------------------------------------------------------------------
        # Cross-frame IoU tracking
        # ------------------------------------------------------------------
        # Assign consistent track IDs across frames using greedy IoU matching
        next_track_id = 0
        # per_frame_tracks[t] = list of (track_id, mask_np)
        per_frame_tracks = []
        # prev_tracks: list of (track_id, mask_np) from previous frame
        prev_tracks = []

        for t in range(B):
            detections = per_frame_detections[t]
            if not detections:
                per_frame_tracks.append([])
                # Keep prev_tracks unchanged for next frame matching
                continue

            current_masks = [d["mask"] for d in detections]

            if not prev_tracks:
                # First frame with detections: assign new track IDs
                frame_tracks = []
                for d in detections:
                    frame_tracks.append((next_track_id, d["mask"]))
                    next_track_id += 1
                per_frame_tracks.append(frame_tracks)
                prev_tracks = frame_tracks
                continue

            # Compute IoU matrix between prev_tracks and current detections
            n_prev = len(prev_tracks)
            n_curr = len(current_masks)
            iou_matrix = np.zeros((n_prev, n_curr), dtype=np.float32)

            for i, (_, prev_mask) in enumerate(prev_tracks):
                prev_bool = prev_mask > 0.5
                for j, curr_mask in enumerate(current_masks):
                    curr_bool = curr_mask > 0.5
                    intersection = np.logical_and(prev_bool, curr_bool).sum()
                    union = np.logical_or(prev_bool, curr_bool).sum()
                    iou_matrix[i, j] = (
                        intersection / union if union > 0 else 0.0
                    )

            # Greedy matching: pick highest IoU pairs
            assigned_curr = set()
            assigned_prev = set()
            frame_tracks = [None] * n_curr

            # Sort all (prev, curr) pairs by IoU descending
            pairs = []
            for i in range(n_prev):
                for j in range(n_curr):
                    if iou_matrix[i, j] > 0.1:  # minimum IoU threshold
                        pairs.append((iou_matrix[i, j], i, j))
            pairs.sort(reverse=True)

            for iou_val, i, j in pairs:
                if i in assigned_prev or j in assigned_curr:
                    continue
                track_id = prev_tracks[i][0]
                frame_tracks[j] = (track_id, current_masks[j])
                assigned_prev.add(i)
                assigned_curr.add(j)

            # Unmatched detections get new track IDs
            for j in range(n_curr):
                if frame_tracks[j] is None:
                    frame_tracks[j] = (next_track_id, current_masks[j])
                    next_track_id += 1

            per_frame_tracks.append(frame_tracks)
            prev_tracks = frame_tracks

        pbar.update(1)

        # ------------------------------------------------------------------
        # Build aligned output
        # ------------------------------------------------------------------
        all_track_ids = sorted(set(
            tid for frame_tracks in per_frame_tracks
            for tid, _ in frame_tracks
        ))
        n_total = len(all_track_ids)

        if n_total == 0:
            _logger.warning(
                "SAM3 Image Segmentation: no objects detected. "
                "Returning single empty mask per frame."
            )
            return (torch.zeros(B, H, W),)

        id_to_slot = {tid: i for i, tid in enumerate(all_track_ids)}

        masks_out = torch.zeros(B, n_total, H, W)
        for t in range(B):
            for track_id, mask_np in per_frame_tracks[t]:
                slot = id_to_slot[track_id]
                masks_out[t, slot] = torch.from_numpy(
                    mask_np.astype(np.float32)
                )

        masks_out = masks_out.reshape(B * n_total, H, W)

        _logger.info(
            "SAM3 Image Segmentation: %d frames, %d prompts, "
            "%d total tracks, output shape %s",
            B, len(prompts), n_total, tuple(masks_out.shape),
        )

        return (masks_out,)
