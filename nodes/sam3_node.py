"""
SAM3 Video Segmentation nodes (Ultralytics backend).

LoadSAM3 loads the SAM3 model via Ultralytics' SAM3VideoSemanticPredictor.
SAM3VideoSegmentation runs text-prompted video segmentation and outputs
frame-aligned masks suitable for PromptHMR / Sapiens downstream nodes.

Output masks have shape (B * N, H, W) where B = number of frames and
N = number of tracked persons across the whole video.  Every frame has
exactly N mask slots; frames where a person is not visible get a zero
mask.

Model checkpoint (``sam3.pt``) is loaded from ``models/sam3/``.
"""

import glob
import logging
import os

import numpy as np
import torch

import comfy.utils
from folder_paths import models_dir

# Checkpoint directory under ComfyUI models/
SAM3_CKPT_DIR = os.path.join(models_dir, "sam3")
os.makedirs(SAM3_CKPT_DIR, exist_ok=True)

_logger = logging.getLogger(__name__)


def _list_checkpoints():
    """Return basenames of available SAM3 checkpoint files."""
    patterns = ["*.pt", "*.pth", "*.safetensors"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(SAM3_CKPT_DIR, pat)))
    basenames = sorted(set(os.path.basename(f) for f in files))
    if not basenames:
        basenames = ["(no checkpoints found)"]
    return basenames


class LoadSAM3Node:
    """Load a SAM3 video segmentation model (Ultralytics backend).

    Checkpoint files should be placed in ``models/sam3/``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_list_checkpoints(),),
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 256,
                        "max": 1024,
                        "step": 32,
                        "tooltip": (
                            "Inference resolution. Lower = faster but less "
                            "accurate. SAM3 is internally fixed at 1008 in "
                            "the official code; ultralytics exposes this."
                        ),
                    },
                ),
                "half": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use FP16 inference (faster on modern GPUs).",
                    },
                ),
                "conf": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Detection confidence threshold.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAM3",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, checkpoint, imgsz, half, conf):
        try:
            from ultralytics.models.sam.predict import (
                SAM3VideoSemanticPredictor,
            )
        except ImportError as e:
            raise ImportError(
                "Ultralytics is required for SAM3. Install with:\n"
                "  pip install -U ultralytics"
            ) from e

        path = os.path.join(SAM3_CKPT_DIR, checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {path}\n"
                f"Please place sam3.pt in {SAM3_CKPT_DIR}/"
            )

        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            imgsz=imgsz,
            model=path,
            half=half,
            save=False,
            verbose=False,
        )
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        _logger.info(
            "Loaded SAM3 (ultralytics): %s (imgsz=%d, half=%s)",
            checkpoint, imgsz, half,
        )
        return ({"predictor": predictor, "imgsz": imgsz, "half": half},)


class SAM3VideoSegmentationNode:
    """Run SAM3 text-prompted video segmentation (Ultralytics backend).

    Outputs aligned masks (B * N, H, W) where every frame has exactly
    N object slots.  Uses ultralytics' built-in tracking IDs for
    cross-frame alignment — no manual slot mapping needed.
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
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    def segment(self, images, sam3, text_prompt):
        predictor = sam3["predictor"]

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # Split comma-separated prompts
        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        if not prompts:
            prompts = ["person"]

        # Convert to BCHW RGB float32 [0, 1] for ultralytics
        source = images.permute(0, 3, 1, 2).contiguous().float()

        pbar = comfy.utils.ProgressBar(B)

        # ------------------------------------------------------------------
        # Run inference streaming.  Ultralytics' SAM3 video predictor
        # returns Results with persistent tracking IDs via r.boxes.id.
        # ------------------------------------------------------------------
        results_iter = predictor(
            source=source,
            text=prompts,
            stream=True,
        )

        # Collect per-frame masks indexed by global track_id.
        # per_frame[t] = {track_id: mask_np}
        per_frame = [{} for _ in range(B)]
        all_track_ids = set()

        frame_idx = 0
        for r in results_iter:
            if frame_idx >= B:
                break

            if r.masks is not None and r.boxes is not None:
                masks_data = r.masks.data  # (N, H', W') torch tensor
                ids = r.boxes.id           # (N,) tensor or None

                if ids is None:
                    # No tracking IDs — fall back to per-frame indices
                    ids = torch.arange(masks_data.shape[0])

                # Resize masks back to original (H, W) if needed
                if masks_data.shape[-2:] != (H, W):
                    masks_data = torch.nn.functional.interpolate(
                        masks_data.unsqueeze(1).float(),
                        size=(H, W),
                        mode="nearest",
                    ).squeeze(1)

                masks_np = masks_data.cpu().numpy().astype(np.float32)
                ids_np = ids.cpu().numpy().astype(int)

                for i, tid in enumerate(ids_np):
                    per_frame[frame_idx][int(tid)] = masks_np[i]
                    all_track_ids.add(int(tid))

            frame_idx += 1
            pbar.update(1)

        if not all_track_ids:
            _logger.warning(
                "SAM3 detected nothing for prompts %s. "
                "Returning single empty mask per frame.",
                prompts,
            )
            return (torch.zeros(B, H, W),)

        # Map track IDs → contiguous slot indices (0, 1, 2, ...)
        sorted_ids = sorted(all_track_ids)
        id_to_slot = {tid: i for i, tid in enumerate(sorted_ids)}
        n_persons = len(sorted_ids)

        # Build aligned output tensor
        masks_out = torch.zeros(B, n_persons, H, W)
        for t in range(B):
            for tid, mask_np in per_frame[t].items():
                slot = id_to_slot[tid]
                masks_out[t, slot] = torch.from_numpy(mask_np)

        # Reshape to (B * N, H, W) — frame-grouped ordering
        masks_out = masks_out.reshape(B * n_persons, H, W)

        _logger.info(
            "SAM3 (ultralytics): %d frames, %d prompts, %d tracked objects, "
            "output shape %s",
            B, len(prompts), n_persons, tuple(masks_out.shape),
        )

        return (masks_out,)
