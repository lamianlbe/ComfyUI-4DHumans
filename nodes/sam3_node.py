"""
SAM3 Video Segmentation nodes.

LoadSAM3 loads the SAM3 model.
SAM3VideoSegmentation runs text-prompted video segmentation and outputs
frame-aligned masks suitable for PromptHMR / Sapiens downstream nodes.

Output masks have shape (B * N, H, W) where B = number of frames and
N = number of detected persons.  Every frame has exactly N mask slots;
frames where a person is not visible get a zero mask.

The sam3 Python package is bundled in ``sam3/`` (copied from
ComfyUI-RMBG).  Model checkpoint (``sam3.pt``) is loaded from
ComfyUI's ``models/sam3/`` directory.
"""

import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
from folder_paths import models_dir

# --- SAM3 Python code lives in <repo>/sam3/ ---
# Internal imports use ``from sam3.model.xxx``, so we put the repo root
# (parent of the sam3 package) on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAM3_PKG = os.path.join(_REPO_ROOT, "sam3")
if _SAM3_PKG not in sys.path:
    sys.path.insert(0, _SAM3_PKG)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Checkpoint directory under ComfyUI models/
SAM3_CKPT_DIR = os.path.join(models_dir, "sam3")
os.makedirs(SAM3_CKPT_DIR, exist_ok=True)

# BPE vocabulary file bundled with the code
SAM3_BPE_PATH = Path(_SAM3_PKG) / "assets" / "bpe_simple_vocab_16e6.txt.gz"

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
    """Load a SAM3 video segmentation model.

    Checkpoint files should be placed in ``models/sam3/``.
    Compatible with the sam3.pt from ComfyUI-RMBG.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_list_checkpoints(),),
            },
        }

    RETURN_TYPES = ("SAM3",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, checkpoint):
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        path = os.path.join(SAM3_CKPT_DIR, checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {path}\n"
                f"Please place sam3.pt in {SAM3_CKPT_DIR}/\n"
                f"(download from https://huggingface.co/1038lab/sam3)"
            )

        kwargs = dict(checkpoint_path=path)
        if SAM3_BPE_PATH.is_file():
            kwargs["bpe_path"] = str(SAM3_BPE_PATH)

        predictor = Sam3VideoPredictor(**kwargs)
        _logger.info("Loaded SAM3 video model: %s", checkpoint)

        return ({"predictor": predictor},)


class SAM3VideoSegmentationNode:
    """Run SAM3 text-prompted video segmentation.

    Outputs aligned masks (B * N, H, W) where every frame has
    exactly N person slots.  Missing persons get zero masks.
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
                            "Text description of objects to segment "
                            "(e.g. 'person', 'dancer')."
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

        pbar = comfy.utils.ProgressBar(B + 2)

        # ------------------------------------------------------------------
        # Convert tensor frames to PIL Image list (SAM3 accepts this
        # directly via load_resource_as_video_frames, no temp files needed)
        # ------------------------------------------------------------------
        pil_frames = []
        for t in range(B):
            frame_np = (images[t] * 255).byte().cpu().numpy()  # (H, W, 3) RGB uint8
            pil_frames.append(Image.fromarray(frame_np))
        pbar.update(1)

        # ------------------------------------------------------------------
        # Start session, add text prompt, propagate
        # ------------------------------------------------------------------
        result = predictor.start_session(resource_path=pil_frames)
        session_id = result["session_id"]

        predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text=text_prompt,
        )
        pbar.update(1)

        # ------------------------------------------------------------------
        # Propagate through video and collect per-frame results
        # ------------------------------------------------------------------
        per_frame_masks = {}  # frame_idx -> {obj_id: mask_np}

        for result in predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="both",
            start_frame_idx=None,
            max_frame_num_to_track=None,
        ):
            frame_idx = result["frame_index"]
            outputs = result["outputs"]
            obj_ids = outputs["out_obj_ids"]          # (N,) int
            binary_masks = outputs["out_binary_masks"]  # (N, H, W) bool

            frame_dict = {}
            for i, oid in enumerate(obj_ids):
                frame_dict[int(oid)] = binary_masks[i]
            per_frame_masks[frame_idx] = frame_dict
            pbar.update(1)

        # Close session
        predictor.close_session(session_id)

        # ------------------------------------------------------------------
        # Alignment: ensure every frame has the same N person slots
        # ------------------------------------------------------------------
        all_obj_ids = sorted(
            set(oid for fd in per_frame_masks.values() for oid in fd)
        )
        n_persons = len(all_obj_ids)

        if n_persons == 0:
            _logger.warning(
                "SAM3 detected no objects with prompt '%s'. "
                "Returning single empty mask per frame.",
                text_prompt,
            )
            n_persons = 1
            return (torch.zeros(B, H, W),)

        id_to_idx = {oid: i for i, oid in enumerate(all_obj_ids)}

        masks_out = torch.zeros(B, n_persons, H, W)
        for t in range(B):
            fd = per_frame_masks.get(t, {})
            for oid, mask_np in fd.items():
                masks_out[t, id_to_idx[oid]] = torch.from_numpy(
                    mask_np.astype(np.float32)
                )

        # Reshape to (B * N, H, W) — frame-grouped ordering
        masks_out = masks_out.reshape(B * n_persons, H, W)

        _logger.info(
            "SAM3 segmentation: %d frames, %d persons, output shape %s",
            B, n_persons, tuple(masks_out.shape),
        )

        return (masks_out,)
