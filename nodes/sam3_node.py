"""
SAM3 Video Segmentation nodes.

LoadSAM3 loads the SAM3 model.
SAM3VideoSegmentation runs text-prompted video segmentation and outputs
frame-aligned masks suitable for PromptHMR / Sapiens downstream nodes.

Output masks have shape (B * N, H, W) where B = number of frames and
N = number of detected persons.  Every frame has exactly N mask slots;
frames where a person is not visible get a zero mask.

Compatible with the checkpoint from ComfyUI-RMBG (``models/sam3/sam3.pt``).
The same ``sam3.pt`` works for both image and video modes.
"""

import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch

import comfy.model_management
import comfy.utils
from folder_paths import models_dir

SAM3_DIR = os.path.join(models_dir, "sam3")
os.makedirs(SAM3_DIR, exist_ok=True)

# BPE vocabulary file required by SAM3's text encoder.
# ComfyUI-RMBG stores it at models/sam3/assets/bpe_simple_vocab_16e6.txt.gz
SAM3_BPE_PATH = Path(SAM3_DIR) / "assets" / "bpe_simple_vocab_16e6.txt.gz"

_logger = logging.getLogger(__name__)


def _list_checkpoints():
    """Return basenames of available SAM3 checkpoint files."""
    patterns = ["*.pt", "*.pth", "*.safetensors"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(SAM3_DIR, pat)))
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
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError:
            raise ImportError(
                "SAM3 is not installed. Please install it with:\n"
                "  pip install sam3\n"
                "or follow the instructions at "
                "https://github.com/facebookresearch/sam3"
            )

        path = os.path.join(SAM3_DIR, checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {path}\n"
                f"Please place model files in {SAM3_DIR}/"
            )

        device = comfy.model_management.get_torch_device()
        device_str = str(device)

        # Build video predictor.  Pass bpe_path if the BPE vocab file
        # exists (installed by ComfyUI-RMBG in models/sam3/assets/).
        kwargs = dict(
            checkpoint_path=path,
            load_from_HF=False,
            device=device_str,
        )
        if SAM3_BPE_PATH.is_file():
            kwargs["bpe_path"] = str(SAM3_BPE_PATH)

        predictor = build_sam3_video_predictor(**kwargs)
        _logger.info("Loaded SAM3 video model: %s", checkpoint)

        return ({"predictor": predictor, "device": device},)


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

        pbar = comfy.utils.ProgressBar(B + 1)

        # Convert to uint8 numpy frames for SAM3
        frames_np = (images * 255).byte().cpu().numpy()  # (B, H, W, 3)

        # ------------------------------------------------------------------
        # Initialise SAM3 state and add text prompt
        # ------------------------------------------------------------------
        state = predictor.init_state(frames_np)
        predictor.add_new_text(
            inference_state=state,
            frame_idx=0,
            text=text_prompt,
        )
        pbar.update(1)

        # ------------------------------------------------------------------
        # Propagate through video and collect per-frame results
        # ------------------------------------------------------------------
        per_frame_masks = {}  # frame_idx -> {obj_id: mask_np}

        for frame_idx, obj_ids, masks_logits in predictor.propagate_in_video(
            inference_state=state,
        ):
            masks_bin = (masks_logits > 0.0).squeeze(1).cpu().numpy()  # (N, H, W)
            frame_dict = {}
            for i, oid in enumerate(obj_ids):
                oid_int = int(oid)
                frame_dict[oid_int] = masks_bin[i]
            per_frame_masks[frame_idx] = frame_dict
            pbar.update(1)

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
