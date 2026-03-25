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

    def _run_single_prompt(self, predictor, pil_frames, B, H, W, text, autocast_ctx):
        """Run segmentation for a single text prompt.

        Returns
        -------
        per_frame_masks : dict[int, dict[int, ndarray]]
            frame_idx -> {obj_id: binary_mask}
        """
        with autocast_ctx:
            result = predictor.start_session(resource_path=pil_frames)
            session_id = result["session_id"]

            predictor.add_prompt(
                session_id=session_id,
                frame_idx=0,
                text=text,
            )

        per_frame_masks = {}
        with autocast_ctx:
            propagation_results = list(predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="both",
                start_frame_idx=None,
                max_frame_num_to_track=None,
            ))

        for res in propagation_results:
            frame_idx = res["frame_index"]
            outputs = res["outputs"]
            obj_ids = outputs["out_obj_ids"]
            binary_masks = outputs["out_binary_masks"]

            frame_dict = {}
            for i, oid in enumerate(obj_ids):
                frame_dict[int(oid)] = binary_masks[i]
            per_frame_masks[frame_idx] = frame_dict

        predictor.close_session(session_id)
        return per_frame_masks

    def segment(self, images, sam3, text_prompt):
        import torch
        predictor = sam3["predictor"]

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # Split comma-separated prompts
        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        if not prompts:
            prompts = ["person"]
        n_prompts = len(prompts)

        pbar = comfy.utils.ProgressBar(B + n_prompts * 2)

        # ------------------------------------------------------------------
        # Convert tensor frames to PIL Image list
        # ------------------------------------------------------------------
        pil_frames = []
        for t in range(B):
            frame_np = (images[t] * 255).byte().cpu().numpy()
            pil_frames.append(Image.fromarray(frame_np))
        pbar.update(1)

        # ------------------------------------------------------------------
        # Setup autocast
        # ------------------------------------------------------------------
        from contextlib import nullcontext
        device = next(predictor.model.parameters()).device
        autocast_ctx = (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        # ------------------------------------------------------------------
        # Run segmentation for each prompt separately
        # ------------------------------------------------------------------
        # Collect all masks with globally unique slot indices.
        # Each prompt gets its own set of obj_ids; we remap them to a
        # contiguous global index so that masks from different prompts
        # occupy distinct slots.
        global_slot_count = 0
        # global_frame_masks[t] = {global_slot: mask_np}
        global_frame_masks = {t: {} for t in range(B)}

        for pi, prompt_text in enumerate(prompts):
            _logger.info("SAM3 prompt %d/%d: '%s'", pi + 1, n_prompts, prompt_text)

            pfm = self._run_single_prompt(
                predictor, pil_frames, B, H, W, prompt_text, autocast_ctx
            )
            pbar.update(1)

            # Collect all obj_ids from this prompt's results
            prompt_obj_ids = sorted(
                set(oid for fd in pfm.values() for oid in fd)
            )

            if not prompt_obj_ids:
                _logger.warning(
                    "SAM3 detected no objects for prompt '%s'", prompt_text
                )
                pbar.update(1)
                continue

            # Map this prompt's obj_ids to global slots
            local_to_global = {}
            for oid in prompt_obj_ids:
                local_to_global[oid] = global_slot_count
                global_slot_count += 1

            # Merge into global_frame_masks
            for t in range(B):
                fd = pfm.get(t, {})
                for oid, mask_np in fd.items():
                    global_frame_masks[t][local_to_global[oid]] = mask_np

            pbar.update(1)

        # ------------------------------------------------------------------
        # Build aligned output
        # ------------------------------------------------------------------
        n_persons = global_slot_count

        if n_persons == 0:
            _logger.warning(
                "SAM3 detected no objects for any prompt. "
                "Returning single empty mask per frame.",
            )
            return (torch.zeros(B, H, W),)

        masks_out = torch.zeros(B, n_persons, H, W)
        for t in range(B):
            for slot, mask_np in global_frame_masks[t].items():
                masks_out[t, slot] = torch.from_numpy(
                    mask_np.astype(np.float32)
                )

        # Reshape to (B * N, H, W) — frame-grouped ordering
        masks_out = masks_out.reshape(B * n_persons, H, W)

        _logger.info(
            "SAM3 segmentation: %d frames, %d prompts, %d total persons, "
            "output shape %s",
            B, n_prompts, n_persons, tuple(masks_out.shape),
        )

        return (masks_out,)
