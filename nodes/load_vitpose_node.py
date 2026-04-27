"""
Load ViTPose via HuggingFace transformers.

Used as a fallback body-pose source when BMP / RTMW miss frames in
extreme scenes. ViTPose has no mask conditioning, so it doesn't suffer
the same degradation BMP's PMPose does when SAM produces noisy masks
(e.g. on very dark / motion-blurred / low-contrast inputs where mask
quality collapses but ViT features remain useful).

Output of the node is a VITPOSE dict consumed by BMPRTMWPoseNode's
optional `vitpose` input. There it acts as a per-(person, frame)
fallback: only fires for slots whose body 0..16 came up empty/low-
confidence after BMP + RTMW + body override.

Hardcoded weight cache (ComfyUI convention):
    models/vitpose/<variant>/   ← transformers from_pretrained-compatible

Variants (HuggingFace IDs, approximate weight sizes):
    vitpose-base-simple    ~360 MB    (90M params, fast)
    vitpose-plus-base      ~360 MB    (multi-dataset, similar size)
    vitpose-plus-large     ~1.5 GB    (400M params)
    vitpose-plus-huge      ~3.5 GB    (900M params, strongest)

Defaults to ``vitpose-plus-huge`` because the user's specific need is
"BMP fails, ViTPose still works on extreme scenes" — only the huge
model is reliably better than BMP/RTMW in those edge cases. Smaller
variants exist for faster but lower-recall fallback if needed.

License: Apache-2.0 code (transformers) + Apache-2.0 weights (the
usyd-community port). Clean for any commercial use.

Required pip dep:
    transformers >= 4.48
"""

import logging
import os

import torch

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


VITPOSE_MODELS_DIR = os.path.join(models_dir, "vitpose")

# variant_name → (HuggingFace repo id, is_plus_variant)
# is_plus_variant flips on dataset_index forwarding (ViTPose+ requires
# a dataset selector tensor; ViTPose-simple doesn't accept it).
_VARIANTS = {
    "vitpose-base-simple": ("usyd-community/vitpose-base-simple", False),
    "vitpose-plus-base":   ("usyd-community/vitpose-plus-base",   True),
    "vitpose-plus-large":  ("usyd-community/vitpose-plus-large",  True),
    "vitpose-plus-huge":   ("usyd-community/vitpose-plus-huge",   True),
}


class LoadViTPoseNode:
    """Load ViTPose via HuggingFace transformers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. ViTPose-plus-huge is "
                            "~900M params — fits Blackwell easily. "
                            "CPU is usable for single-image debugging "
                            "but unusably slow for video."
                        ),
                    },
                ),
                "variant": (
                    list(_VARIANTS.keys()),
                    {
                        "default": "vitpose-plus-huge",
                        "tooltip": (
                            "Model size. plus-huge is recommended as "
                            "the BMP fallback because the user's pain "
                            "point is 'BMP fails on extreme scenes' — "
                            "only the huge model reliably outperforms "
                            "BMP+RTMW in those cases. Smaller "
                            "variants are listed for faster fallback "
                            "if you don't need maximum recall."
                        ),
                    },
                ),
                "dtype": (
                    ["float32", "bfloat16", "float16"],
                    {
                        "default": "float32",
                        "tooltip": (
                            "Weight + activation dtype. fp32 is "
                            "safest. bf16/fp16 cuts VRAM in half and "
                            "speeds up Blackwell tensor-core matmul "
                            "~2×. Match the dtype you use for BMP for "
                            "consistency."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("VITPOSE",)
    RETURN_NAMES = ("vitpose",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, variant, dtype):
        try:
            from transformers import (
                AutoProcessor,
                VitPoseForPoseEstimation,
            )
        except ImportError as e:
            raise ImportError(
                "ViTPose requires transformers >= 4.48. Install/upgrade:\n"
                "  pip install -U 'transformers>=4.48'"
            ) from e

        repo_id, is_plus = _VARIANTS[variant]

        # Prefer local cache: models/vitpose/<variant>/ — works because
        # transformers' from_pretrained accepts a folder path containing
        # model.safetensors + config.json + processor files. Fall back
        # to fetching from HuggingFace if local copy missing.
        local_dir = os.path.join(VITPOSE_MODELS_DIR, variant)
        if os.path.isdir(local_dir) and os.path.isfile(
            os.path.join(local_dir, "config.json")
        ):
            source = local_dir
            _logger.info(
                "Loading ViTPose %s from local cache: %s",
                variant, local_dir,
            )
        else:
            source = repo_id
            _logger.info(
                "Loading ViTPose %s from HuggingFace: %s "
                "(local cache miss at %s — will download to "
                "~/.cache/huggingface/hub/)",
                variant, repo_id, local_dir,
            )

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadViTPose: requested CUDA but torch.cuda is unavailable, "
                "falling back to CPU."
            )

        torch_dtype = {
            "float32":  torch.float32,
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
        }[dtype]

        processor = AutoProcessor.from_pretrained(source)
        model = VitPoseForPoseEstimation.from_pretrained(
            source, torch_dtype=torch_dtype,
        )
        model = model.to(device_str)
        model.eval()

        _logger.info(
            "ViTPose %s ready (variant=%s plus=%s dtype=%s device=%s).",
            variant, variant, is_plus, dtype, device_str,
        )

        return ({
            "backend":   "hf",
            "model":     model,
            "processor": processor,
            "device":    device_str,
            "variant":   variant,
            "is_plus":   is_plus,
            "torch_dtype": torch_dtype,
        },)
