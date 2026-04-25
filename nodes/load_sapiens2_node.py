"""
Load Sapiens2 pose model — Meta's 308-keypoint standalone backbone.

We use the standalone sapiens2_lib (torch + numpy + cv2 + safetensors
only), so no mmcv / mmdet / mmpose / mmpretrain dependency is needed.

Currently exposes the 1B variant only — the size the user has
validated on Blackwell. To add 0.4b/0.8b/5b: extend
``_VARIANT_TO_HPARAMS`` below, drop the corresponding .safetensors
into ``models/sapiens2/``, and add an entry to the dropdown.

Weight path (ComfyUI convention):
    models/sapiens2/sapiens2_1b_pose.safetensors

Download from Meta's release page (CC-BY-NC 4.0):
    https://github.com/facebookresearch/sapiens2

License note: Sapiens2 is CC-BY-NC 4.0 — no commercial use of the
weights or derived outputs.
"""

import logging
import os

from folder_paths import models_dir

from ..sapiens2_lib import (
    SAPIENS2_1B_HPARAMS,
    SAPIENS2_HEAD_HPARAMS,
    Sapiens2PosePipeline,
    ensure_lib_importable,
)

_logger = logging.getLogger(__name__)


SAPIENS2_MODELS_DIR = os.path.join(models_dir, "sapiens2")

# Right now we only ship the 1B option. Adding more variants is just:
#   1. add a hparams dict here
#   2. add a (variant_name, expected_filename) entry below
#   3. expose in the dropdown
_VARIANT_TO_HPARAMS = {
    "sapiens2_1b": SAPIENS2_1B_HPARAMS,
}
_VARIANT_TO_FILENAME = {
    "sapiens2_1b": "sapiens2_1b_pose.safetensors",
}


class LoadSapiens2Node:
    """Load Sapiens2 pose model (1B, 308-keypoint head)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. Sapiens2-1B is ~3.5 GB "
                            "fp32 / 1.8 GB fp16 — comfortably fits "
                            "Blackwell. CPU works but each forward "
                            "takes 10-20 seconds, only practical for "
                            "single-image debugging."
                        ),
                    },
                ),
                "variant": (
                    list(_VARIANT_TO_HPARAMS.keys()),
                    {
                        "default": "sapiens2_1b",
                        "tooltip": (
                            "Sapiens2 size variant. Currently only "
                            "1B is wired up — that's the one validated "
                            "in our Blackwell + cu130 + Py3.13 env. "
                            "0.4b / 0.8b / 5b can be added later if "
                            "speed/quality tradeoffs need exploring."
                        ),
                    },
                ),
                "dtype": (
                    ["float32", "bfloat16", "float16"],
                    {
                        "default": "float32",
                        "tooltip": (
                            "Weight + activation dtype. fp32 is the "
                            "safest default; bf16/fp16 cut VRAM in "
                            "half and ~2x throughput on Blackwell "
                            "tensor cores. Sapiens2 weights are "
                            "shipped fp32 — we cast on load."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAPIENS2",)
    RETURN_NAMES = ("sapiens2",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, variant, dtype):
        ensure_lib_importable()

        try:
            import torch
        except ImportError as e:
            raise ImportError("torch required") from e

        # Validate weight is on disk before we even try to allocate
        # backbone params — saves a confusing OOM-like error if the
        # checkpoint is missing.
        filename = _VARIANT_TO_FILENAME[variant]
        ckpt_path = os.path.join(SAPIENS2_MODELS_DIR, filename)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Sapiens2 checkpoint not found at:\n  {ckpt_path}\n\n"
                f"Place the {variant} *_pose.safetensors file under "
                f"{SAPIENS2_MODELS_DIR}/. Download from Meta's "
                f"sapiens2 release (CC-BY-NC 4.0)."
            )

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadSapiens2: requested CUDA but torch.cuda.is_available() "
                "is False — falling back to CPU."
            )

        torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]

        _logger.info(
            "Loading Sapiens2 (variant=%s dtype=%s device=%s) from %s",
            variant, dtype, device_str, ckpt_path,
        )

        pipe = Sapiens2PosePipeline.from_safetensors(
            path=ckpt_path,
            device=device_str,
            arch_hparams=_VARIANT_TO_HPARAMS[variant],
            head_hparams=SAPIENS2_HEAD_HPARAMS,
        )

        # Cast all params + buffers to the requested dtype. Inputs in
        # the predict() path go through the same cast at forward time.
        if torch_dtype != torch.float32:
            pipe.model.to(dtype=torch_dtype)

        _logger.info(
            "Sapiens2 ready. Output: 308 keypoints + per-keypoint "
            "confidence scores in [0, 1]. Plug into "
            "Sapiens2InstancePoseNode downstream for the SAM3+iterate "
            "pipeline."
        )

        return ({
            "pipeline":     pipe,
            "device":       device_str,
            "variant":      variant,
            "dtype":        dtype,
            "torch_dtype":  torch_dtype,
            "ckpt_path":    ckpt_path,
        },)
