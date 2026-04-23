"""
Load Neural Localizer Fields (NLF) 3D pose model.

Paper: https://arxiv.org/abs/2407.07532 (NeurIPS 2024)
Repo : https://github.com/isarandi/nlf

License: noncommercial research use.

Only the .torchscript file is required. The SMPL-X .npz used in the
NLF demo is solely for mesh-vertex reconstruction (not needed for our
joint-based pipeline).

Hardcoded checkpoint path: models/nlf/nlf_l_multi_0.3.2.torchscript
"""

import os

import torch

from folder_paths import models_dir

# Hardcoded checkpoint path.
NLF_CKPT_PATH = os.path.join(
    models_dir, "nlf", "nlf_l_multi_0.3.2.torchscript"
)


class LoadNLFNode:
    """Load the NLF (Neural Localizer Fields) 3D pose model.

    Outputs a POSE3D dict compatible with the Sapiens PromptHMR Human
    Pose node; selects NLF as the 3D backbone at inference time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {
                        "default": "float32",
                        "tooltip": (
                            "Autocast precision for NLF forward pass. "
                            "NLF is a TorchScript model; autocast wraps "
                            "its calls. fp32 is safest, bf16/fp16 faster "
                            "on modern GPUs."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSE3D",)
    RETURN_NAMES = ("pose_3d_model",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, dtype="float32"):
        if not os.path.isfile(NLF_CKPT_PATH):
            raise FileNotFoundError(
                f"NLF checkpoint not found: {NLF_CKPT_PATH}\n"
                f"Download from https://github.com/isarandi/nlf/releases "
                f"and place as models/nlf/nlf_l_multi_0.3.2.torchscript"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(NLF_CKPT_PATH, map_location=device).eval()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        return ({
            "backend": "nlf",
            "model": model,
            "dtype": dtype,
            "torch_dtype": torch_dtype,
            "checkpoint_path": NLF_CKPT_PATH,
        },)
