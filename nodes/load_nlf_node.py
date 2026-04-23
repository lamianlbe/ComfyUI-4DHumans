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

import logging
import os

import torch

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


def _check_jit_script_function():
    """Warn if another custom node has monkey-patched torch.jit.script.

    Mirrors kijai/ComfyUI-WanVideoWrapper's check; patched torch.jit
    functions can cause subtle NLF TorchScript failures.
    """
    if torch.jit.script.__name__ != "script":
        module = getattr(torch.jit.script, "__module__", "unknown")
        _logger.warning(
            "torch.jit.script has been patched by %s.%s — this may "
            "break the NLF TorchScript model.",
            module, torch.jit.script.__name__,
        )

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

        _check_jit_script_function()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(NLF_CKPT_PATH, map_location=device).eval()

        # Warm up with profiling executor enabled — avoids the
        # "vector::_M_range_check" JIT bug on first real call.
        if torch.cuda.is_available():
            dummy = torch.zeros(1, 3, 256, 256, device=device)
            jit_prev = torch._C._jit_set_profiling_executor(True)
            try:
                with torch.inference_mode():
                    for _ in range(2):
                        _ = model.detect_smpl_batched(dummy)
            except Exception as e:
                _logger.warning("NLF warmup failed (continuing): %s", e)
            finally:
                torch._C._jit_set_profiling_executor(jit_prev)
            _logger.info("NLF warmed up on %s", device)

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
