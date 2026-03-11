"""
ComfyUI node for loading a Sapiens pose estimation model.

Places model checkpoints under::

    ComfyUI/models/sapiens/pose/
        sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2   (recommended)

Any ``*torchscript*.pt2`` or ``*.pth`` file in that directory will be listed
as a selectable checkpoint.
"""

import glob
import logging
import os

import torch
from torchvision import transforms

import comfy.model_management
from folder_paths import models_dir

SAPIENS_POSE_DIR = os.path.join(models_dir, "sapiens", "pose")
os.makedirs(SAPIENS_POSE_DIR, exist_ok=True)

# Sapiens pose models output heatmaps at 192×256 (W×H)
HEATMAP_W, HEATMAP_H = 192, 256
# Default inference resolution
INFER_SIZE = (1024, 768)  # (H, W) for transforms.Resize

_logger = logging.getLogger(__name__)


def _list_checkpoints():
    """Return basenames of available checkpoint files."""
    patterns = ["*.pt2", "*.pth"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(SAPIENS_POSE_DIR, pat)))
    basenames = sorted(set(os.path.basename(f) for f in files))
    if not basenames:
        basenames = ["(no checkpoints found)"]
    return basenames


def _build_preprocessor():
    """ImageNet-normalised preprocessor matching Sapiens training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INFER_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class LoadSapiensNode:
    """
    Loads a Sapiens Goliath pose estimation model (TorchScript or PyTorch).

    Checkpoint files should be placed in ``models/sapiens/pose/``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_list_checkpoints(),),
                "dtype": (["float32", "bfloat16"],
                          {"default": "float32"}),
            },
        }

    RETURN_TYPES = ("SAPIENS",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, checkpoint, dtype="float32"):
        path = os.path.join(SAPIENS_POSE_DIR, checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Sapiens checkpoint not found: {path}\n"
                f"Please place model files in {SAPIENS_POSE_DIR}/"
            )

        device = comfy.model_management.get_torch_device()
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        # Try TorchScript first (works for .pt2 and many .pth files)
        model = None
        try:
            model = torch.jit.load(path, map_location="cpu")
            _logger.info("Loaded as TorchScript: %s", checkpoint)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                raise RuntimeError(
                    f"'{checkpoint}' is a state-dict / checkpoint, not a "
                    f"loadable model. Please use the TorchScript (.pt2) "
                    f"version instead, or export with: "
                    f"torch.jit.save(torch.jit.trace(model, dummy), path)"
                )
            model = obj
            _logger.info("Loaded as nn.Module: %s", checkpoint)

        model.eval()
        model.to(device).to(torch_dtype)

        preprocessor = _build_preprocessor()

        _logger.info("Loaded Sapiens pose model: %s (dtype=%s)", checkpoint, dtype)

        return ({
            "model": model,
            "preprocessor": preprocessor,
            "device": device,
            "dtype": torch_dtype,
            "heatmap_size": (HEATMAP_W, HEATMAP_H),
        },)
