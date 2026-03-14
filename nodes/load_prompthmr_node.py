import os
import sys

import torch
import comfy.model_management
from folder_paths import models_dir

# ── Bundled PromptHMR code ──────────────────────────────────────────────────
# prompthmr_lib/ contains the inference-only subset of the PromptHMR repo.
# We add it to sys.path so that `import prompt_hmr` and `import data_config`
# resolve to the bundled copies, with no external clone or env var needed.

_REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_PATH, "prompthmr_lib")

# ── ComfyUI model directory for PromptHMR assets ────────────────────────────
#
#   models/prompthmr/
#   ├── phmr/
#   │   ├── config.yaml              (model config)
#   │   └── checkpoint.ckpt          (model checkpoint)
#   └── body_models/
#       ├── smplx/
#       │   └── SMPLX_NEUTRAL.npz
#       ├── smpl/
#       │   └── SMPL_NEUTRAL.pkl
#       ├── smplx2smpl.pkl
#       └── smplx2smpl_joints.npy
#
PROMPTHMR_PATH = os.path.join(models_dir, "prompthmr")
_MODEL_DIR = os.path.join(PROMPTHMR_PATH, "phmr")
_BODY_MODELS_DIR = os.path.join(PROMPTHMR_PATH, "body_models")
_SMPLX_DIR = os.path.join(_BODY_MODELS_DIR, "smplx")
_SMPL_DIR = os.path.join(_BODY_MODELS_DIR, "smpl")

_REQUIRED_FILES = {
    os.path.join(_MODEL_DIR, "config.yaml"): "PromptHMR config",
    os.path.join(_MODEL_DIR, "checkpoint.ckpt"): "PromptHMR checkpoint",
    os.path.join(_SMPLX_DIR, "SMPLX_NEUTRAL.npz"): "SMPL-X neutral body model",
    os.path.join(_SMPL_DIR, "SMPL_NEUTRAL.pkl"): "SMPL neutral body model",
    os.path.join(_BODY_MODELS_DIR, "smplx2smpl.pkl"): "SMPL-X to SMPL mapping matrix",
    os.path.join(_BODY_MODELS_DIR, "smplx2smpl_joints.npy"): "SMPL-X to SMPL joint mapping",
}


def _check_required_files():
    missing = []
    for path, desc in _REQUIRED_FILES.items():
        if not os.path.exists(path):
            missing.append(f"  {desc}  ->  {path}")
    if missing:
        raise FileNotFoundError(
            "The following PromptHMR files are missing.\n"
            "Please place them under  models/prompthmr/  before loading:\n\n"
            + "\n".join(missing)
        )


def _ensure_lib_importable():
    """Add the bundled prompthmr_lib to sys.path and patch model paths."""
    if _LIB_PATH not in sys.path:
        sys.path.insert(0, _LIB_PATH)

    # Patch data_config (bundled) with actual model paths
    import data_config
    data_config.SMPLX_PATH = _SMPLX_DIR
    data_config.SMPL_PATH = _SMPL_DIR
    data_config.SMPLX2SMPL = os.path.join(_BODY_MODELS_DIR, "smplx2smpl.pkl")


def _load_model():
    """Load PromptHMR model from the ComfyUI models directory."""
    _ensure_lib_importable()

    # Patch hardcoded paths in prompt_hmr modules before model construction
    import prompt_hmr.models.phmr as phmr_module
    phmr_module.SMPLX_MODEL_DIR = _SMPLX_DIR
    phmr_module.SMPL_MODEL_DIR = _SMPL_DIR
    phmr_module.SMPLX2SMPL = os.path.join(_BODY_MODELS_DIR, "smplx2smpl.pkl")

    import prompt_hmr.smpl_family.smpl_wrapper as smpl_wrapper_module
    smpl_wrapper_module.SMPLX2SMPL_JOINTS = os.path.join(
        _BODY_MODELS_DIR, "smplx2smpl_joints.npy"
    )

    import prompt_hmr.models.components.smpl_decoder as smpl_decoder_module
    smpl_decoder_module.SMPLX_MODEL_DIR = _SMPLX_DIR

    from prompt_hmr import load_model_from_folder

    model = load_model_from_folder(str(_MODEL_DIR))
    return model


class LoadPromptHMRNode:
    """
    Loads a PromptHMR model for 3D human mesh recovery.

    Model files must be placed under models/prompthmr/::

        models/prompthmr/
        ├── phmr/
        │   ├── config.yaml
        │   └── checkpoint.ckpt
        └── body_models/
            ├── smplx/
            │   └── SMPLX_NEUTRAL.npz
            ├── smpl/
            │   └── SMPL_NEUTRAL.pkl
            ├── smplx2smpl.pkl
            └── smplx2smpl_joints.npy
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("PROMPTHMR",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self):
        _check_required_files()
        model = _load_model()
        return ({"model": model, "img_size": 896},)
