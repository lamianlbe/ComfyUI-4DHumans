import os
import sys

import comfy.model_management
from folder_paths import models_dir
from omegaconf import OmegaConf

from .. import REPO_PATH

# ── ComfyUI model directory for all PHALP assets ─────────────────────────────
# Actual layout downloaded by PHALP:
#
#   models/phalp/
#   ├── 3D/
#   │   ├── models/
#   │   │   ├── smpl/
#   │   │   │   └── SMPL_NEUTRAL.pkl
#   │   │   └── smpl_mean_params.npz   (duplicate, ignored)
#   │   ├── SMPL_to_J19.pkl
#   │   ├── smpl_mean_params.npz
#   │   ├── texture.npz
#   │   ├── head_faces.npy
#   │   ├── mean_std.npy
#   │   ├── bmap_256.npy
#   │   └── fmap_256.npy
#   ├── weights/
#   │   ├── hmar_v2_weights.pth
#   │   ├── pose_predictor.pth
#   │   └── pose_predictor.yaml
#   └── ava/
#       ├── ava_labels.pkl
#       └── ava_class_mapping.pkl
PHALP_PATH          = os.path.join(models_dir, "phalp")
_PHALP_3D_PATH      = os.path.join(PHALP_PATH, "3D")
_PHALP_SMPL_PATH    = os.path.join(_PHALP_3D_PATH, "models", "smpl")
_PHALP_WEIGHTS_PATH = os.path.join(PHALP_PATH, "weights")
_PHALP_AVA_PATH     = os.path.join(PHALP_PATH, "ava")

# All files that must exist before the tracker can be initialised
_REQUIRED_FILES = {
    os.path.join(_PHALP_SMPL_PATH,    "SMPL_NEUTRAL.pkl"):      "SMPL neutral body model",
    os.path.join(_PHALP_3D_PATH,      "SMPL_to_J19.pkl"):       "SMPL-to-J19 joint regressor",
    os.path.join(_PHALP_3D_PATH,      "smpl_mean_params.npz"):  "SMPL mean parameters",
    os.path.join(_PHALP_3D_PATH,      "texture.npz"):           "SMPL texture atlas",
    os.path.join(_PHALP_3D_PATH,      "head_faces.npy"):        "SMPL head faces",
    os.path.join(_PHALP_3D_PATH,      "mean_std.npy"):          "pose mean/std normalisation",
    os.path.join(_PHALP_3D_PATH,      "bmap_256.npy"):          "UV bmap",
    os.path.join(_PHALP_3D_PATH,      "fmap_256.npy"):          "UV fmap",
    os.path.join(_PHALP_WEIGHTS_PATH, "hmar_v2_weights.pth"):   "HMAR model weights",
    os.path.join(_PHALP_WEIGHTS_PATH, "pose_predictor.pth"):    "pose predictor weights",
    os.path.join(_PHALP_WEIGHTS_PATH, "pose_predictor.yaml"):   "pose predictor config",
    os.path.join(_PHALP_AVA_PATH,     "ava_labels.pkl"):        "AVA labels",
    os.path.join(_PHALP_AVA_PATH,     "ava_class_mapping.pkl"): "AVA class mapping",
}


def _check_required_files():
    missing = [
        f"  {desc}  →  {path}"
        for path, desc in _REQUIRED_FILES.items()
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "The following PHALP model files are missing.\n"
            "Please place them under  models/phalp/  before loading:\n\n"
            + "\n".join(missing)
        )


def _ensure_phalp_importable():
    try:
        import phalp  # noqa: F401
        return True
    except ImportError:
        candidates = [os.path.join(os.path.dirname(REPO_PATH), "PHALP")]
        for p in candidates:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
                try:
                    import phalp  # noqa: F401
                    return True
                except ImportError:
                    sys.path.remove(p)
        return False


def _register_torch_safe_globals():
    """Allow OmegaConf types that PHALP embeds in its checkpoints.

    PyTorch 2.6 changed torch.load's default to weights_only=True, which
    blocks unpickling of arbitrary globals.  PHALP's checkpoints contain
    omegaconf.DictConfig objects, so we allowlist them here.
    """
    import torch
    import torch.serialization as ts
    try:
        from omegaconf import DictConfig, ListConfig
        if hasattr(ts, "add_safe_globals"):
            ts.add_safe_globals([DictConfig, ListConfig])
    except Exception:
        pass  # older PyTorch or omegaconf not available — no-op


def _build_comfy_phalp_class():
    """PHALP subclass that skips all downloads."""
    from phalp.trackers.PHALP import PHALP

    class ComfyPHALP(PHALP):
        def cached_download_from_drive(self, additional_urls=None):
            pass  # files are managed by the user; errors already raised above

    return ComfyPHALP


def _make_comfy_cfg(n_init, device_str, tmp_dir):
    from phalp.configs.base import FullConfig

    cfg = OmegaConf.structured(FullConfig)

    cfg.video.source     = "dummy"
    cfg.video.output_dir = tmp_dir

    cfg.device         = device_str
    cfg.render.enable  = False
    cfg.detect_shots   = False
    cfg.overwrite      = True
    cfg.phalp.n_init   = n_init
    cfg.phalp.detector = "vitdet"

    cfg.SMPL.MODEL_PATH            = _PHALP_SMPL_PATH
    cfg.SMPL.JOINT_REGRESSOR_EXTRA = os.path.join(_PHALP_3D_PATH,   "SMPL_to_J19.pkl")
    cfg.SMPL.TEXTURE               = os.path.join(_PHALP_3D_PATH,   "texture.npz")
    cfg.MODEL.SMPL_HEAD.SMPL_MEAN_PARAMS = os.path.join(_PHALP_3D_PATH, "smpl_mean_params.npz")

    cfg.hmr.hmar_path               = os.path.join(_PHALP_WEIGHTS_PATH, "hmar_v2_weights.pth")
    cfg.pose_predictor.weights_path = os.path.join(_PHALP_WEIGHTS_PATH, "pose_predictor.pth")
    cfg.pose_predictor.config_path  = os.path.join(_PHALP_WEIGHTS_PATH, "pose_predictor.yaml")
    cfg.pose_predictor.mean_std     = os.path.join(_PHALP_3D_PATH,      "mean_std.npy")

    cfg.render.head_mask_path = os.path.join(_PHALP_3D_PATH, "head_faces.npy")

    cfg.ava_config.ava_labels_path         = os.path.join(_PHALP_AVA_PATH, "ava_labels.pkl")
    cfg.ava_config.ava_class_mappping_path = os.path.join(_PHALP_AVA_PATH, "ava_class_mapping.pkl")

    return cfg


class LoadPHALPNode:
    """
    Loads a PHALP tracker from local model files.

    All files must be placed manually under  models/phalp/  before use:

        models/phalp/
        ├── smpl/
        │   ├── SMPL_NEUTRAL.pkl
        │   ├── SMPL_to_J19.pkl
        │   └── smpl_mean_params.npz
        ├── weights/
        │   ├── hmar_v2_weights.pth
        │   ├── pose_predictor.pth
        │   └── pose_predictor.yaml
        ├── 3D/
        │   ├── head_faces.npy
        │   ├── mean_std.npy
        │   ├── texture.npz
        │   ├── bmap_256.npy
        │   └── fmap_256.npy
        └── ava/
            ├── ava_labels.pkl
            └── ava_class_mapping.pkl

    If any file is missing a clear error is raised listing what is absent.

    Prerequisites
    -------------
    Install phalp from source::

        pip install -e /path/to/PHALP
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_init": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": (
                            "Consecutive detections required before a track is "
                            "confirmed and drawn. "
                            "1 = skeleton visible from the very first frame. "
                            "Higher values suppress short false-positive tracks."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PHALP",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, n_init):
        if not _ensure_phalp_importable():
            raise RuntimeError(
                "phalp package not found.\n"
                "Install it with:  pip install -e /path/to/PHALP\n"
                "then restart ComfyUI."
            )

        _check_required_files()
        _register_torch_safe_globals()

        import tempfile
        device  = comfy.model_management.get_torch_device()
        tmp_dir = tempfile.mkdtemp(prefix="phalp_comfy_")

        ComfyPHALP = _build_comfy_phalp_class()
        cfg        = _make_comfy_cfg(n_init, str(device), tmp_dir)
        tracker    = ComfyPHALP(cfg)
        tracker.eval()

        return ({"tracker": tracker, "cfg": cfg},)
