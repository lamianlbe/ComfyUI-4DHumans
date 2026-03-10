import os
import sys

import comfy.model_management
from folder_paths import models_dir
from omegaconf import OmegaConf

from .. import REPO_PATH

# ── ComfyUI model directories ────────────────────────────────────────────────
# Shared with 4DHumans (SMPL model files are identical)
SMPL_PATH   = os.path.join(models_dir, "smpl")
# PHALP-specific weights and auxiliary 3D data
PHALP_PATH  = os.path.join(models_dir, "phalp")

_PHALP_WEIGHTS_PATH = os.path.join(PHALP_PATH, "weights")
_PHALP_3D_PATH      = os.path.join(PHALP_PATH, "3D")
_PHALP_AVA_PATH     = os.path.join(PHALP_PATH, "ava")


def _ensure_phalp_importable():
    """
    Try to import phalp.  If not found on sys.path, look for a sibling
    'PHALP' checkout next to this repo.
    """
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


def _build_comfy_phalp_class():
    """
    Dynamically build a PHALP subclass whose cached_download_from_drive()
    puts everything under ComfyUI's models directory instead of ~/.cache/phalp/.

    Shared SMPL files (SMPL_NEUTRAL.pkl, SMPL_to_J19.pkl, smpl_mean_params.npz)
    are stored in models/smpl/ and reused by both 4DHumans and PHALP.
    PHALP-specific assets go to models/phalp/.
    """
    from phalp.trackers.PHALP import PHALP
    from phalp.utils.utils_download import cache_url
    from phalp.utils.utils import convert_pkl

    class ComfyPHALP(PHALP):

        def cached_download_from_drive(self, additional_urls=None):
            # ── create directories ───────────────────────────────────────
            for d in [SMPL_PATH, _PHALP_WEIGHTS_PATH, _PHALP_3D_PATH, _PHALP_AVA_PATH]:
                os.makedirs(d, exist_ok=True)

            # ── SMPL neutral model  (shared with 4DHumans) ───────────────
            smpl_neutral = os.path.join(SMPL_PATH, "SMPL_NEUTRAL.pkl")
            if not os.path.exists(smpl_neutral):
                raw_pkl = os.path.join(SMPL_PATH, "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
                cache_url(
                    "https://github.com/classner/up/raw/master/models/3D/"
                    "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                    raw_pkl,
                )
                convert_pkl(raw_pkl, smpl_neutral)
                os.remove(raw_pkl)

            # ── files shared with 4DHumans (download to models/smpl/) ────
            shared_3d = {
                "SMPL_to_J19.pkl":      "https://www.cs.utexas.edu/~pavlakos/phalp/3D/SMPL_to_J19.pkl",
                "smpl_mean_params.npz": "https://www.cs.utexas.edu/~pavlakos/phalp/3D/smpl_mean_params.npz",
            }
            for fname, url in shared_3d.items():
                dst = os.path.join(SMPL_PATH, fname)
                if not os.path.exists(dst):
                    cache_url(url, dst)

            # ── PHALP-specific 3D auxiliary data ─────────────────────────
            phalp_3d = {
                "head_faces.npy":  "https://www.cs.utexas.edu/~pavlakos/phalp/3D/head_faces.npy",
                "mean_std.npy":    "https://www.cs.utexas.edu/~pavlakos/phalp/3D/mean_std.npy",
                "texture.npz":     "https://www.cs.utexas.edu/~pavlakos/phalp/3D/texture.npz",
                "bmap_256.npy":    "https://www.cs.utexas.edu/~pavlakos/phalp/bmap_256.npy",
                "fmap_256.npy":    "https://www.cs.utexas.edu/~pavlakos/phalp/fmap_256.npy",
            }
            for fname, url in phalp_3d.items():
                dst = os.path.join(_PHALP_3D_PATH, fname)
                if not os.path.exists(dst):
                    cache_url(url, dst)

            # ── PHALP model weights ───────────────────────────────────────
            phalp_weights = {
                "hmar_v2_weights.pth": "https://www.cs.utexas.edu/~pavlakos/phalp/weights/hmar_v2_weights.pth",
                "pose_predictor.pth":  "https://www.cs.utexas.edu/~pavlakos/phalp/weights/pose_predictor_40006.ckpt",
                "pose_predictor.yaml": "https://www.cs.utexas.edu/~pavlakos/phalp/weights/config_40006.yaml",
            }
            for fname, url in phalp_weights.items():
                dst = os.path.join(_PHALP_WEIGHTS_PATH, fname)
                if not os.path.exists(dst):
                    cache_url(url, dst)

            # ── AVA auxiliary data ────────────────────────────────────────
            phalp_ava = {
                "ava_labels.pkl":        "https://www.cs.utexas.edu/~pavlakos/phalp/ava/ava_labels.pkl",
                "ava_class_mapping.pkl": "https://www.cs.utexas.edu/~pavlakos/phalp/ava/ava_class_mappping.pkl",
            }
            for fname, url in phalp_ava.items():
                dst = os.path.join(_PHALP_AVA_PATH, fname)
                if not os.path.exists(dst):
                    cache_url(url, dst)

            # ── any extra URLs passed by subclasses ───────────────────────
            for fname, (url, dest_dir) in (additional_urls or {}).items():
                dst = os.path.join(dest_dir, fname)
                if not os.path.exists(dst):
                    cache_url(url, dst)

    return ComfyPHALP


def _make_comfy_cfg(n_init, device_str, tmp_dir):
    """
    Build a FullConfig with all paths redirected to ComfyUI's models directory.
    Shared SMPL files point to models/smpl/; PHALP-specific files to models/phalp/.
    """
    from phalp.configs.base import FullConfig

    cfg = OmegaConf.structured(FullConfig)

    # ── required but unused by our per-frame API ─────────────────────────────
    cfg.video.source     = "dummy"
    cfg.video.output_dir = tmp_dir

    # ── runtime settings ─────────────────────────────────────────────────────
    cfg.device           = device_str
    cfg.render.enable    = False      # we render ourselves
    cfg.detect_shots     = False
    cfg.overwrite        = True
    cfg.phalp.n_init     = n_init
    cfg.phalp.detector   = "vitdet"

    # ── SMPL paths  (shared with 4DHumans) ───────────────────────────────────
    cfg.SMPL.MODEL_PATH           = SMPL_PATH
    cfg.SMPL.JOINT_REGRESSOR_EXTRA = os.path.join(SMPL_PATH, "SMPL_to_J19.pkl")
    cfg.SMPL.TEXTURE              = os.path.join(_PHALP_3D_PATH, "texture.npz")
    cfg.MODEL.SMPL_HEAD.SMPL_MEAN_PARAMS = os.path.join(SMPL_PATH, "smpl_mean_params.npz")

    # ── PHALP weights ─────────────────────────────────────────────────────────
    cfg.hmr.hmar_path               = os.path.join(_PHALP_WEIGHTS_PATH, "hmar_v2_weights.pth")
    cfg.pose_predictor.weights_path = os.path.join(_PHALP_WEIGHTS_PATH, "pose_predictor.pth")
    cfg.pose_predictor.config_path  = os.path.join(_PHALP_WEIGHTS_PATH, "pose_predictor.yaml")
    cfg.pose_predictor.mean_std     = os.path.join(_PHALP_3D_PATH, "mean_std.npy")

    # ── PHALP 3D auxiliary ────────────────────────────────────────────────────
    cfg.render.head_mask_path = os.path.join(_PHALP_3D_PATH, "head_faces.npy")

    # ── AVA ───────────────────────────────────────────────────────────────────
    cfg.ava_config.ava_labels_path        = os.path.join(_PHALP_AVA_PATH, "ava_labels.pkl")
    cfg.ava_config.ava_class_mappping_path = os.path.join(_PHALP_AVA_PATH, "ava_class_mapping.pkl")

    return cfg


class LoadPHALPNode:
    """
    Loads a PHALP tracker.

    Model storage
    -------------
    Shared SMPL files are stored in (and reused from) ComfyUI's
    **models/smpl/** directory alongside the 4DHumans SMPL model:

        models/smpl/SMPL_NEUTRAL.pkl
        models/smpl/SMPL_to_J19.pkl
        models/smpl/smpl_mean_params.npz

    PHALP-specific weights and auxiliary data go to **models/phalp/**:

        models/phalp/weights/hmar_v2_weights.pth
        models/phalp/weights/pose_predictor.pth  (.yaml)
        models/phalp/3D/   (head_faces, mean_std, texture, bmap/fmap …)
        models/phalp/ava/  (AVA label files)

    All files are downloaded automatically on first run.

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

        import tempfile
        device  = comfy.model_management.get_torch_device()
        tmp_dir = tempfile.mkdtemp(prefix="phalp_comfy_")

        ComfyPHALP = _build_comfy_phalp_class()
        cfg        = _make_comfy_cfg(n_init, str(device), tmp_dir)
        tracker    = ComfyPHALP(cfg)
        tracker.eval()

        return ({"tracker": tracker, "cfg": cfg},)
