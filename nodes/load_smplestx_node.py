import os
import sys

import torch
from torch.nn.parallel.data_parallel import DataParallel
from collections import OrderedDict

import comfy.model_management
from folder_paths import models_dir

from .. import REPO_PATH

# ── ComfyUI model directory for SMPLest-X assets ─────────────────────────────
#
#   models/smplestx/
#   ├── smplest_x_h.pth.tar              (model checkpoint)
#   └── human_model_files/
#       └── smplx/
#           ├── SMPLX_NEUTRAL.npz
#           ├── SMPLX_MALE.npz            (optional)
#           ├── SMPLX_FEMALE.npz           (optional)
#           ├── SMPLX_to_J14.pkl
#           ├── MANO_SMPLX_vertex_ids.pkl
#           └── SMPL-X__FLAME_vertex_ids.npy
#
SMPLESTX_PATH = os.path.join(models_dir, "smplestx")
_HUMAN_MODEL_PATH = os.path.join(SMPLESTX_PATH, "human_model_files")
_SMPLX_PATH = os.path.join(_HUMAN_MODEL_PATH, "smplx")
_DEFAULT_CKPT = os.path.join(SMPLESTX_PATH, "smplest_x_h.pth.tar")

_REQUIRED_FILES = {
    os.path.join(_SMPLX_PATH, "SMPLX_NEUTRAL.npz"):          "SMPL-X neutral body model",
    os.path.join(_SMPLX_PATH, "SMPLX_to_J14.pkl"):           "SMPL-X to J14 regressor",
    os.path.join(_SMPLX_PATH, "MANO_SMPLX_vertex_ids.pkl"):  "MANO SMPLX vertex IDs",
    os.path.join(_SMPLX_PATH, "SMPL-X__FLAME_vertex_ids.npy"): "SMPLX FLAME vertex IDs",
}

# ── SMPLest-X repo path ──────────────────────────────────────────────────────
_SMPLESTX_REPO = os.path.join(os.path.dirname(REPO_PATH), "SMPLest-X")


def _check_required_files():
    missing = []
    for path, desc in _REQUIRED_FILES.items():
        if not os.path.exists(path):
            missing.append(f"  {desc}  →  {path}")
    if not os.path.exists(_DEFAULT_CKPT):
        missing.append(f"  Model checkpoint  →  {_DEFAULT_CKPT}")
    if missing:
        raise FileNotFoundError(
            "The following SMPLest-X model files are missing.\n"
            "Please place them under  models/smplestx/  before loading:\n\n"
            + "\n".join(missing)
        )


def _ensure_smplestx_importable():
    """Add SMPLest-X repo to sys.path so its bare-module imports resolve."""
    try:
        from models.SMPLest_X import get_model  # noqa: F401
        return True
    except ImportError:
        pass

    if os.path.isdir(_SMPLESTX_REPO) and _SMPLESTX_REPO not in sys.path:
        sys.path.insert(0, _SMPLESTX_REPO)
        try:
            from models.SMPLest_X import get_model  # noqa: F401
            return True
        except ImportError:
            sys.path.remove(_SMPLESTX_REPO)
    return False


def _build_cfg():
    """Build a minimal Config for SMPLest-X ViT-Huge inference."""
    from main.config import Config

    return Config({
        "model": {
            "model_type": "vit_huge",
            "pretrained_model_path": _DEFAULT_CKPT,
            "human_model_path": _HUMAN_MODEL_PATH,
            "encoder_pretrained_model_path": None,
            "encoder_config": {
                "num_classes": 80,
                "task_tokens_num": 80,
                "img_size": (256, 192),
                "patch_size": 16,
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
                "ratio": 1,
                "use_checkpoint": False,
                "mlp_ratio": 4,
                "qkv_bias": True,
                "drop_path_rate": 0.55,
            },
            "decoder_config": {
                "feat_dim": 1280,
                "dim_out": 512,
                "task_tokens_num": 80,
            },
            "input_img_shape": (512, 384),
            "input_body_shape": (256, 192),
            "output_hm_shape": (16, 16, 12),
            "focal": (5000, 5000),
            "princpt": (192 / 2, 256 / 2),
            "body_3d_size": 2,
            "hand_3d_size": 0.3,
            "face_3d_size": 0.3,
            "camera_3d_size": 2.5,
        },
        "data": {
            "bbox_ratio": 1.2,
            "testset": "ComfyUI",
        },
        "log": {
            "exp_name": None,
            "output_dir": None,
            "model_dir": None,
            "log_dir": None,
            "result_dir": None,
        },
    })


def _load_model(cfg):
    """Load SMPLest-X model with checkpoint, return DataParallel model in eval mode."""
    from models.SMPLest_X import get_model
    from human_models.human_models import SMPLX

    # Initialise the SMPLX singleton (must happen before get_model)
    SMPLX(cfg.model.human_model_path)

    model = get_model(cfg, "test")
    model = DataParallel(model).cuda()

    ckpt = torch.load(cfg.model.pretrained_model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in ckpt["network"].items():
        if "module" not in k:
            k = "module." + k
        k = (
            k.replace("backbone", "encoder")
            .replace("body_rotation_net", "body_regressor")
            .replace("hand_rotation_net", "hand_regressor")
        )
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()
    return model


class LoadSMPLestXNode:
    """
    Loads a SMPLest-X whole-body pose estimator (ViT-Huge, 137 joints).

    All files must be placed manually under  models/smplestx/  before use:

        models/smplestx/
        ├── smplest_x_h.pth.tar
        └── human_model_files/
            └── smplx/
                ├── SMPLX_NEUTRAL.npz
                ├── SMPLX_to_J14.pkl
                ├── MANO_SMPLX_vertex_ids.pkl
                └── SMPL-X__FLAME_vertex_ids.npy

    The SMPLest-X source repo must be cloned next to ComfyUI-4DHumans::

        pip install -r /path/to/SMPLest-X/requirements.txt
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SMPLESTX",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self):
        if not _ensure_smplestx_importable():
            raise RuntimeError(
                "SMPLest-X package not found.\n"
                f"Expected repo at: {_SMPLESTX_REPO}\n"
                "Clone it and install its requirements, then restart ComfyUI."
            )

        _check_required_files()

        cfg = _build_cfg()
        model = _load_model(cfg)

        return ({"model": model, "cfg": cfg},)
