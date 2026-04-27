"""
Load WiLoR — Imperial/Snap's hand pose estimator with MANO mesh output.

WiLoR architecture: ViT-Huge backbone (~632 M params) + transformer
decoder regressing MANO parameters (pose + shape + cam) → MANO model
emits 21 3D joints + 778-vertex hand mesh → perspective projection
gives 21 2D keypoints in original image coordinates.

We integrate WiLoR as the *optional* high-quality hand override on top
of RTMW-x's 133-point base. When connected to BMPRTMWPoseNode, its
21-point hand outputs replace COCO-WB indices 91..111 (left hand) and
112..132 (right hand).

Required weights under ``ComfyUI/models/wilor/``:

    wilor_final.ckpt              (~2 GB, lightning checkpoint, HF)
    detector.pt                   (~50 MB, YOLO hand detector, HF)
    mano_data/MANO_RIGHT.pkl      (~5 MB, registered download from
                                    https://mano.is.tue.mpg.de/)

Auto-provisioned (we copy from our vendored ``wilor_configs/`` on
first load if missing):

    mano_data/mano_mean_params.npz   (~1.2 KB)

Vendored small support files in repo:

    wilor_configs/model_config.yaml      (verbatim from WiLoR
                                          pretrained_models/)
    wilor_configs/mano_mean_params.npz

Quirk worked around in this loader: ``wilor.models.load_wilor()``
unconditionally rewrites ``model_cfg.MANO.MODEL_PATH = './mano_data/'``
to a CWD-relative path. ``smplx.MANOLayer`` then reads
``MODEL_PATH/MANO_RIGHT.pkl`` at construction time. So we briefly
``os.chdir`` to ``ComfyUI/models/wilor/`` during the load — after
which the model carries everything in memory and CWD doesn't matter.

The WiLoR python package itself is vendored under
``ComfyUI-4DHumans/wilor_lib/`` so users don't have to clone the
upstream repo (which has no setup.py and isn't pip-installable).
The vendor only includes the inference-side code (~250 KB);
weights / demo images / training code are NOT included.

License: WiLoR code Apache-2.0, weights CC-BY-NC. MANO from MPI is
research-only (registration required). This integration assumes
non-commercial use.
"""

import logging
import os
import shutil

import torch

from folder_paths import models_dir

from ..wilor_lib import ensure_lib_importable as _ensure_wilor_importable

_logger = logging.getLogger(__name__)


WILOR_MODELS_DIR = os.path.join(models_dir, "wilor")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WILOR_CONFIGS_ROOT = os.path.join(REPO_ROOT, "wilor_configs")

CKPT_FILENAME       = "wilor_final.ckpt"
DETECTOR_FILENAME   = "detector.pt"
MANO_RIGHT_FILENAME = "MANO_RIGHT.pkl"
MANO_MEAN_FILENAME  = "mano_mean_params.npz"

CKPT_URL     = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt"
DETECTOR_URL = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt"
MANO_URL     = "https://mano.is.tue.mpg.de/   (registration required, download mano_v1_2.zip and extract MANO_RIGHT.pkl)"


def _ensure_runtime_layout() -> str:
    """Make sure ComfyUI/models/wilor/ has the right structure that
    WiLoR's load_wilor expects when run from inside that directory.

    Returns the directory we should ``chdir`` into for the load call.

    Layout (when this returns successfully):

        models/wilor/
        ├── wilor_final.ckpt           (user-provided)
        ├── detector.pt                (user-provided)
        ├── model_config.yaml          (auto-copied from wilor_configs/)
        └── mano_data/
            ├── MANO_RIGHT.pkl         (user-provided)
            └── mano_mean_params.npz   (auto-copied from wilor_configs/)
    """
    os.makedirs(WILOR_MODELS_DIR, exist_ok=True)
    mano_dir = os.path.join(WILOR_MODELS_DIR, "mano_data")
    os.makedirs(mano_dir, exist_ok=True)

    # Required user-provided files
    ckpt_path     = os.path.join(WILOR_MODELS_DIR, CKPT_FILENAME)
    detector_path = os.path.join(WILOR_MODELS_DIR, DETECTOR_FILENAME)
    mano_path     = os.path.join(mano_dir, MANO_RIGHT_FILENAME)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"WiLoR checkpoint not found at:\n  {ckpt_path}\n\n"
            f"Download (~2 GB) from:\n  {CKPT_URL}\n"
            f"Place at the path above. Typical command:\n"
            f"  mkdir -p {WILOR_MODELS_DIR}\n"
            f"  wget -O {ckpt_path} \\\n"
            f"    {CKPT_URL}"
        )
    if not os.path.isfile(detector_path):
        raise FileNotFoundError(
            f"WiLoR YOLO hand detector not found at:\n  {detector_path}\n\n"
            f"Download (~50 MB) from:\n  {DETECTOR_URL}\n"
            f"Place at the path above."
        )
    if not os.path.isfile(mano_path):
        raise FileNotFoundError(
            f"MANO_RIGHT.pkl not found at:\n  {mano_path}\n\n"
            f"MANO is research-only and requires manual registration:\n"
            f"  1. Sign up at https://mano.is.tue.mpg.de/ (free, "
            f"     ~5 minutes, MIT-style research license).\n"
            f"  2. After confirming your email, log in and download "
            f"     'Models & Code' (mano_v1_2.zip).\n"
            f"  3. Extract and copy ``models/MANO_RIGHT.pkl`` to:\n"
            f"     {mano_path}"
        )

    # Auto-copy small files we vendor in the repo if missing locally.
    cfg_src = os.path.join(WILOR_CONFIGS_ROOT, "model_config.yaml")
    cfg_dst = os.path.join(WILOR_MODELS_DIR, "model_config.yaml")
    if not os.path.isfile(cfg_dst):
        if not os.path.isfile(cfg_src):
            raise FileNotFoundError(
                f"Vendored WiLoR model_config.yaml missing: {cfg_src}"
            )
        shutil.copy2(cfg_src, cfg_dst)
        _logger.info("WiLoR: copied model_config.yaml → %s", cfg_dst)

    mean_src = os.path.join(WILOR_CONFIGS_ROOT, MANO_MEAN_FILENAME)
    mean_dst = os.path.join(mano_dir, MANO_MEAN_FILENAME)
    if not os.path.isfile(mean_dst):
        if not os.path.isfile(mean_src):
            raise FileNotFoundError(
                f"Vendored mano_mean_params.npz missing: {mean_src}"
            )
        shutil.copy2(mean_src, mean_dst)
        _logger.info("WiLoR: copied mano_mean_params.npz → %s", mean_dst)

    return WILOR_MODELS_DIR


class LoadWiLoRNode:
    """Load WiLoR ViT-H + MANO + YOLO hand detector."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. WiLoR ViT-H is heavy "
                            "(~700 M params, ~217 GFLOPs per hand at "
                            "256×256 input). CPU works but is "
                            "effectively single-image only — use CUDA "
                            "for video."
                        ),
                    },
                ),
                "fast": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable WiLoR's --fast mode: fp16 weights "
                            "+ torch.compile on the backbone + "
                            "selective ViT layer dropping. Cuts "
                            "per-hand cost ~30-50% with very small "
                            "accuracy loss. Recommended on Blackwell "
                            "and similar tensor-core GPUs."
                        ),
                    },
                ),
                "detector_conf": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.05,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": (
                            "YOLO hand-detector confidence threshold. "
                            "0.3 is WiLoR demo's default. Lower (0.15) "
                            "rescues partially-occluded / out-of-frame "
                            "hands at the cost of more false positives "
                            "(which our wrist-distance matcher will "
                            "filter)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WILOR",)
    RETURN_NAMES = ("wilor",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, fast, detector_conf):
        # Make our vendored ``wilor/`` package importable. Has to happen
        # BEFORE ``from wilor.models import load_wilor`` further down.
        _ensure_wilor_importable()
        runtime_dir = _ensure_runtime_layout()

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadWiLoR: requested CUDA but torch.cuda.is_available() "
                "is False — falling back to CPU."
            )

        try:
            from wilor.models import load_wilor
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Failed to import wilor / ultralytics. The wilor package "
                "is vendored under ComfyUI-4DHumans/wilor_lib/ and should "
                "have been added to sys.path automatically. If this error "
                "persists, the runtime deps are likely missing — install "
                "them with:\n"
                "  pip install pytorch-lightning smplx timm yacs "
                "omegaconf hydra-core scikit-image rich ultralytics\n\n"
                "Underlying error: " + str(e)
            ) from e

        _logger.info(
            "Loading WiLoR (fast=%s) on %s from %s",
            fast, device_str, runtime_dir,
        )

        # WiLoR's load_wilor() rewrites the cfg's MANO paths to
        # './mano_data/...' and then smplx.MANOLayer reads MANO_RIGHT.pkl
        # relative to CWD. So we cd into models/wilor/ for the load.
        original_cwd = os.getcwd()
        try:
            os.chdir(runtime_dir)
            model, model_cfg = load_wilor(
                checkpoint_path="./" + CKPT_FILENAME,
                cfg_path="./model_config.yaml",
            )
            detector = YOLO("./" + DETECTOR_FILENAME)
        finally:
            os.chdir(original_cwd)

        if fast:
            torch.set_float32_matmul_precision("high")
            model = model.half()
            try:
                model.backbone = torch.compile(model.backbone)
                model.backbone.skip_blocks = True
            except Exception as e:
                _logger.warning(
                    "WiLoR torch.compile / layer-drop failed (%s); "
                    "fp16 still active.", e,
                )

        model = model.to(device_str)
        detector = detector.to(device_str)
        model.eval()

        _logger.info(
            "WiLoR ready: ViT-H backbone + MANO head + YOLO hand "
            "detector. Output per detection: 21 keypoints + MANO mesh."
        )

        return ({
            "model":          model,
            "model_cfg":      model_cfg,
            "detector":       detector,
            "device":         device_str,
            "fast":           bool(fast),
            "detector_conf":  float(detector_conf),
            "runtime_dir":    runtime_dir,
        },)
