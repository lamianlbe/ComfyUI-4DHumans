"""
Load Fast SAM 3D Body + MHR2SMPL mapper.

Paths (all hardcoded, must exist before loading; otherwise the node
raises FileNotFoundError with a clear message about where to put it):

    models/fastsam3dbody/sam-3d-body-dinov3/
        model.ckpt
        model_config.yaml              (in same dir or parent dir)
        assets/mhr_model.pt

    models/fastsam3dbody/mhr2smpl/
        best_model.pth                 MHR→SMPL main mapper
        mhr2smpl_mapping.npz           (with triangle_ids, baryc_coords, mhr_vert_ids)
        smpl_vert_sample_indices.npy   (==sample_idx.npy from Fast SAM 3D Body experiments)

    models/fastsam3dbody/mhr2smpl/smoother/     (required when use_smoother=True)
        smoother_best.pth
        smoother_config.json

    models/prompthmr/body_models/smpl/SMPL_NEUTRAL.pkl   (reused from PromptHMR)

No TensorRT in this first cut; the PyTorch path is validated first.
"""

import logging
import os

import torch

from folder_paths import models_dir

from ..fastsam3dbody_lib import ensure_lib_importable

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded paths
# ---------------------------------------------------------------------------
SAM3D_DIR       = os.path.join(models_dir, "fastsam3dbody", "sam-3d-body-dinov3")
SAM3D_CKPT      = os.path.join(SAM3D_DIR, "model.ckpt")
SAM3D_MHR_PATH  = os.path.join(SAM3D_DIR, "assets", "mhr_model.pt")

MHR2SMPL_DIR          = os.path.join(models_dir, "fastsam3dbody", "mhr2smpl")
MHR2SMPL_CKPT         = os.path.join(MHR2SMPL_DIR, "best_model.pth")
MHR2SMPL_MAPPING      = os.path.join(MHR2SMPL_DIR, "mhr2smpl_mapping.npz")
MHR2SMPL_SAMPLE_IDX   = os.path.join(MHR2SMPL_DIR, "smpl_vert_sample_indices.npy")
MHR2SMPL_SMOOTHER_DIR = os.path.join(MHR2SMPL_DIR, "smoother")

SMPL_NEUTRAL_PKL = os.path.join(
    models_dir, "prompthmr", "body_models", "smpl", "SMPL_NEUTRAL.pkl"
)


def _require(path, what):
    """Raise a clear FileNotFoundError if *path* doesn't exist."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{what} not found at: {path}\n"
            f"Please place the required file at this exact location."
        )


class LoadFastSAM3DBodyNode:
    """Load the SAM 3D Body estimator plus MHR2SMPL mapper.

    Outputs a dict ready to be consumed by FastSAM3DBodyRTMFacePose.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_smoother": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Apply MHR2SMPL's MLP SmootherMLP over a 5-frame "
                            "sliding window. Denoises SMPL joint positions at "
                            "negligible cost. Turn off only when debugging."
                        ),
                    },
                ),
                "dtype": (
                    ["float32", "bfloat16", "float16"],
                    {
                        "default": "float32",
                        "tooltip": (
                            "Autocast dtype for the Fast SAM 3D Body forward. "
                            "fp32 is safest. bf16/fp16 give a modest speedup "
                            "on Ampere+/Blackwell but may slightly affect "
                            "the MHR mesh geometry quality."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FASTSAM3DBODY",)
    RETURN_NAMES = ("fast_sam_3d_body",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, use_smoother, dtype):
        # 1. Validate every required file up front
        _require(SAM3D_CKPT,             "Fast SAM 3D Body checkpoint (model.ckpt)")
        _require(SAM3D_MHR_PATH,         "Fast SAM 3D Body MHR model (assets/mhr_model.pt)")
        _require(MHR2SMPL_CKPT,          "MHR2SMPL main model (best_model.pth)")
        _require(MHR2SMPL_MAPPING,       "MHR→SMPL mapping (mhr2smpl_mapping.npz)")
        _require(MHR2SMPL_SAMPLE_IDX,    "SMPL vertex sample indices (smpl_vert_sample_indices.npy)")
        _require(SMPL_NEUTRAL_PKL,       "SMPL neutral body model (SMPL_NEUTRAL.pkl)")

        smoother_dir = None
        if use_smoother:
            smoother_ckpt = os.path.join(MHR2SMPL_SMOOTHER_DIR, "smoother_best.pth")
            smoother_cfg  = os.path.join(MHR2SMPL_SMOOTHER_DIR, "smoother_config.json")
            _require(smoother_ckpt, "SmootherMLP checkpoint (smoother_best.pth)")
            _require(smoother_cfg,  "SmootherMLP config (smoother_config.json)")
            smoother_dir = MHR2SMPL_SMOOTHER_DIR

        # 2. Set up vendored library imports
        ensure_lib_importable()

        # 3. Build the SAM 3D Body model + MHR predictor
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _logger.info("Loading SAM 3D Body from %s ...", SAM3D_CKPT)
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=SAM3D_CKPT,
            device=device,
            mhr_path=SAM3D_MHR_PATH,
        )

        # Estimator runs detection/SAM2/FOV internally — we pass everything
        # external (bboxes, masks, yolo keypoints), so don't wire those up.
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        # 4. Build MHR2SMPL mapper
        from infer_multiview import MHR2SMPLMultiView

        _logger.info(
            "Loading MHR2SMPL (smoother=%s)",
            "on" if use_smoother else "off",
        )
        mhr2smpl = MHR2SMPLMultiView(
            model_path=MHR2SMPL_CKPT,
            mapping_path=MHR2SMPL_MAPPING,
            sample_idx_path=MHR2SMPL_SAMPLE_IDX,
            device=str(device),
            smoother_dir=smoother_dir,
        )

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map[dtype]

        _logger.info(
            "Fast SAM 3D Body ready: image_size=%s, dtype=%s, smoother=%s",
            list(model_cfg.MODEL.IMAGE_SIZE), dtype, bool(smoother_dir),
        )

        return ({
            "estimator": estimator,
            "model_cfg": model_cfg,
            "mhr2smpl": mhr2smpl,
            "smpl_model_path": SMPL_NEUTRAL_PKL,
            "dtype": dtype,
            "torch_dtype": torch_dtype,
            "device": device,
            "use_smoother": bool(smoother_dir),
        },)
