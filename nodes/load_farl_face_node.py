"""
Load FaRL face aligner (68-point 300W landmark detector) via pyfacer.

Replaces the RTMPose-Face node in the Fast SAM 3D Body pipeline.
RTMPose-Face's SimCC argmax decoder was shown to produce
bilaterally-mirrored landmarks on alternating frames of a nearly
stationary face, caused by bimodal heatmap peaks (true + mirror
ghost). FaRL uses a ViT backbone with a 448×448 input and a
heatmap-regression head at a much larger receptive field, which is
structurally more robust against the mirror-ghost failure mode.
Additionally, FaRL outputs 68 landmarks in the standard 300W (iBUG)
convention — a perfect match for COCO-WholeBody indices 23..90 with
zero remapping.

Hardcoded path (ComfyUI convention — no torch hub auto-download):
    models/farl/face_alignment.farl.ibug300w.main_ema_jit.pt

Download manually from:
    https://github.com/FacePerceiver/facer/releases/download/models-v1/face_alignment.farl.ibug300w.main_ema_jit.pt
(~617 MB TorchScript, includes both the aligner head and the ViT-B
backbone state_dict embedded as an extra_files entry.)

License: MIT (FaRL) + MIT (pyfacer).

Install:
    pip install pyfacer
"""

import logging
import os

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


# pyfacer model name for 68-pt 300W (iBUG) landmarks at 448×448 input.
# Other 'farl/...' aligners in pyfacer target parsing masks, not
# landmarks, so we hard-code the ibug300w checkpoint here.
_FARL_FACE_ALIGNER_NAME = "farl/ibug300w/448"

# The local checkpoint we expect. The filename MUST still contain "jit"
# because pyfacer's loader branches on that substring to pick
# torch.jit.load vs torch.load — we keep the upstream filename verbatim
# so this isn't an issue.
FARL_FACE_CHECKPOINT_FILENAME = (
    "face_alignment.farl.ibug300w.main_ema_jit.pt"
)
FARL_FACE_CHECKPOINT_URL = (
    "https://github.com/FacePerceiver/facer/releases/download/"
    "models-v1/" + FARL_FACE_CHECKPOINT_FILENAME
)
FARL_FACE_CHECKPOINT = os.path.join(
    models_dir, "farl", FARL_FACE_CHECKPOINT_FILENAME,
)


class LoadFaRLFaceNode:
    """Load a pyfacer FaRL face aligner for 68-point face landmarks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. FaRL is a ~617 MB ViT-B "
                            "and runs in <20 ms per frame batch on a "
                            "modern GPU; CPU works but is ~10× slower."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FARLFACE",)
    RETURN_NAMES = ("farl_face",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device):
        if not os.path.isfile(FARL_FACE_CHECKPOINT):
            raise FileNotFoundError(
                f"FaRL face checkpoint not found at:\n"
                f"  {FARL_FACE_CHECKPOINT}\n\n"
                f"Download it manually (~617 MB) from:\n"
                f"  {FARL_FACE_CHECKPOINT_URL}\n\n"
                f"and place it at the path above. Typical shell command:\n"
                f"  mkdir -p {os.path.dirname(FARL_FACE_CHECKPOINT)}\n"
                f"  wget -O {FARL_FACE_CHECKPOINT} \\\n"
                f"    {FARL_FACE_CHECKPOINT_URL}"
            )

        try:
            import torch
            import facer
        except ImportError as e:
            raise ImportError(
                "pyfacer required. Install with:\n"
                "  pip install pyfacer"
            ) from e

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadFaRLFace: requested CUDA but torch.cuda.is_available() "
                "is False — falling back to CPU."
            )

        _logger.info(
            "Loading FaRL face aligner (%s) on device=%s from %s",
            _FARL_FACE_ALIGNER_NAME, device_str, FARL_FACE_CHECKPOINT,
        )

        # ``model_path`` forwards through to FaRLFaceAlignment.__init__
        # which calls pyfacer's download_jit helper — the helper detects
        # a non-URL path and skips downloading, loading torch.jit.load
        # directly from our local file.
        face_aligner = facer.face_aligner(
            _FARL_FACE_ALIGNER_NAME,
            device=device_str,
            model_path=FARL_FACE_CHECKPOINT,
        )
        try:
            face_aligner.eval()
        except AttributeError:
            pass

        _logger.info(
            "FaRL face aligner loaded. Output: 68 pts in 300W/iBUG "
            "convention (matches COCO-WholeBody 23..90)."
        )

        return ({
            "aligner": face_aligner,
            "device": device_str,
            "model_name": _FARL_FACE_ALIGNER_NAME,
            "model_path": FARL_FACE_CHECKPOINT,
        },)
