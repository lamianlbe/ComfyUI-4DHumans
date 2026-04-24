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

Model weights auto-download to `~/.cache/torch/hub/checkpoints/` via
``pyfacer`` on first use (roughly ~120 MB). License: MIT (FaRL) and
MIT (pyfacer).

Install:
    pip install pyfacer
"""

import logging

_logger = logging.getLogger(__name__)


# pyfacer model name for 68-pt 300W (iBUG) landmarks at 448×448 input.
# Other 'farl/...' aligners in pyfacer target parsing masks, not
# landmarks, so we hard-code the ibug300w checkpoint here.
_FARL_FACE_ALIGNER_NAME = "farl/ibug300w/448"


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
                            "Inference device. FaRL is a ~120 MB ViT "
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
        try:
            import torch
            import facer
        except ImportError as e:
            raise ImportError(
                "pyfacer required. Install with:\n"
                "  pip install pyfacer\n"
                "pyfacer will auto-download FaRL checkpoints on first use."
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
            "Loading FaRL face aligner (%s) on device=%s",
            _FARL_FACE_ALIGNER_NAME, device_str,
        )

        face_aligner = facer.face_aligner(
            _FARL_FACE_ALIGNER_NAME, device=device_str,
        )
        # Ensure eval mode — pyfacer already does this in its constructor
        # on recent versions, but belt and braces.
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
        },)
