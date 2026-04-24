"""
Load RTMPose-Face (106-point facial landmark detector) via rtmlib.

Replaces Sapiens for face-specific keypoints (COCO-WholeBody indices
23-90). Sapiens stays available under the old nodes; this one is part
of the Fast SAM 3D Body pipeline.

Hardcoded path:
    models/rtmpose-face/rtmpose-m-face.onnx

We wrap MMPose's ONNX export via the ``rtmlib`` library so the
preprocessing (top-down affine crop → 256×256), ONNX forward, SimCC
decode and back-projection all go through the upstream reference
implementation. This eliminates any possibility of a bug in our own
preprocess/decode being responsible for the alternating-frame issues
we chased earlier — if rtmlib also reproduces them, it's a model-level
property and we'll layer a side-prior decoder on top in a follow-up.
"""

import logging
import os

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


RTMPOSE_FACE_ONNX = os.path.join(models_dir, "rtmpose-face", "rtmpose-m-face.onnx")


class LoadRTMPoseFaceNode:
    """Load an RTMPose-Face ONNX model via ``rtmlib.RTMPose``.

    Building the instance is cheap (< 1 s) — session is created eagerly
    inside rtmlib's constructor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (
                    ["cpu", "cuda"],
                    {
                        "default": "cpu",
                        "tooltip": (
                            "onnxruntime execution provider for rtmlib. "
                            "Default is 'cpu' — CUDA has reproduced the "
                            "alternating mirror bug in earlier tests; we "
                            "confirmed on CPU the model itself gives the "
                            "same pattern, so this is purely a speed knob "
                            "and either is correct."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("RTMPOSEFACE",)
    RETURN_NAMES = ("rtmpose_face",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, provider):
        if not os.path.isfile(RTMPOSE_FACE_ONNX):
            raise FileNotFoundError(
                f"RTMPose-Face ONNX not found at: {RTMPOSE_FACE_ONNX}\n"
                f"Download an RTMPose-m face model (Face6 / 106 landmarks "
                f"recommended) from the MMPose model zoo and place it at "
                f"this exact location."
            )

        try:
            from rtmlib import RTMPose
        except ImportError as e:
            raise ImportError(
                "rtmlib required. Install with:\n"
                "  pip install rtmlib\n"
                "and also one of:\n"
                "  pip install onnxruntime-gpu   (CUDA build)\n"
                "  pip install onnxruntime       (CPU-only build)"
            ) from e

        device = "cuda" if provider == "cuda" else "cpu"

        _logger.info(
            "Loading RTMPose-Face via rtmlib from %s (device=%s)",
            RTMPOSE_FACE_ONNX, device,
        )

        # rtmlib's RTMPose expects the model_input_size matching the ONNX
        # export (256x256 for the face model). Mean/std are the standard
        # ImageNet-RGB numbers MMPose trains on; passing ComfyUI's RGB
        # frames to rtmlib will apply these to matching channels.
        face = RTMPose(
            onnx_model=RTMPOSE_FACE_ONNX,
            model_input_size=(256, 256),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            backend="onnxruntime",
            device=device,
        )

        _logger.info(
            "RTMPose-Face loaded: device=%s, input_size=%s",
            device, (256, 256),
        )

        return ({
            "face": face,
            "device": device,
            "onnx_path": RTMPOSE_FACE_ONNX,
        },)
