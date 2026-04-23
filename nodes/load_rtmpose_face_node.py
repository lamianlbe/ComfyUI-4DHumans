"""
Load RTMPose-Face (68-point facial landmark detector).

Replaces Sapiens for the face-specific 68 keypoints (COCO-WholeBody
indices 23-90). Sapiens stays available under the old nodes; this one
is part of the Fast SAM 3D Body pipeline.

Hardcoded path:
    models/rtmpose-face/rtmpose-m-face.onnx

Expected ONNX:
- Input: (N, 3, 256, 256) float32 [0, 1]  (preprocess matches MMPose)
- Output: simcc_x (N, 68, 256*simcc_split) and simcc_y (N, 68, 256*simcc_split)
  OR keypoints (N, 68, 2) + scores (N, 68) — depends on export.
"""

import logging
import os

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


RTMPOSE_FACE_ONNX = os.path.join(models_dir, "rtmpose-face", "rtmpose-m-face.onnx")


class LoadRTMPoseFaceNode:
    """Load an RTMPose-Face ONNX model via onnxruntime.

    Building the InferenceSession is cheap (< 1 s). We eagerly create it
    so the first call in the inference node doesn't pay that cost.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "onnxruntime execution provider. 'cuda' uses "
                            "CUDAExecutionProvider; 'cpu' falls back to CPU."
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
                f"Download an RTMPose-m face model (COCO-WholeBody 68 "
                f"landmarks recommended) from the MMPose model zoo and "
                f"place it at this exact location."
            )

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime required. Install with:\n"
                "  pip install onnxruntime-gpu   (CUDA build)\n"
                "or\n"
                "  pip install onnxruntime       (CPU-only build)"
            ) from e

        if provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        _logger.info(
            "Loading RTMPose-Face ONNX from %s (providers=%s)",
            RTMPOSE_FACE_ONNX, providers,
        )
        session = ort.InferenceSession(RTMPOSE_FACE_ONNX, providers=providers)

        # Inspect I/O to confirm expected shape & make it available downstream.
        input_info = session.get_inputs()[0]
        output_infos = [(o.name, o.shape) for o in session.get_outputs()]
        _logger.info(
            "RTMPose-Face I/O: input %s %s  outputs %s",
            input_info.name, input_info.shape, output_infos,
        )

        return ({
            "session": session,
            "input_name": input_info.name,
            "input_shape": input_info.shape,  # e.g. [N, 3, 256, 256] or dynamic
            "output_names": [o.name for o in session.get_outputs()],
            "providers": providers,
            "onnx_path": RTMPOSE_FACE_ONNX,
        },)
