"""
Load ViTPose via onnxruntime — alternative backend to LoadViTPoseNode.

Used when the user has a pre-exported ONNX model and onnxruntime-gpu
configured for their environment (incl. cu13 / Blackwell). Same
``VITPOSE`` output contract as the HF transformers loader, so
BMPRTMWPose's fallback path consumes either backend transparently.

The runtime impl is self-contained:
  - Top-down affine crop (cv2.getAffineTransform / warpAffine) replaces
    transformers' VitPoseImageProcessor.
  - ImageNet mean/std normalisation matches mmpose ViTPose convention.
  - Heatmap argmax decode (no DARK refinement — fallback fires on
    extreme cases where sub-pixel precision matters less than recall).
  - Inverse affine maps keypoints from heatmap → input crop → original
    image coords.

Input/output shapes are introspected from the ONNX session (works for
both 256×192 and 384×288 ViTPose exports, and for both 17-keypoint
body-only and 133-keypoint wholebody models). Wholebody output's first
17 keypoints follow COCO-17 ordering, which is what we use for the
body-fallback path in BMPRTMWPose.

Hardcoded weight path (ComfyUI convention):
    models/vitpose/<your_model>.onnx   (with the sibling .bin if
                                         the model uses external data)

The user's known files:
    models/detection/onnx/vitpose_h_wholebody_model.onnx
    models/detection/onnx/vitpose_h_wholebody_data.bin

You provide the path via the ``onnx_path`` input — accepts both
absolute paths and paths relative to ``ComfyUI/models/``.
"""

import logging
import os

import numpy as np

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


def _resolve_onnx_path(path: str) -> str:
    """Accept absolute paths and ``models/...`` relative paths.

    ComfyUI users typically refer to model files by their relative path
    under ``models/``, so we normalise that into an absolute path
    rooted at ``models_dir``.
    """
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    # Try relative to models/
    rel = os.path.join(models_dir, path)
    if os.path.isfile(rel):
        return rel
    # Try as-is (cwd-relative)
    if os.path.isfile(path):
        return os.path.abspath(path)
    return path  # caller will raise FileNotFoundError below


class LoadViTPoseONNXNode:
    """Load ViTPose from an ONNX file via onnxruntime-gpu."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "onnx_path": (
                    "STRING",
                    {
                        "default": "detection/onnx/vitpose_h_wholebody_model.onnx",
                        "tooltip": (
                            "Path to the ONNX model file. Accepts "
                            "absolute paths or paths relative to "
                            "ComfyUI/models/. If the model uses "
                            "external data (e.g. a sibling .bin "
                            "file for >2GB models), keep them next "
                            "to the .onnx — onnxruntime auto-finds "
                            "external data by filename convention."
                        ),
                    },
                ),
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. CUDA uses "
                            "CUDAExecutionProvider; falls back to "
                            "CPU if CUDA EP isn't available in the "
                            "current onnxruntime build."
                        ),
                    },
                ),
                "use_tensorrt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Try TensorRT execution provider before "
                            "CUDA EP. Faster inference (1.5-2x) but "
                            "first-load engine build can take 30-60s. "
                            "Requires onnxruntime-gpu built with "
                            "TensorRT support and matching TRT "
                            "version installed."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("VITPOSE",)
    RETURN_NAMES = ("vitpose",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, onnx_path, device, use_tensorrt):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is not installed. Install:\n"
                "  pip install onnxruntime-gpu  (for CUDA)\n"
                "  pip install onnxruntime      (CPU only)"
            ) from e

        resolved_path = _resolve_onnx_path(onnx_path)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"ViTPose ONNX file not found:\n  {resolved_path}\n\n"
                f"Place it under ComfyUI/models/ and reference it by "
                f"relative path, e.g.:\n"
                f"  detection/onnx/vitpose_h_wholebody_model.onnx\n\n"
                f"For models > 2 GB the export usually creates a "
                f"sibling .bin file containing external weights — "
                f"keep them in the same directory."
            )

        # Build provider list. Order matters — onnxruntime tries them
        # in sequence and falls back if one fails.
        providers = []
        available = ort.get_available_providers()
        if device == "cuda":
            if use_tensorrt and "TensorrtExecutionProvider" in available:
                # TRT typically wants a cache dir for engine build artefacts
                trt_cache = os.path.join(
                    models_dir, "vitpose", "_trt_cache",
                )
                os.makedirs(trt_cache, exist_ok=True)
                providers.append(("TensorrtExecutionProvider", {
                    "trt_engine_cache_enable":   True,
                    "trt_engine_cache_path":     trt_cache,
                    "trt_fp16_enable":           True,
                }))
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")  # always fallback

        _logger.info(
            "Loading ViTPose ONNX from %s with providers=%s",
            resolved_path, [p[0] if isinstance(p, tuple) else p
                            for p in providers],
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            resolved_path, sess_options=sess_options, providers=providers,
        )

        # Introspect input/output shapes. Common ViTPose exports:
        #   input:  (1, 3, H, W)  with H=256/W=192 or H=384/W=288
        #   output: (1, K, H/4, W/4) heatmap, K=17 (body) or 133 (wholebody)
        in_shape = session.get_inputs()[0].shape
        out_shape = session.get_outputs()[0].shape
        # Some exports use dynamic axes (-1 / "batch"). Fall back to
        # standard ViTPose dims if the export is shape-agnostic.
        input_h = int(in_shape[2]) if isinstance(in_shape[2], int) and in_shape[2] > 0 else 256
        input_w = int(in_shape[3]) if isinstance(in_shape[3], int) and in_shape[3] > 0 else 192
        num_kpts = int(out_shape[1]) if isinstance(out_shape[1], int) and out_shape[1] > 0 else 17

        active_provider = session.get_providers()[0]
        _logger.info(
            "ViTPose ONNX ready. input shape=(B, 3, %d, %d), "
            "output keypoints=%d, active provider=%s",
            input_h, input_w, num_kpts, active_provider,
        )

        return ({
            "backend":     "onnx",
            "session":     session,
            "input_name":  session.get_inputs()[0].name,
            "input_h":     input_h,
            "input_w":     input_w,
            "num_kpts":    num_kpts,
            "device":      "cuda" if "CUDA" in active_provider or "Tensorrt" in active_provider else "cpu",
            "onnx_path":   resolved_path,
        },)
