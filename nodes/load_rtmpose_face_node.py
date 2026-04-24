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
                    ["cpu", "cuda"],
                    {
                        "default": "cpu",
                        "tooltip": (
                            "onnxruntime execution provider. Default is "
                            "'cpu' because the CUDAExecutionProvider path "
                            "has been observed to produce "
                            "bilaterally-mirrored 106-pt output on every "
                            "other frame for RTMPose-Face even with "
                            "deterministic session options; CPU is known "
                            "good. Try 'cuda' only if you need the speed "
                            "and have verified stable output on your "
                            "hardware."
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

        # We observed a reproducible alternating-frame bug with
        # CUDAExecutionProvider on MMPose's RTMPose-Face ONNX: on 4
        # consecutive calls the 106-pt output on slots 0 and 2 was
        # correct while slots 1 and 3 were a ~bilaterally-mirrored
        # version.  The input bboxes were essentially identical, so
        # the root cause is almost certainly CUDA EP non-determinism
        # (cuDNN algo search / memory pattern reuse picking different
        # kernels across calls, which for a near-symmetric face nudges
        # SimCC argmax between the true peak and its mirror peak).
        #
        # Neutralise by locking every knob that can introduce
        # call-to-call variation:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # These two are the big ones for non-determinism across
        # consecutive session.run() calls:
        #   enable_mem_pattern: pre-plans tensor allocations from the
        #       first run and reuses that pattern. If the second run
        #       lands slightly different, the reused layout can cause
        #       stale data to leak into the next op.
        #   enable_cpu_mem_arena: similar pooling on CPU.
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        # Keep optimisations but avoid aggressive fusions that could
        # bake in assumptions about repeated buffers.
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        if provider == "cuda":
            # Pin cuDNN to a fixed conv algorithm ("DEFAULT" = cuDNN's
            # default, not a per-input heuristic search). HEURISTIC /
            # EXHAUSTIVE pick different kernels depending on workspace
            # availability, which is a common source of
            # call-to-call output drift on near-identical inputs.
            cuda_provider_options = {
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
                "arena_extend_strategy": "kNextPowerOfTwo",
            }
            providers = [
                ("CUDAExecutionProvider", cuda_provider_options),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        _logger.info(
            "Loading RTMPose-Face ONNX from %s (providers=%s, sequential, "
            "single-threaded)",
            RTMPOSE_FACE_ONNX, providers,
        )
        session = ort.InferenceSession(
            RTMPOSE_FACE_ONNX, sess_options=sess_options, providers=providers,
        )

        # Inspect I/O to confirm expected shape & make it available downstream.
        input_info = session.get_inputs()[0]
        output_infos = [(o.name, o.shape) for o in session.get_outputs()]
        _logger.info(
            "RTMPose-Face I/O: input %s %s  outputs %s",
            input_info.name, input_info.shape, output_infos,
        )

        # Warn when ONNX was exported with a fixed batch dimension
        # (common for MMPose RTMPose deployment exports); the
        # inference code will fall back to one-frame-at-a-time runs.
        if input_info.shape and isinstance(input_info.shape[0], int) \
                and input_info.shape[0] >= 1:
            _logger.info(
                "RTMPose-Face ONNX uses STATIC batch=%d; will run one "
                "frame per session.run call.",
                input_info.shape[0],
            )

        return ({
            "session": session,
            "input_name": input_info.name,
            "input_shape": input_info.shape,  # e.g. [N, 3, 256, 256] or dynamic
            "output_names": [o.name for o in session.get_outputs()],
            "providers": providers,
            "onnx_path": RTMPOSE_FACE_ONNX,
        },)
