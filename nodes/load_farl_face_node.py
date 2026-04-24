"""
Load pyfacer's RetinaFace face detector + FaRL 68-point face aligner.

This is the exact pipeline pyfacer's own samples use
(``samples/face_alignment.ipynb``): RetinaFace produces a
``(rects, points, scores)`` dict per image, then FaRL's face_aligner
consumes the 5-point landmarks to compute its similarity alignment to
a canonical 448×448 face template and regresses 68 heatmaps.

Why run RetinaFace instead of deriving a 5-point array from MHR head
keypoints:

  * The 5-point input needs IMAGE-space L/R eye ordering plus two real
    mouth corners. MHR only gives us nose + eyes + ears, so the mouth
    corners would have to be SYNTHESIZED from canonical ratios — that
    only works for perfectly front-facing upright subjects and
    silently degrades for any head roll / profile / tilt.
  * RetinaFace is trained on WIDER FACE and outputs real 5-point
    landmarks for any pose; pyfacer's FaRLFaceAlignment is trained
    against the SAME detector's output distribution, so there is zero
    convention mismatch to manage.
  * RetinaFace MobileNet-0.25 is ~1.7 MB and runs in <5 ms/frame, so
    the extra pass is effectively free.

Why FaRL over RTMPose-Face: RTMPose's SimCC argmax decoder produces
bilaterally-mirrored landmarks on alternating frames of a nearly
stationary face (bimodal heatmap peaks: true peak + mirror ghost).
FaRL's ViT-B backbone + 448×448 input + heatmap-regression head is
structurally robust against the mirror-ghost failure mode.
Additionally, FaRL outputs 68 landmarks in the standard 300W (iBUG)
convention — a perfect match for COCO-WholeBody indices 23..90 with
zero remapping.

Hardcoded paths (ComfyUI convention — no torch hub auto-download):
    models/farl/face_alignment.farl.ibug300w.main_ema_jit.pt  (~617 MB)
    models/farl/mobilenet0.25_Final.pth                       (~1.7 MB)

Download manually from:
    https://github.com/FacePerceiver/facer/releases/download/models-v1/face_alignment.farl.ibug300w.main_ema_jit.pt
    https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth

License: MIT (FaRL) + MIT (pyfacer) + MIT (RetinaFace).

Install:
    pip install pyfacer
"""

import logging
import os

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


# pyfacer model names. The 'farl/ibug300w/448' aligner is the only
# pyfacer FaRL variant that outputs 68-pt landmarks in the 300W/iBUG
# layout (other 'farl/...' names target parsing masks, not landmarks).
_FARL_FACE_ALIGNER_NAME   = "farl/ibug300w/448"
_RETINAFACE_DETECTOR_NAME = "retinaface/mobilenet"

# FaRL 68-pt aligner. Filename MUST still contain "jit" because
# pyfacer's loader branches on that substring to pick torch.jit.load vs
# torch.load — we keep the upstream filename verbatim.
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

# RetinaFace MobileNet-0.25 detector weights from the upstream
# elliottzheng/face-detection release — the same URL pyfacer uses
# internally when ``model_path`` is not provided. Passing our local
# path via kwargs keeps torch hub quiet.
RETINAFACE_CHECKPOINT_FILENAME = "mobilenet0.25_Final.pth"
RETINAFACE_CHECKPOINT_URL = (
    "https://github.com/elliottzheng/face-detection/releases/"
    "download/0.0.1/" + RETINAFACE_CHECKPOINT_FILENAME
)
RETINAFACE_CHECKPOINT = os.path.join(
    models_dir, "farl", RETINAFACE_CHECKPOINT_FILENAME,
)


class LoadFaRLFaceNode:
    """Load pyfacer's RetinaFace detector + FaRL 68-pt face aligner."""

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
                            "modern GPU; CPU works but is ~10× slower. "
                            "RetinaFace MobileNet rides along on the "
                            "same device (~1.7 MB, trivial overhead)."
                        ),
                    },
                ),
                "detection_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.99,
                        "step": 0.05,
                        "tooltip": (
                            "RetinaFace confidence threshold. 0.5 is a "
                            "safe default for fronts/3/4 profiles; "
                            "lower it (e.g. 0.3) to rescue hard "
                            "profile / occluded faces at the cost of "
                            "more false positives."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FARLFACE",)
    RETURN_NAMES = ("farl_face",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, detection_threshold):
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

        if not os.path.isfile(RETINAFACE_CHECKPOINT):
            raise FileNotFoundError(
                f"RetinaFace MobileNet checkpoint not found at:\n"
                f"  {RETINAFACE_CHECKPOINT}\n\n"
                f"Download it manually (~1.7 MB) from:\n"
                f"  {RETINAFACE_CHECKPOINT_URL}\n\n"
                f"and place it at the path above. Typical shell command:\n"
                f"  mkdir -p {os.path.dirname(RETINAFACE_CHECKPOINT)}\n"
                f"  wget -O {RETINAFACE_CHECKPOINT} \\\n"
                f"    {RETINAFACE_CHECKPOINT_URL}"
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

        # RetinaFace first (cheap). Passing ``model_path`` through to
        # RetinaFaceDetector → load_net → load_model routes the weights
        # through torch.load against our local file, so no torch hub
        # download ever happens.
        _logger.info(
            "Loading RetinaFace detector (%s) on device=%s from %s",
            _RETINAFACE_DETECTOR_NAME, device_str, RETINAFACE_CHECKPOINT,
        )
        face_detector = facer.face_detector(
            _RETINAFACE_DETECTOR_NAME,
            device=device_str,
            model_path=RETINAFACE_CHECKPOINT,
            threshold=float(detection_threshold),
        )
        try:
            face_detector.eval()
        except AttributeError:
            pass

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
            "FaRL face stack loaded. Detector: RetinaFace MobileNet-0.25 "
            "(5-pt landmarks, threshold=%.2f). Aligner: FaRL ViT-B, 68 "
            "pts in 300W/iBUG convention (matches COCO-WholeBody 23..90).",
            float(detection_threshold),
        )

        return ({
            "detector": face_detector,
            "aligner":  face_aligner,
            "device":   device_str,
            "detector_name": _RETINAFACE_DETECTOR_NAME,
            "detector_path": RETINAFACE_CHECKPOINT,
            "detection_threshold": float(detection_threshold),
            "aligner_name":  _FARL_FACE_ALIGNER_NAME,
            "aligner_path":  FARL_FACE_CHECKPOINT,
            # Back-compat: older callers looked at "model_name"/"model_path"
            # for the aligner; mirror them here so a stale integration
            # won't KeyError.
            "model_name": _FARL_FACE_ALIGNER_NAME,
            "model_path": FARL_FACE_CHECKPOINT,
        },)
