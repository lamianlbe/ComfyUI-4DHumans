"""
Load BBoxMaskPose (BMP) — iterative detect → pose → SAM2-mask pipeline
tuned for occluded crowd scenes.

Paper: Purkrabek et al., ICCV 2025 (arxiv:2412.01562), SOTA on OCHuman
for 2D pose (49.2 AP) and human-instance segmentation (34 AP).

Unlike our other segmenters (SAM3 / YOLO-seg / CrowdSAM), BMP gives us
**three outputs from a single forward**: per-instance bboxes, masks,
AND 17-point COCO keypoints — because its internal SAM2 is prompted
by the pose keypoints (not bboxes), so pose is a load-bearing part
of the mask generation path, not an afterthought.

BMP's relative strengths based on our tests:
  * 2D keypoints:   strong (beats YOLO11m-Pose on occluded frames)
  * Masks:          good for typical separation, but merges people
                    when RTMDet can't draw two bboxes for tightly
                    overlapping bodies
  * Speed:          ~200-500 ms / frame on Blackwell class GPUs

This node loads the full 4-model stack (RTMDet-ins-L detector,
SAM 2.1 Hiera base+ for pose-prompted segmentation, PMPose-b for
pose estimation, plus SAM 2.1's shared backbone) and returns a
single dict downstream nodes consume.

Prerequisites (pip installed into your ComfyUI env):
    - mmcv >= 2.2.0
    - mmdet >= 3.3.0
    - mmpose >= 1.3.1
    - mmpretrain (transient dep of mmpose registry)
    - bboxmaskpose  (pip install from github.com/MiraPurkrabek/BBoxMaskPose)
    - sam2 (Meta's SAM 2.1, pulled by BMP)

Weights (auto-downloaded from HuggingFace on first use, cached under
~/.cache/huggingface/hub/), no manual placement required. If offline,
pre-populate the cache or set HF_HOME.

License note: BMP is GPL-3.0. By loading this node, your ComfyUI
graph inherits that license for any derived works distributed with
BMP weights / inference results.
"""

import logging

_logger = logging.getLogger(__name__)


# BMP config aliases shipped in bboxmaskpose/configs/. Each trades off
# how aggressively SAM2 is prompted (num_pos_keypoints) vs recall.
_BMP_CONFIGS = ["bmp_v2", "bmp_D3", "bmp_J1"]

# PMPose variants shipped in HF. 'b' is the default (best recall/speed
# balance per the paper); 's'/'l'/'h' trade accuracy for speed.
_PMPOSE_VARIANTS = [
    "PMPose-s", "PMPose-b", "PMPose-l", "PMPose-h",
    "MaskPose-s", "MaskPose-b", "MaskPose-l", "MaskPose-h",
]


class LoadBMPNode:
    """Load BBoxMaskPose full pipeline (RTMDet + SAM2.1 + PMPose)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. BMP is heavy (~4 GiB VRAM "
                            "peak for the three-model stack); CPU works "
                            "but is effectively unusable for video."
                        ),
                    },
                ),
                "config": (
                    _BMP_CONFIGS,
                    {
                        "default": "bmp_v2",
                        "tooltip": (
                            "BMP iteration config. bmp_v2 (default) is "
                            "2 iterations with 3 positive keypoint "
                            "prompts per SAM2 call — the paper's "
                            "recommended OCHuman setting. bmp_D3 and "
                            "bmp_J1 are legacy variants with slightly "
                            "different prompt counts."
                        ),
                    },
                ),
                "pose_variant": (
                    _PMPOSE_VARIANTS,
                    {
                        "default": "PMPose-b",
                        "tooltip": (
                            "Pose model weight variant. 'b' is the "
                            "balanced default. 'PMPose-*' gives full "
                            "presence/visibility/keypoint outputs; "
                            "'MaskPose-*' is the predecessor (no "
                            "presence/visibility probabilities but "
                            "faster). Small→huge: s/b/l/h."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("BMP",)
    RETURN_NAMES = ("bmp",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, config, pose_variant):
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch required") from e

        # Lazy-import the heavy packages so node registration stays cheap
        # even if BMP isn't installed yet — user sees the error only when
        # they try to load.
        try:
            from bboxmaskpose import BBoxMaskPose
            from pmpose import PMPose
        except ImportError as e:
            raise ImportError(
                "BBoxMaskPose is not installed in this Python env. "
                "Install with:\n"
                "  pip install \"git+https://github.com/MiraPurkrabek/BBoxMaskPose.git\"\n"
                "BMP also requires mmcv/mmdet/mmpose/mmpretrain installed "
                "and compatible with your torch + CUDA. See project README."
            ) from e

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadBMP: requested CUDA but torch.cuda.is_available() is "
                "False — falling back to CPU (BMP will be unusably slow)."
            )

        _logger.info(
            "Loading BMP stack: config=%s pose_variant=%s device=%s",
            config, pose_variant, device_str,
        )

        # Step 1: PMPose. Construct explicitly so the user-chosen variant
        # takes effect; if we pass pose_model=None, BBoxMaskPose builds
        # the variant baked into the YAML config instead.
        pose_model = PMPose(
            device=device_str,
            variant=pose_variant,
            from_pretrained=True,
        )

        # Step 2: BBoxMaskPose wires RTMDet + SAM 2.1 + PMPose together
        # and loads all three onto `device_str`. First call will fetch
        # ~1.5-2 GB of HF weights if not cached.
        bmp_model = BBoxMaskPose(
            config=config,
            device=device_str,
            pose_model=pose_model,
        )

        _logger.info(
            "BMP ready. Run BMPInstanceSegmentation downstream to do "
            "per-frame inference + cross-frame tracking."
        )

        return ({
            "bmp":           bmp_model,
            "pose_model":    pose_model,
            "device":        device_str,
            "config":        config,
            "pose_variant":  pose_variant,
        },)
