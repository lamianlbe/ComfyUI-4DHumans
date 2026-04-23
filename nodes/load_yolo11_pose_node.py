"""
Load YOLO11m-Pose (body-keypoint detector).

Used as the optional ``hand_box_source="yolo_pose"`` helper for
Fast SAM 3D Body — it provides the 17 COCO body keypoints so the SAM 3D
Body head can derive better hand bboxes than its internal body decoder.

Hardcoded path:
    models/fastsam3dbody/yolo11m-pose.pt
"""

import logging
import os

import torch

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


YOLO11_POSE_CKPT = os.path.join(
    models_dir, "fastsam3dbody", "yolo11m-pose.pt"
)


class LoadYOLO11PoseNode:
    """Load Ultralytics YOLO11m-Pose. No parameters — the only use case
    in this pipeline is feeding 17 keypoints to Fast SAM 3D Body's
    hand_box_source='yolo_pose' mode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("YOLO11POSE",)
    RETURN_NAMES = ("yolo11_pose",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self):
        if not os.path.isfile(YOLO11_POSE_CKPT):
            raise FileNotFoundError(
                f"YOLO11m-Pose checkpoint not found at: {YOLO11_POSE_CKPT}\n"
                f"Download yolo11m-pose.pt from Ultralytics and place it "
                f"at this exact location."
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with:\n"
                "  pip install -U ultralytics"
            ) from e

        _logger.info("Loading YOLO11m-Pose from %s", YOLO11_POSE_CKPT)
        model = YOLO(YOLO11_POSE_CKPT)

        return ({
            "model": model,
            "checkpoint_path": YOLO11_POSE_CKPT,
        },)
