"""
Sapiens Multiple Person Pose Tracking node.

Uses PHALP for multi-person detection/tracking, then runs Sapiens
COCO-WholeBody 133-keypoint pose estimation on each detected person.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox
from .load_phalp_node import _ensure_phalp_importable


class SapiensMultiPoseNode:
    """Multi-person Sapiens pose estimation via PHALP tracking."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sapiens": ("SAPIENS",),
                "phalp": ("PHALP",),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Overlay skeletons on the original frame "
                            "instead of a black canvas."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, sapiens, phalp, debug):
        if not _ensure_phalp_importable():
            raise RuntimeError("phalp package not found. See 'Load PHALP' node.")

        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape

        new_size = max(img_h, img_w)
        left = (new_size - img_w) // 2
        top = (new_size - img_h) // 2
        measurements = [img_h, img_w, new_size, left, top]

        tracker = phalp["tracker"]
        tracker.setup_deepsort()

        pbar = comfy.utils.ProgressBar(B)
        pose_images = []

        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
            frame_name = str(t)

            # Detect people
            (pred_bbox, pred_bbox_pad, pred_masks,
             pred_scores, pred_classes,
             gt_tids, gt_annots) = tracker.get_detections(
                img_np, frame_name, t, {}, measurements)

            # Run Sapiens on each detection
            sapiens_results = []
            for i in range(len(pred_bbox)):
                result = run_sapiens_on_bbox(img_np, pred_bbox[i], sapiens)
                sapiens_results.append(result)

            # Update tracker state (required for temporal consistency)
            extra_data = tracker.run_additional_models(
                img_np, pred_bbox, pred_masks, pred_scores,
                pred_classes, frame_name, t, measurements, gt_tids, gt_annots)
            detections = tracker.get_human_features(
                img_np, pred_masks, pred_bbox, pred_bbox_pad,
                pred_scores, frame_name, pred_classes, t,
                measurements, gt_tids, gt_annots, extra_data)
            tracker.tracker.predict()
            tracker.tracker.update(detections, t, frame_name, shot=0)

            # Render all detected people
            if debug:
                canvas = img_np.copy()
            else:
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            for result in sapiens_results:
                if result is not None:
                    canvas = render_sapiens_dwpose(
                        canvas, result["pixel_kp"], img_h, img_w)

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0))
            pbar.update(1)

        return (torch.stack(pose_images),)
