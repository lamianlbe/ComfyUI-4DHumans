"""
Sapiens Multiple Person Pose Tracking node.

Uses PHALP for multi-person detection/tracking, then runs Sapiens
COCO-WholeBody 133-keypoint pose estimation on each detected person.

Supports FPS interpolation to 30fps using nearest-frame selection.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox
from .load_phalp_node import _ensure_phalp_importable
from ._pose_utils import compute_resampled_indices


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
                "interpolate_30fps": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Resample output to 30fps using nearest-frame "
                            "selection. Requires fps input."
                        ),
                    },
                ),
            },
            "optional": {
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": (
                            "Input video FPS. Used when interpolate_30fps "
                            "is enabled."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, sapiens, phalp, debug,
                    interpolate_30fps=False, fps=24.0):
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

        pbar = comfy.utils.ProgressBar(2 * B)

        # Pass 1: detection + inference
        frame_results = []  # per-frame list of Sapiens results
        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
            frame_name = str(t)

            (pred_bbox, pred_bbox_pad, pred_masks,
             pred_scores, pred_classes,
             gt_tids, gt_annots) = tracker.get_detections(
                img_np, frame_name, t, {}, measurements)

            sapiens_results = []
            for i in range(len(pred_bbox)):
                result = run_sapiens_on_bbox(img_np, pred_bbox[i], sapiens)
                if result is not None:
                    sapiens_results.append(result)

            extra_data = tracker.run_additional_models(
                img_np, pred_bbox, pred_masks, pred_scores,
                pred_classes, frame_name, t, measurements, gt_tids, gt_annots)
            detections = tracker.get_human_features(
                img_np, pred_masks, pred_bbox, pred_bbox_pad,
                pred_scores, frame_name, pred_classes, t,
                measurements, gt_tids, gt_annots, extra_data)
            tracker.tracker.predict()
            tracker.tracker.update(detections, t, frame_name, shot=0)

            frame_results.append(sapiens_results)
            pbar.update(1)

        # FPS resampling
        do_resample = interpolate_30fps and fps > 0 and abs(fps - 30.0) > 0.1
        if do_resample:
            src_indices = compute_resampled_indices(B, fps, 30.0)
            n_out = len(src_indices)
        else:
            src_indices = list(range(B))
            n_out = B

        # Pass 2: render
        pose_images = []
        for out_t in range(n_out):
            src_t = src_indices[out_t]
            if debug:
                canvas = (images_nchw[src_t].permute(1, 2, 0)
                          * 255).byte().numpy().copy()
            else:
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            for result in frame_results[src_t]:
                canvas = render_sapiens_dwpose(
                    canvas, result["pixel_kp"], img_h, img_w)

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0))
            pbar.update(1)

        return (torch.stack(pose_images),)
