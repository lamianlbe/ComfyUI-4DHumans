"""
Sapiens 2D Human Pose node.

Accepts RGB images, per-person masks (frame-grouped), and a Sapiens model.
For each frame and each person, the image is masked (regions outside the
person's mask are blacked out) before running Sapiens inference.

Output is a POSE_2D dict with per-person, per-frame COCO-WholeBody 133
keypoints, structured identically to POSE_3D for easy downstream pairing.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox


def _mask_to_bbox(mask_np):
    """Convert a binary mask (H, W) to a [x1, y1, x2, y2] bbox array."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class SapiensPoseNode:
    """Sapiens 2D whole-body pose estimation (mask-based, multi-person)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "sapiens": ("SAPIENS",),
            },
        }

    RETURN_TYPES = ("POSE_2D",)
    RETURN_NAMES = ("pose_2d",)
    FUNCTION = "estimate_pose"
    CATEGORY = "4dhumans"

    def estimate_pose(self, images, masks, sapiens):
        B, img_h, img_w, _C = images.shape

        # masks: (M, H, W) frame-grouped: frame1's N persons, frame2's N, ...
        if masks.dim() == 4:
            masks = masks[..., 0]
        M = masks.shape[0]
        if M % B != 0:
            raise ValueError(
                f"Sapiens 2D Human Pose: mask count ({M}) must be a "
                f"multiple of frame count ({B}). Got {M} masks for {B} frames."
            )
        n_persons = M // B
        masks = masks.reshape(B, n_persons, masks.shape[-2], masks.shape[-1])

        pbar = comfy.utils.ProgressBar(B * n_persons)

        # Per-person storage
        persons = []
        for _ in range(n_persons):
            persons.append({"keypoints": [None] * B})

        for t in range(B):
            img_np = (images[t] * 255).byte().cpu().numpy()  # (H, W, 3)

            for p_idx in range(n_persons):
                mask_frame = masks[t, p_idx].cpu().numpy()

                bbox = _mask_to_bbox(mask_frame)
                if bbox is None:
                    pbar.update(1)
                    continue

                # Apply mask: black out regions outside person
                mask_bool = mask_frame > 0.5
                masked_img = img_np.copy()
                masked_img[~mask_bool] = 0

                # Run Sapiens on masked image with person bbox
                result = run_sapiens_on_bbox(masked_img, bbox, sapiens)
                if result is not None:
                    persons[p_idx]["keypoints"][t] = result["pixel_kp"]  # (133, 3)

                pbar.update(1)

        pose_2d = {
            "n_persons": n_persons,
            "n_frames": B,
            "img_h": img_h,
            "img_w": img_w,
            "persons": persons,
        }

        return (pose_2d,)
