"""
PromptHMR 3D Human Pose Estimation node.

Accepts an image batch with per-frame masks and runs PromptHMR to recover
3D human mesh (SMPL-X). Renders the projected 2D body joints as an
OpenPose-style skeleton overlay.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_openpose import render_openpose


def _mask_to_bbox(mask_np):
    """Convert a binary mask (H, W) to a [x1, y1, x2, y2, score] bbox tensor."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return torch.tensor([[x1, y1, x2, y2, 1.0]], dtype=torch.float32)


class PromptHMRPoseNode:
    """PromptHMR single-person 3D pose estimation with mask prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "prompthmr": ("PROMPTHMR",),
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

    def render_pose(self, images, masks, prompthmr, debug):
        from .load_prompthmr_node import _ensure_lib_importable
        _ensure_lib_importable()

        from torch.amp import autocast
        from prompt_hmr.models.inference import prepare_batch

        model = prompthmr["model"]
        img_size = prompthmr["img_size"]

        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape

        # Ensure masks shape is (B, H, W)
        if masks.dim() == 4:
            masks = masks[:, 0]

        pbar = comfy.utils.ProgressBar(2 * B)

        # Pass 1: inference per frame
        body_joints_2d_list = [None] * B

        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().cpu().numpy()
            mask_np = masks[t].cpu().numpy()

            # Derive bbox from mask
            bbox = _mask_to_bbox(mask_np)
            if bbox is None:
                pbar.update(1)
                continue

            # Prepare mask for PromptHMR: binary uint8 (H, W)
            mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255

            inputs = [{
                "image_cv": img_np,
                "boxes": bbox,
                "masks": np.array([mask_uint8]),
                "text": None,
            }]

            with torch.no_grad(), autocast("cuda"):
                batch = prepare_batch(inputs, img_size=img_size, interaction=False)
                output = model(batch, use_mean_hands=True)[0]

            # Get 2D projected body joints: (N_persons, J, 2)
            joints_2d = output["body_joints2d"].detach().cpu().numpy()

            # Unscale from padded img_size coords back to original image coords
            scale = batch[0]["image_scale"]
            offset = batch[0]["image_offset"]
            if isinstance(offset, torch.Tensor):
                offset = offset.numpy()
            offset = np.array(offset)

            # joints_2d is in padded/scaled space, undo that
            joints_2d_orig = (joints_2d - offset[None, None, :]) / scale

            # Use first person detection
            kp2d = joints_2d_orig[0]  # (J, 2)
            n_joints = kp2d.shape[0]
            # Add confidence = 1.0 for all joints
            kp_with_conf = np.concatenate(
                [kp2d, np.ones((n_joints, 1), dtype=np.float32)], axis=1
            )

            # PromptHMR body_joints2d has 25 OpenPose joints
            body_joints_2d_list[t] = kp_with_conf

            pbar.update(1)

        # Pass 2: render
        pose_images = []
        for t in range(B):
            if debug:
                canvas = (
                    images_nchw[t].permute(1, 2, 0) * 255
                ).byte().cpu().numpy().copy()
            else:
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            if body_joints_2d_list[t] is not None:
                canvas = render_openpose(canvas, body_joints_2d_list[t])

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0)
            )
            pbar.update(1)

        return (torch.stack(pose_images),)
