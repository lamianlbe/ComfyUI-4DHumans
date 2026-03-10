import os
import torch
import numpy as np
import comfy.model_management
import comfy.utils

from ..utils.process_humans import load_image
from ..humans4d.hmr2.utils import recursive_to
from ..humans4d.hmr2.utils.geometry import perspective_projection
from ..humans4d.hmr2.utils.renderer import cam_crop_to_full
from ..humans4d.hmr2.utils.render_openpose import render_openpose


def _predict_openpose_keypoints(batch, model, model_cfg, device, img_w, img_h, clip_boundary):
    """
    Run HMR2 on a batch of person crops and return 2D OpenPose keypoints
    projected into full-image pixel coordinates.

    Args:
        clip_boundary (float): Keypoints projected outside the image boundary
            extended by this many pixels are set to confidence=0 and will not
            be rendered.  Use 0 for strict clipping at the image edge, or a
            larger value to allow slightly off-screen keypoints.
            A negative value disables clipping entirely (all confidence=1).

    Returns:
        np.ndarray: shape (N_persons, 25, 3) with columns (x, y, confidence).
    """
    with torch.no_grad():
        out = model(batch)

    pred_cam = out["pred_cam"]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()

    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH
        / model_cfg.MODEL.IMAGE_SIZE
        * img_size.max()
    )

    pred_cam_t_full = cam_crop_to_full(
        pred_cam, box_center, box_size, img_size, scaled_focal_length
    )  # (B, 3)

    # pred_keypoints_3d: (B, N_joints, 3); first 25 match OpenPose layout
    pred_keypoints_3d = out["pred_keypoints_3d"].float()
    batch_size = pred_keypoints_3d.shape[0]

    fl_val = scaled_focal_length.item()
    fl_2d = torch.full((batch_size, 2), fl_val, device=device, dtype=torch.float32)

    cx = img_size[0, 0].item() / 2.0
    cy = img_size[0, 1].item() / 2.0
    cam_center = torch.tensor([[cx, cy]] * batch_size, device=device, dtype=torch.float32)

    kp_2d = perspective_projection(
        pred_keypoints_3d,
        translation=pred_cam_t_full.float(),
        focal_length=fl_2d,
        camera_center=cam_center,
    )  # (B, N, 2)

    openpose_xy = kp_2d[:, :25, :].detach().cpu().numpy()  # (B, 25, 2)
    confidence = np.ones((batch_size, 25, 1), dtype=np.float32)

    if clip_boundary >= 0:
        lo_x, hi_x = -clip_boundary, img_w + clip_boundary
        lo_y, hi_y = -clip_boundary, img_h + clip_boundary
        out_of_frame = (
            (openpose_xy[:, :, 0] < lo_x) | (openpose_xy[:, :, 0] > hi_x) |
            (openpose_xy[:, :, 1] < lo_y) | (openpose_xy[:, :, 1] > hi_y)
        )  # (B, 25)
        confidence[out_of_frame] = 0.0

    return np.concatenate([openpose_xy, confidence], axis=-1)  # (B, 25, 3)


class HumanPoseControlNetNode:
    """
    Runs 4DHumans (HMR2) on every frame in an IMAGE batch, projects the
    predicted 3D skeleton into 2D, and renders it in OpenPose style on a
    black canvas.  The output is an IMAGE batch ready to be used as a
    ControlNet openpose condition.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "hmr": ("HMR",),
                "detectron": ("DETECTRON",),
                "clip_boundary": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 512.0,
                    "step": 1.0,
                    "tooltip": (
                        "Keypoints projected outside the image boundary by more than "
                        "this many pixels are hidden (confidence=0). "
                        "0 = strict edge clipping. "
                        "Increase to allow slightly off-screen keypoints to still be drawn. "
                        "Set to -1 to disable clipping entirely."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, hmr, detectron, clip_boundary):
        device = comfy.model_management.get_torch_device()
        model, model_cfg = hmr["model"], hmr["model_cfg"]
        model = model.to(device)
        detectron.model.to(device)

        # images: (B, H, W, 3) float [0,1]  – ComfyUI convention
        images_nchw = images.permute(0, 3, 1, 2)  # (B, C, H, W)
        B, C, height, width = images_nchw.shape

        pose_images = []
        pbar = comfy.utils.ProgressBar(B)

        for img_tensor in images_nchw:
            # Convert single frame to HWC uint8 numpy for detectron
            img_np = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()

            # Black canvas for ControlNet pose rendering
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            dataloader = load_image(model_cfg, img_np, detectron)
            for batch in dataloader:
                batch = recursive_to(batch, device)
                persons_kp = _predict_openpose_keypoints(
                    batch, model, model_cfg, device,
                    img_w=width, img_h=height, clip_boundary=clip_boundary,
                )
                # persons_kp: (N_persons, 25, 3)
                for person_kp in persons_kp:
                    canvas = render_openpose(canvas, person_kp)

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0)
            )
            pbar.update(1)

        return (torch.stack(pose_images),)
