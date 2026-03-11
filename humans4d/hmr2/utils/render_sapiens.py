"""
Render Sapiens pose keypoints as DWPose-style skeleton images.

Uses the existing ``draw_pose()`` renderer from ``scail/draw_pose_utils.py``
after converting Sapiens keypoints to DWPose dict format.

Supports:
- COCO-WholeBody 133 keypoints (preferred — face is native iBUG 68)
- Goliath 308+ keypoints (legacy — requires complex face mapping)
- Flat 137 intermediate format (legacy Goliath timeline)
"""

import numpy as np

from .sapiens_inference import (
    coco_wb_to_dwpose, flat137_to_dwpose, goliath_to_dwpose,
)
from .scail.draw_pose_utils import draw_pose


def render_sapiens_dwpose(canvas, kp_data, img_h, img_w):
    """
    Render a single person's Sapiens keypoints on *canvas*.

    Parameters
    ----------
    canvas : ndarray (H, W, 3) uint8
    kp_data : ndarray (133, 3)  COCO-WholeBody pixel keypoints
              OR ndarray (137, 3)  flat Sapiens format (legacy Goliath)
              OR ndarray (N, 3) with N>137  raw Goliath pixel keypoints
              OR dict  already in DWPose format
    img_h, img_w : image dimensions (for coordinate normalisation)

    Returns
    -------
    canvas : ndarray (H, W, 3) uint8  with skeleton drawn on it
    """
    if isinstance(kp_data, dict):
        pose_dict = kp_data
    elif kp_data.shape[0] == 133:
        # COCO-WholeBody — face is already iBUG 68, no mapping needed
        pose_dict = coco_wb_to_dwpose(kp_data, img_h, img_w)
    elif kp_data.shape[0] > 137:
        # Raw Goliath pixel keypoints (legacy)
        pose_dict = goliath_to_dwpose(kp_data, img_h, img_w)
    else:
        # Flat 137 format (legacy Goliath timeline)
        pose_dict = flat137_to_dwpose(kp_data, img_h, img_w)

    rendered = draw_pose(
        pose_dict, img_h, img_w,
        show_feet=False,
        show_body=True,
        show_hand=True,
        show_face=True,
        optimized_face=True,
        face_scale=0.75,
    )

    # Overlay rendered skeleton onto canvas (non-zero pixels)
    mask = rendered > 0
    canvas[mask] = rendered[mask]

    return canvas
