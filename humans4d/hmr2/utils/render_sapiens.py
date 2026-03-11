"""
Render Sapiens COCO-WholeBody 133-keypoint pose as DWPose-style skeleton.

Uses the existing ``draw_pose()`` renderer from ``scail/draw_pose_utils.py``
after converting COCO-WholeBody keypoints to DWPose dict format.
"""

import numpy as np

from .sapiens_inference import coco_wb_to_dwpose
from .scail.draw_pose_utils import draw_pose


def render_sapiens_dwpose(canvas, kp_data, img_h, img_w):
    """
    Render a single person's Sapiens keypoints on *canvas*.

    Parameters
    ----------
    canvas : ndarray (H, W, 3) uint8
    kp_data : ndarray (133, 3)  COCO-WholeBody pixel keypoints
              OR dict  already in DWPose format
    img_h, img_w : image dimensions (for coordinate normalisation)

    Returns
    -------
    canvas : ndarray (H, W, 3) uint8  with skeleton drawn on it
    """
    if isinstance(kp_data, dict):
        pose_dict = kp_data
    else:
        pose_dict = coco_wb_to_dwpose(kp_data, img_h, img_w)

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
