"""
Render Sapiens pose keypoints as DWPose-style skeleton images.

Uses the existing ``draw_pose()`` renderer from ``scail/draw_pose_utils.py``
after converting Sapiens keypoints to DWPose dict format.
"""

import numpy as np

from .sapiens_inference import flat137_to_dwpose, goliath_to_dwpose
from .scail.draw_pose_utils import draw_pose


def render_sapiens_dwpose(canvas, kp137_or_dwpose, img_h, img_w):
    """
    Render a single person's Sapiens keypoints on *canvas*.

    Parameters
    ----------
    canvas : ndarray (H, W, 3) uint8
    kp137_or_dwpose : ndarray (137, 3)  our flat Sapiens format
                      OR dict  already in DWPose format
    img_h, img_w : image dimensions (for coordinate normalisation)

    Returns
    -------
    canvas : ndarray (H, W, 3) uint8  with skeleton drawn on it
    """
    if isinstance(kp137_or_dwpose, dict):
        pose_dict = kp137_or_dwpose
    else:
        pose_dict = flat137_to_dwpose(kp137_or_dwpose, img_h, img_w)

    rendered = draw_pose(
        pose_dict, img_h, img_w,
        show_feet=False,
        show_body=True,
        show_hand=True,
        show_face=True,
        optimized_face=False,
        face_scale=0.75,
    )

    # Overlay rendered skeleton onto canvas (non-zero pixels)
    mask = rendered > 0
    canvas[mask] = rendered[mask]
    return canvas
