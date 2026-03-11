"""
Render Sapiens pose keypoints as DWPose-style skeleton images.

Uses the existing ``draw_pose()`` renderer from ``scail/draw_pose_utils.py``
after converting Sapiens keypoints to DWPose dict format.
"""

import cv2
import numpy as np

from .sapiens_inference import (
    flat137_to_dwpose, goliath_to_dwpose, CONF_THRESHOLD,
)
from .scail.draw_pose_utils import draw_pose


def render_sapiens_dwpose(canvas, kp_data, img_h, img_w):
    """
    Render a single person's Sapiens keypoints on *canvas*.

    Parameters
    ----------
    canvas : ndarray (H, W, 3) uint8
    kp_data : ndarray (137, 3)  our flat Sapiens format
              OR ndarray (N, 3) with N>137  raw Goliath pixel keypoints
              OR dict  already in DWPose format
    img_h, img_w : image dimensions (for coordinate normalisation)

    Returns
    -------
    canvas : ndarray (H, W, 3) uint8  with skeleton drawn on it
    """
    if isinstance(kp_data, dict):
        pose_dict = kp_data
        raw_goliath = None
    elif kp_data.shape[0] > 137:
        # Raw Goliath pixel keypoints — convert body+hands to DWPose,
        # face will be rendered directly from all Goliath face points.
        pose_dict = goliath_to_dwpose(kp_data, img_h, img_w)
        raw_goliath = kp_data
    else:
        pose_dict = flat137_to_dwpose(kp_data, img_h, img_w)
        raw_goliath = None

    # Render body + hands via DWPose renderer (no face — we draw it ourselves)
    rendered = draw_pose(
        pose_dict, img_h, img_w,
        show_feet=False,
        show_body=True,
        show_hand=True,
        show_face=(raw_goliath is None),  # only DWPose face if no raw data
        optimized_face=False,
        face_scale=0.75,
    )

    # Overlay rendered skeleton onto canvas (non-zero pixels)
    mask = rendered > 0
    canvas[mask] = rendered[mask]

    # If we have raw Goliath data, draw ALL face keypoints directly
    if raw_goliath is not None:
        _draw_goliath_face(canvas, raw_goliath)

    return canvas


def _draw_goliath_face(canvas, pixel_kp, conf_thr=CONF_THRESHOLD):
    """
    Draw all Goliath face keypoints (index 70+) directly on canvas.
    This renders all ~150 dense face points from Goliath without
    converting to iBUG 68 first — useful for debugging the mapping.
    """
    H, W = canvas.shape[:2]
    radius = max(int(min(H, W) / 400), 1)
    num_kp = pixel_kp.shape[0]

    # Face keypoints start at index 70 in Goliath
    for i in range(70, min(num_kp, 220)):
        x, y, c = pixel_kp[i]
        if c < conf_thr:
            continue
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(canvas, (ix, iy), radius,
                       (255, 255, 255), thickness=-1)
