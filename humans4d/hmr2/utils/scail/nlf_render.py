"""
Adapted from ComfyUI-SCAIL-Pose/NLFPoseExtract/nlf_render.py
Only the rendering functions needed for SCAIL-style output.
Imports fixed for our package layout.
"""
import numpy as np
import torch
import copy
import logging

try:
    from .taichi_cylinder import render_whole as render_whole_taichi
except Exception:
    render_whole_taichi = None

from .render_torch import render_whole as render_whole_torch
from .draw_pose_utils import draw_pose_to_canvas_np


def process_data_to_COCO_format(joints):
    """Convert NLF/SMPL 24-joint array to COCO 18-joint format."""
    if joints.ndim != 2:
        raise ValueError(f"Expected shape (24,2) or (24,3), got {joints.shape}")
    dim = joints.shape[1]
    mapping = {
        15: 0,   # head
        12: 1,   # neck
        17: 2,   # left shoulder
        16: 5,   # right shoulder
        19: 3,   # left elbow
        18: 6,   # right elbow
        21: 4,   # left hand
        20: 7,   # right hand
        2: 8,    # left pelvis
        1: 11,   # right pelvis
        5: 9,    # left knee
        4: 12,   # right knee
        8: 10,   # left feet
        7: 13,   # right feet
    }
    new_joints = np.zeros((18, dim), dtype=joints.dtype)
    for src, dst in mapping.items():
        new_joints[dst] = joints[src]
    return new_joints


def intrinsic_matrix_from_field_of_view(imshape, fov_degrees=55):
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    return np.array([
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1],
    ])


def get_single_pose_cylinder_specs(args, include_missing=False):
    """Build cylinder specs for a single frame's poses."""
    idx, pose, focal, princpt, height, width, colors, limb_seq, draw_seq = args
    cylinder_specs = []
    for joints3d in pose:
        if joints3d is None:
            if include_missing:
                for line_idx in draw_seq:
                    cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
            continue
        if isinstance(joints3d, torch.Tensor):
            if torch.sum(torch.abs(joints3d)) < 0.01:
                if include_missing:
                    for line_idx in draw_seq:
                        cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
                continue
            joints3d = joints3d.cpu().numpy()
        elif isinstance(joints3d, np.ndarray):
            if np.sum(np.abs(joints3d)) < 0.01:
                if include_missing:
                    for line_idx in draw_seq:
                        cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
                continue
        else:
            if include_missing:
                for line_idx in draw_seq:
                    cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
            continue

        joints3d = process_data_to_COCO_format(joints3d)
        for line_idx in draw_seq:
            line = limb_seq[line_idx]
            start, end = line[0], line[1]
            if np.sum(joints3d[start]) == 0 or np.sum(joints3d[end]) == 0:
                if include_missing:
                    cylinder_specs.append((np.zeros(3), np.zeros(3), colors[line_idx]))
                continue
            else:
                cylinder_specs.append((joints3d[start], joints3d[end], colors[line_idx]))
    return cylinder_specs


def render_nlf_as_images(smpl_poses, dw_poses, height, width, video_length,
                         intrinsic_matrix=None, draw_2d=True, draw_face=True,
                         draw_hands=True, render_backend="taichi"):
    """Render SCAIL-style pose images from 3D body + 2D hand/face data."""

    base_colors_255_dict = {
        "Red": [255, 0, 0],
        "Orange": [255, 85, 0],
        "Golden Orange": [255, 170, 0],
        "Yellow": [255, 240, 0],
        "Yellow-Green": [180, 255, 0],
        "Bright Green": [0, 255, 0],
        "Light Green-Blue": [0, 255, 85],
        "Aqua": [0, 255, 170],
        "Cyan": [0, 255, 255],
        "Sky Blue": [0, 170, 255],
        "Medium Blue": [0, 85, 255],
        "Pure Blue": [0, 0, 255],
        "Purple-Blue": [85, 0, 255],
        "Medium Purple": [170, 0, 255],
        "Grey": [150, 150, 150],
        "Pink-Magenta": [255, 0, 170],
        "Dark Pink": [255, 0, 85],
        "Violet": [100, 0, 255],
        "Dark Violet": [50, 0, 255],
    }

    ordered_colors_255 = [
        base_colors_255_dict["Red"],
        base_colors_255_dict["Cyan"],
        base_colors_255_dict["Orange"],
        base_colors_255_dict["Golden Orange"],
        base_colors_255_dict["Sky Blue"],
        base_colors_255_dict["Medium Blue"],
        base_colors_255_dict["Yellow-Green"],
        base_colors_255_dict["Bright Green"],
        base_colors_255_dict["Light Green-Blue"],
        base_colors_255_dict["Pure Blue"],
        base_colors_255_dict["Purple-Blue"],
        base_colors_255_dict["Medium Purple"],
        base_colors_255_dict["Grey"],
        base_colors_255_dict["Pink-Magenta"],
        base_colors_255_dict["Dark Violet"],
        base_colors_255_dict["Pink-Magenta"],
        base_colors_255_dict["Dark Violet"],
    ]

    limb_seq = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    ]

    draw_seq = [
        0, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    ]

    colors = [[c / 300 + 0.15 for c in color_rgb] + [0.8]
              for color_rgb in ordered_colors_255]

    if dw_poses is not None:
        aligned_poses = copy.deepcopy(dw_poses)

    if intrinsic_matrix is None:
        intrinsic_matrix = intrinsic_matrix_from_field_of_view((height, width))
    focal_x = intrinsic_matrix[0, 0]
    focal_y = intrinsic_matrix[1, 1]
    princpt = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])

    cylinder_specs_list = []
    for i in range(video_length):
        cylinder_specs = get_single_pose_cylinder_specs(
            (i, smpl_poses[i], None, None, None, None, colors, limb_seq, draw_seq))
        cylinder_specs_list.append(cylinder_specs)

    if render_backend == "taichi" and render_whole_taichi is not None:
        try:
            frames_np_rgba = render_whole_taichi(
                cylinder_specs_list, H=height, W=width,
                fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
        except Exception:
            logging.warning("Taichi rendering failed. Falling back to torch rendering.")
            frames_np_rgba = render_whole_torch(
                cylinder_specs_list, H=height, W=width,
                fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])
    else:
        frames_np_rgba = render_whole_torch(
            cylinder_specs_list, H=height, W=width,
            fx=focal_x, fy=focal_y, cx=princpt[0], cy=princpt[1])

    if dw_poses is not None and draw_2d:
        canvas_2d = draw_pose_to_canvas_np(
            aligned_poses, pool=None, H=height, W=width, reshape_scale=0,
            show_feet_flag=False, show_body_flag=False, show_cheek_flag=True,
            dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
        for i in range(len(frames_np_rgba)):
            frame_img = frames_np_rgba[i]
            canvas_img = canvas_2d[i]
            mask = canvas_img != 0
            frame_img[:, :, :3][mask] = canvas_img[mask]
            frames_np_rgba[i] = frame_img

    return frames_np_rgba
