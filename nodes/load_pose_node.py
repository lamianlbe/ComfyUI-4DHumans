"""
Load Pose Data node.

Loads a .npz file previously saved by the Save Pose Data node and
reconstructs a unified POSES dict with visibility flags preserved.
"""

import glob
import os

import numpy as np

import folder_paths


def _list_pose_files():
    """Return basenames of available .npz pose files in the output dir."""
    output_dir = folder_paths.get_output_directory()
    files = sorted(glob.glob(os.path.join(output_dir, "*.npz")))
    basenames = [os.path.basename(f) for f in files]
    if not basenames:
        basenames = ["(no files found)"]
    return basenames


def npz_to_poses(npz):
    """Reconstruct a POSES dict from a loaded .npz archive."""
    n_persons = int(npz["n_persons"])
    n_frames = int(npz["n_frames"])
    img_h = int(npz["img_h"])
    img_w = int(npz["img_w"])
    fps = float(npz["fps"])

    persons = []
    for i in range(n_persons):
        vis_key = f"person_{i}_visible"
        visible = bool(npz[vis_key]) if vis_key in npz else True

        keypoints = [None] * n_frames
        for j in range(n_frames):
            key = f"p2d_p{i}_f{j}"
            if key in npz:
                keypoints[j] = npz[key]

        person = {
            "visible": visible,
            "body_joints2d": [None] * n_frames,
            "body_joints": [None] * n_frames,
            "smpl_j3d": [None] * n_frames,
            "keypoints": keypoints,
        }
        for j in range(n_frames):
            for field in ("body_joints2d", "body_joints", "smpl_j3d"):
                key = f"p3d_p{i}_{field}_f{j}"
                if key in npz:
                    person[field][j] = npz[key]
        persons.append(person)

    cam_int = [None] * n_frames
    scale = [None] * n_frames
    offset = [None] * n_frames
    for j in range(n_frames):
        ci_key = f"cam_int_f{j}"
        if ci_key in npz:
            cam_int[j] = npz[ci_key]
            scale[j] = float(npz[f"scale_f{j}"])
            offset[j] = npz[f"offset_f{j}"]

    return {
        "n_persons": n_persons,
        "n_frames": n_frames,
        "img_h": img_h,
        "img_w": img_w,
        "fps": fps,
        "persons": persons,
        "cam_int": cam_int,
        "scale": scale,
        "offset": offset,
    }


class LoadPoseDataNode:
    """Load unified POSES from a .npz file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_list_pose_files(),),
            },
        }

    RETURN_TYPES = ("POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    @classmethod
    def IS_CHANGED(cls, file):
        output_dir = folder_paths.get_output_directory()
        path = os.path.join(output_dir, file)
        if os.path.isfile(path):
            return os.path.getmtime(path)
        return float("nan")

    def load(self, file):
        output_dir = folder_paths.get_output_directory()
        path = os.path.join(output_dir, file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pose data file not found: {path}")

        npz = np.load(path, allow_pickle=False)
        poses = npz_to_poses(npz)

        return (poses,)
