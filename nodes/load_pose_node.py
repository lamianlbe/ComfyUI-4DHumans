"""
Load Pose Data node.

Loads a .npz file previously saved by the Save Pose Data node and
reconstructs POSE_2D, POSE_3D, and fps.
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


class LoadPoseDataNode:
    """Load POSE_2D + POSE_3D + fps from a .npz file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_list_pose_files(),),
            },
        }

    RETURN_TYPES = ("POSE_2D", "POSE_3D", "FLOAT")
    RETURN_NAMES = ("pose_2d", "pose_3d", "fps")
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    @classmethod
    def IS_CHANGED(cls, file):
        """Force re-execution when file content changes."""
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

        n_persons = int(npz["n_persons"])
        n_frames = int(npz["n_frames"])
        img_h = int(npz["img_h"])
        img_w = int(npz["img_w"])
        fps = float(npz["fps"])

        # ----- Reconstruct POSE_2D -----
        pose_2d_persons = []
        for i in range(n_persons):
            keypoints = [None] * n_frames
            for j in range(n_frames):
                key = f"p2d_p{i}_f{j}"
                if key in npz:
                    keypoints[j] = npz[key]  # (133, 3)
            pose_2d_persons.append({"keypoints": keypoints})

        pose_2d = {
            "n_persons": n_persons,
            "n_frames": n_frames,
            "img_h": img_h,
            "img_w": img_w,
            "persons": pose_2d_persons,
        }

        # ----- Reconstruct POSE_3D -----
        pose_3d_persons = []
        for i in range(n_persons):
            person = {
                "body_joints2d": [None] * n_frames,
                "body_joints": [None] * n_frames,
                "smpl_j3d": [None] * n_frames,
            }
            for j in range(n_frames):
                for field in ("body_joints2d", "body_joints", "smpl_j3d"):
                    key = f"p3d_p{i}_{field}_f{j}"
                    if key in npz:
                        person[field][j] = npz[key]
            pose_3d_persons.append(person)

        cam_int = [None] * n_frames
        scale = [None] * n_frames
        offset = [None] * n_frames
        for j in range(n_frames):
            ci_key = f"cam_int_f{j}"
            if ci_key in npz:
                cam_int[j] = npz[ci_key]
                scale[j] = float(npz[f"scale_f{j}"])
                offset[j] = npz[f"offset_f{j}"]

        pose_3d = {
            "n_persons": n_persons,
            "n_frames": n_frames,
            "img_h": img_h,
            "img_w": img_w,
            "persons": pose_3d_persons,
            "cam_int": cam_int,
            "scale": scale,
            "offset": offset,
        }

        return (pose_2d, pose_3d, fps)
