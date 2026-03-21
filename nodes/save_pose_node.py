"""
Save Pose Data node.

Serialises POSE_2D + POSE_3D + fps to a compressed .npz file in ComfyUI's
output directory.  The file can later be reloaded by the Load Pose Data node.
"""

import os

import numpy as np

import folder_paths


def _next_filename(directory, prefix, ext=".npz"):
    """Return {prefix}_{counter:05d}{ext} with an auto-incrementing counter."""
    counter = 1
    while True:
        name = f"{prefix}_{counter:05d}{ext}"
        if not os.path.exists(os.path.join(directory, name)):
            return name
        counter += 1


class SavePoseDataNode:
    """Save POSE_2D + POSE_3D + fps to a .npz file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_2d": ("POSE_2D",),
                "pose_3d": ("POSE_3D",),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "pose_data"},
                ),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "4dhumans"

    def save(self, pose_2d, pose_3d, fps, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        filename = _next_filename(output_dir, filename_prefix)
        path = os.path.join(output_dir, filename)

        n_persons = pose_3d["n_persons"]
        n_frames = pose_3d["n_frames"]

        data = {
            "n_persons": np.int32(n_persons),
            "n_frames": np.int32(n_frames),
            "img_h": np.int32(pose_3d["img_h"]),
            "img_w": np.int32(pose_3d["img_w"]),
            "fps": np.float32(fps),
        }

        # ----- POSE_2D persons -----
        for i in range(n_persons):
            for j in range(n_frames):
                kp = pose_2d["persons"][i]["keypoints"][j]
                if kp is not None:
                    data[f"p2d_p{i}_f{j}"] = np.asarray(kp, dtype=np.float32)

        # ----- POSE_3D persons -----
        for i in range(n_persons):
            person = pose_3d["persons"][i]
            for j in range(n_frames):
                for key in ("body_joints2d", "body_joints", "smpl_j3d"):
                    val = person[key][j]
                    if val is not None:
                        data[f"p3d_p{i}_{key}_f{j}"] = np.asarray(
                            val, dtype=np.float32
                        )

        # ----- Camera data -----
        for j in range(n_frames):
            cam_int = pose_3d["cam_int"][j]
            if cam_int is not None:
                data[f"cam_int_f{j}"] = np.asarray(cam_int, dtype=np.float64)
                data[f"scale_f{j}"] = np.float64(pose_3d["scale"][j])
                data[f"offset_f{j}"] = np.asarray(
                    pose_3d["offset"][j], dtype=np.float64
                )

        np.savez_compressed(path, **data)

        return {"ui": {"text": [f"Saved: {filename}"]}}
