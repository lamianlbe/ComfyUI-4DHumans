"""
Save Pose Data node.

Serialises unified POSES to a compressed .npz file in ComfyUI's output
directory.  All persons are saved (including invisible ones) with their
visibility flags preserved for non-destructive editing.
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


def poses_to_npz_dict(poses):
    """Convert a POSES dict to a flat dict suitable for np.savez_compressed."""
    n_persons = poses["n_persons"]
    n_frames = poses["n_frames"]

    data = {
        "n_persons": np.int32(n_persons),
        "n_frames": np.int32(n_frames),
        "img_h": np.int32(poses["img_h"]),
        "img_w": np.int32(poses["img_w"]),
        "fps": np.float32(poses["fps"]),
    }

    for i in range(n_persons):
        person = poses["persons"][i]
        data[f"person_{i}_visible"] = np.bool_(person.get("visible", True))

        for j in range(n_frames):
            # Sapiens 2D keypoints
            kp = person["keypoints"][j]
            if kp is not None:
                data[f"p2d_p{i}_f{j}"] = np.asarray(kp, dtype=np.float32)

            # PromptHMR 3D data
            for key in ("body_joints2d", "body_joints", "smpl_j3d"):
                val = person[key][j]
                if val is not None:
                    data[f"p3d_p{i}_{key}_f{j}"] = np.asarray(
                        val, dtype=np.float32
                    )

    # Camera data
    for j in range(n_frames):
        cam_int = poses["cam_int"][j]
        if cam_int is not None:
            data[f"cam_int_f{j}"] = np.asarray(cam_int, dtype=np.float64)
            data[f"scale_f{j}"] = np.float64(poses["scale"][j])
            data[f"offset_f{j}"] = np.asarray(
                poses["offset"][j], dtype=np.float64
            )

    return data


class SavePoseDataNode:
    """Save unified POSES to a .npz file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("POSES",),
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

    def save(self, poses, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        filename = _next_filename(output_dir, filename_prefix)
        path = os.path.join(output_dir, filename)

        data = poses_to_npz_dict(poses)
        np.savez_compressed(path, **data)

        return {"ui": {"text": [f"Saved: {filename}"]}}
