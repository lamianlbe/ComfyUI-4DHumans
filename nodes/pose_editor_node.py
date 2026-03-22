"""
Interactive Pose Editor node.

Displays a video preview of detected poses with person IDs using an
HTML5 <video> element with H.264 MP4.  Users can scrub to any frame,
toggle person visibility (triggers re-render), and download edited NPZ.

Temporal filtering parameters (velocity_threshold, smooth_sigma) are
exposed so users can tune outlier rejection in real time.  The filter
always derives from ``keypoints_raw`` (the original Sapiens output),
never from a previously filtered result, so changes are non-destructive.

The downloaded NPZ stores ``keypoints_raw``, ``smooth_sigma``, and
``velocity_threshold`` alongside the filtered ``keypoints`` so that
a later Load can fully restore the editor state.

Backend:
  - Renders debug frames with person ID labels (only visible persons)
  - Encodes as H.264 MP4 via imageio PyAV plugin
  - Caches POSES + original images for re-rendering on visibility change

Frontend (web/js/pose_editor.js):
  - <video> element with native controls (play/pause/scrub)
  - Per-person toggle buttons (trigger re-render on click)
  - Sliders for velocity_threshold and smooth_sigma (trigger re-filter)
  - Download NPZ button
"""

import io as _io
import os
import copy
import time

import numpy as np
import cv2

import folder_paths
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ._pose_utils import fuse_3d_body_with_sapiens, temporal_filter_keypoints
from .save_pose_node import poses_to_npz_dict

# Global cache: node_id -> {poses, images_np, output_dir, subfolder,
#                            velocity_threshold, smooth_sigma}
_EDITOR_CACHE = {}


def _apply_temporal_filter(poses, velocity_threshold, smooth_sigma):
    """Apply temporal filtering from keypoints_raw → keypoints.

    Always starts from raw data so parameters can be changed freely.
    """
    n_persons = poses["n_persons"]
    total_repaired = 0

    for p_idx in range(n_persons):
        person = poses["persons"][p_idx]
        raw = person.get("keypoints_raw")
        if raw is None:
            # No raw data – nothing to filter
            continue

        if velocity_threshold <= 0:
            # No filtering: just copy raw → keypoints
            person["keypoints"] = [
                kp.copy() if kp is not None else None for kp in raw
            ]
            continue

        filtered, n_rep = temporal_filter_keypoints(
            raw,
            velocity_threshold=velocity_threshold,
            window_size=5,
            smooth_sigma=smooth_sigma,
        )
        person["keypoints"] = filtered
        total_repaired += n_rep

    if total_repaired > 0:
        print(f"[PoseEditor] Temporal filter (vel={velocity_threshold:.1f}, "
              f"sigma={smooth_sigma:.1f}): repaired {total_repaired} frames")


def _render_video(node_id):
    """Re-render the preview video using cached data and current visibility."""
    cache = _EDITOR_CACHE[node_id]
    poses = cache["poses"]
    images_np = cache["images_np"]
    output_dir = cache["output_dir"]

    n_persons = poses["n_persons"]
    B = poses["n_frames"]
    img_h = poses["img_h"]
    img_w = poses["img_w"]
    fps = poses["fps"]

    # Generate unique filename each time to bust browser cache
    ts = int(time.time() * 1000) % 100000
    mp4_filename = f"PoseEditor_{node_id}_{ts}.mp4"
    mp4_path = os.path.join(output_dir, mp4_filename)

    rendered_frames = []
    for t in range(B):
        canvas = images_np[t].copy()

        for p_idx in range(n_persons):
            person = poses["persons"][p_idx]
            if not person.get("visible", True):
                continue

            j2d = person["body_joints2d"][t]
            sapiens_kp = person["keypoints"][t]

            if j2d is not None:
                kp = fuse_3d_body_with_sapiens(j2d, sapiens_kp)
                canvas = render_sapiens_dwpose(canvas, kp, img_h, img_w)
            elif sapiens_kp is not None:
                canvas = render_sapiens_dwpose(canvas, sapiens_kp, img_h, img_w)

            # Draw person ID label near nose
            label_pos = None
            if j2d is not None:
                nx, ny = float(j2d[0, 0]), float(j2d[0, 1])
                if nx > 1 or ny > 1:
                    label_pos = (int(nx), int(ny) - 20)
            elif sapiens_kp is not None:
                nx, ny = float(sapiens_kp[0, 0]), float(sapiens_kp[0, 1])
                if nx > 1 or ny > 1:
                    label_pos = (int(nx), int(ny) - 20)

            if label_pos is not None:
                label = f"P{p_idx}"
                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    canvas_bgr, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA,
                )
                canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

        rendered_frames.append(np.clip(canvas, 0, 255).astype(np.uint8))

    import imageio.v3 as iio
    with iio.imopen(mp4_path, "w", plugin="pyav") as out_file:
        out_file.init_video_stream("libx264", fps=max(fps, 1.0))
        for frame in rendered_frames:
            out_file.write_frame(frame)

    return mp4_filename


def _register_routes():
    """Register API endpoints on first import."""
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    @PromptServer.instance.routes.post("/pose_editor/toggle_visibility")
    async def _toggle_visibility(request):
        """Toggle person visibility and re-render the video."""
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        person_id = int(data.get("person_id", -1))
        visible = bool(data.get("visible", True))

        if node_id not in _EDITOR_CACHE:
            return web.json_response(
                {"error": "Node not found in cache"}, status=404
            )

        cache = _EDITOR_CACHE[node_id]
        poses = cache["poses"]
        if 0 <= person_id < poses["n_persons"]:
            poses["persons"][person_id]["visible"] = visible

        # Re-render video with updated visibility
        mp4_filename = _render_video(node_id)

        return web.json_response({
            "ok": True,
            "video": {
                "filename": mp4_filename,
                "subfolder": cache.get("subfolder", ""),
                "type": "temp",
            },
        })

    @PromptServer.instance.routes.post("/pose_editor/update_filter")
    async def _update_filter(request):
        """Update temporal filter parameters, re-filter, and re-render."""
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        velocity_threshold = float(data.get("velocity_threshold", 3.0))
        smooth_sigma = float(data.get("smooth_sigma", 0.0))

        if node_id not in _EDITOR_CACHE:
            return web.json_response(
                {"error": "Node not found in cache"}, status=404
            )

        cache = _EDITOR_CACHE[node_id]
        cache["velocity_threshold"] = velocity_threshold
        cache["smooth_sigma"] = smooth_sigma

        # Re-apply filter from raw keypoints
        _apply_temporal_filter(
            cache["poses"], velocity_threshold, smooth_sigma
        )

        # Re-render video
        mp4_filename = _render_video(node_id)

        return web.json_response({
            "ok": True,
            "video": {
                "filename": mp4_filename,
                "subfolder": cache.get("subfolder", ""),
                "type": "temp",
            },
        })

    @PromptServer.instance.routes.get("/pose_editor/download_npz")
    async def _download_npz(request):
        node_id = str(request.query.get("node_id", ""))
        if node_id not in _EDITOR_CACHE:
            return web.json_response(
                {"error": "Node not found in cache"}, status=404
            )

        cache = _EDITOR_CACHE[node_id]
        poses = cache["poses"]

        # Store filter params in poses for serialisation
        poses["_filter_velocity_threshold"] = cache.get(
            "velocity_threshold", 0.0)
        poses["_filter_smooth_sigma"] = cache.get("smooth_sigma", 0.0)

        data = poses_to_npz_dict(poses)

        buf = _io.BytesIO()
        np.savez_compressed(buf, **data)
        buf.seek(0)

        return web.Response(
            body=buf.read(),
            content_type="application/octet-stream",
            headers={
                "Content-Disposition":
                    "attachment; filename=pose_data_edited.npz"
            },
        )


_register_routes()


class PoseEditorNode:
    """Interactive pose editor with H.264 video preview and person toggles."""

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("POSES",),
                "images": ("IMAGE",),
                "velocity_threshold": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "Outlier detection sensitivity. Higher = less "
                            "aggressive filtering. 0 = disabled."
                        ),
                    },
                ),
                "smooth_sigma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": (
                            "Gaussian temporal smoothing sigma. 0 = no "
                            "smoothing, only outlier repair."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "edit"
    CATEGORY = "4dhumans"

    def edit(self, poses, images, velocity_threshold, smooth_sigma,
             unique_id=None):
        node_id = str(unique_id) if unique_id is not None else "unknown"

        poses_edit = copy.deepcopy(poses)
        images_np = (images * 255).byte().cpu().numpy()  # (B, H, W, 3)

        n_persons = poses_edit["n_persons"]
        fps = poses_edit["fps"]

        # Restore filter params from loaded NPZ if present, but prefer
        # the node's widget values (which the user may have changed).
        # The widget values are the authoritative source.
        # (NPZ-stored values are only used as defaults in the frontend.)

        # Apply temporal filter from raw → keypoints
        _apply_temporal_filter(poses_edit, velocity_threshold, smooth_sigma)

        # Determine output folder
        full_output_folder, _, _, subfolder, _ = (
            folder_paths.get_save_image_path(
                "PoseEditor", self.output_dir
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        # Cache everything needed for re-renders
        _EDITOR_CACHE[node_id] = {
            "poses": poses_edit,
            "images_np": images_np,
            "output_dir": full_output_folder,
            "subfolder": subfolder,
            "velocity_threshold": velocity_threshold,
            "smooth_sigma": smooth_sigma,
        }

        # Initial render
        mp4_filename = _render_video(node_id)

        person_visibility = [
            poses_edit["persons"][p].get("visible", True)
            for p in range(n_persons)
        ]

        ui_data = {
            "video": [{
                "filename": mp4_filename,
                "subfolder": subfolder,
                "type": self.type,
            }],
            "n_persons": [n_persons],
            "person_visibility": [person_visibility],
            "node_id": [node_id],
            "fps": [fps],
            "velocity_threshold": [velocity_threshold],
            "smooth_sigma": [smooth_sigma],
        }

        return {"ui": ui_data}
