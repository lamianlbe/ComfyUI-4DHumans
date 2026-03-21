"""
Interactive Pose Editor node.

Displays a video preview of detected poses with person IDs.  Users can
toggle person visibility and download the edited POSES as an NPZ file
directly from the browser.

Backend responsibilities:
  - Render debug frames with person ID labels and save to temp dir
  - Cache POSES data keyed by node unique_id
  - Provide API endpoints for visibility toggling and NPZ download

Frontend (web/js/pose_editor.js):
  - Video playback with scrubber
  - Per-person toggle buttons
  - Download NPZ button
"""

import io
import os
import json
import copy

import numpy as np
import cv2
import torch

import folder_paths
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ._pose_utils import (
    fuse_3d_body_with_sapiens,
)
from .save_pose_node import poses_to_npz_dict

# Global cache: node_id -> poses dict (mutable, edited in-place)
_EDITOR_CACHE = {}


def _register_routes():
    """Register API endpoints on first import."""
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    @PromptServer.instance.routes.post("/pose_editor/update_visibility")
    async def _update_visibility(request):
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        person_id = int(data.get("person_id", -1))
        visible = bool(data.get("visible", True))

        if node_id not in _EDITOR_CACHE:
            return web.json_response(
                {"error": "Node not found in cache"}, status=404
            )

        poses = _EDITOR_CACHE[node_id]
        if 0 <= person_id < poses["n_persons"]:
            poses["persons"][person_id]["visible"] = visible

        return web.json_response({"ok": True})

    @PromptServer.instance.routes.get("/pose_editor/download_npz")
    async def _download_npz(request):
        node_id = str(request.query.get("node_id", ""))
        if node_id not in _EDITOR_CACHE:
            return web.json_response(
                {"error": "Node not found in cache"}, status=404
            )

        poses = _EDITOR_CACHE[node_id]
        data = poses_to_npz_dict(poses)

        buf = io.BytesIO()
        np.savez_compressed(buf, **data)
        buf.seek(0)

        return web.Response(
            body=buf.read(),
            content_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=pose_data_edited.npz"
            },
        )


_register_routes()


class PoseEditorNode:
    """Interactive pose editor with video preview and person toggles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("POSES",),
                "images": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "edit"
    CATEGORY = "4dhumans"

    def edit(self, poses, images, unique_id=None):
        node_id = str(unique_id) if unique_id is not None else "unknown"

        # Deep copy so edits don't mutate upstream data
        poses_edit = copy.deepcopy(poses)

        n_persons = poses_edit["n_persons"]
        B = poses_edit["n_frames"]
        img_h = poses_edit["img_h"]
        img_w = poses_edit["img_w"]
        fps = poses_edit["fps"]

        images_np = (images * 255).byte().cpu().numpy()  # (B, H, W, 3)

        pbar = comfy.utils.ProgressBar(B)

        temp_dir = folder_paths.get_temp_directory()
        subfolder = f"pose_editor_{node_id}"
        frame_dir = os.path.join(temp_dir, subfolder)
        os.makedirs(frame_dir, exist_ok=True)

        frame_files = []

        for t in range(B):
            canvas = images_np[t].copy()  # (H, W, 3) RGB

            # Render all persons (even invisible) so user can see who to toggle
            for p_idx in range(n_persons):
                person = poses_edit["persons"][p_idx]
                j2d = person["body_joints2d"][t]
                sapiens_kp = person["keypoints"][t]

                if j2d is not None:
                    kp = fuse_3d_body_with_sapiens(j2d, sapiens_kp)
                    canvas = render_sapiens_dwpose(canvas, kp, img_h, img_w)
                elif sapiens_kp is not None:
                    canvas = render_sapiens_dwpose(
                        canvas, sapiens_kp, img_h, img_w
                    )

                # Draw person ID label near nose (COCO-WB index 0)
                label_pos = None
                if j2d is not None:
                    # OpenPose25 index 0 = nose
                    nx, ny = float(j2d[0, 0]), float(j2d[0, 1])
                    if nx > 1 or ny > 1:
                        label_pos = (int(nx), int(ny) - 20)
                elif sapiens_kp is not None:
                    nx, ny = float(sapiens_kp[0, 0]), float(sapiens_kp[0, 1])
                    if nx > 1 or ny > 1:
                        label_pos = (int(nx), int(ny) - 20)

                if label_pos is not None:
                    vis_tag = "" if person.get("visible", True) else " [hidden]"
                    label = f"P{p_idx}{vis_tag}"
                    # BGR for cv2
                    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        canvas_bgr, label, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA,
                    )
                    canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

            # Save frame as JPEG
            fname = f"frame_{t:05d}.jpg"
            fpath = os.path.join(frame_dir, fname)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imencode(".jpg", canvas_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(fpath)

            frame_files.append({
                "filename": f"{subfolder}/{fname}",
                "type": "temp",
                "subfolder": "",
            })

            pbar.update(1)

        # Cache for API access
        _EDITOR_CACHE[node_id] = poses_edit

        # Build visibility list
        person_visibility = [
            poses_edit["persons"][p].get("visible", True)
            for p in range(n_persons)
        ]

        ui_data = {
            "frames": frame_files,
            "fps": [fps],
            "n_persons": [n_persons],
            "person_visibility": [person_visibility],
            "node_id": [node_id],
        }

        return {"ui": ui_data}
