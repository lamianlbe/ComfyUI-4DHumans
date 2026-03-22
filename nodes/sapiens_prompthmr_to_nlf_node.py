"""
Sapiens PromptHMR to NLF Poses node.

Converts unified POSES data into formats compatible with SCAIL-Pose's
"Render NLF Poses" node.  Only processes visible persons.

Outputs:
  - nlf_poses (NLFPRED): list of per-frame tensors (n_visible, 24, 3)
    in SMPL 24-joint format for 3D skeleton rendering.
  - dw_poses (DWPOSES): dict with per-frame body/face/hands in DWPose
    format for 2D overlay rendering.

Interpolates all results to 16fps for WAN SCAIL compatibility.

Camera transform: PromptHMR's 3D joints are in the model's padded/scaled
camera space. This node transforms them to the NLF camera space (55 deg FOV)
so that 3D rendering projects correctly onto the original image.
"""

import numpy as np
import torch
import comfy.utils

from ._pose_utils import (
    openpose25_to_dwpose_body,
    coco_wb133_to_dwpose_face_hands,
    resample_keypoints,
)

TARGET_FPS = 16.0
NLF_FOV_DEGREES = 55.0
# PromptHMR outputs SMPL joints in metres; NLF's cylinder renderer expects
# millimetres (default cylinder radius=21.5, zfar up to 25 000).
METRES_TO_MM = 1000.0


def _nlf_intrinsic(img_h, img_w):
    """Compute NLF's default intrinsic matrix (55 deg FOV)."""
    larger = max(img_h, img_w)
    focal = larger / (2.0 * np.tan(NLF_FOV_DEGREES * np.pi / 360.0))
    return np.array([
        [focal, 0, img_w / 2.0],
        [0, focal, img_h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)


def _transform_j3d_to_nlf_camera(j3d, cam_int, scale, offset, K_nlf):
    """
    Transform 3D joints from PromptHMR's padded/scaled camera space
    to NLF's camera space.

    We solve for X_nlf, Y_nlf keeping Z, then apply a uniform scale
    k = f_nlf / f_modified to bring depth into NLF's expected range.
    """
    f_m = cam_int[0, 0]
    cx_m = cam_int[0, 2]
    cy_m = cam_int[1, 2]

    f_nlf = K_nlf[0, 0]
    cx_nlf = K_nlf[0, 2]
    cy_nlf = K_nlf[1, 2]

    X, Y, Z = j3d[:, 0], j3d[:, 1], j3d[:, 2]

    u_orig = (f_m * X / Z + cx_m - offset[0]) / scale
    v_orig = (f_m * Y / Z + cy_m - offset[1]) / scale

    X_nlf = (u_orig - cx_nlf) * Z / f_nlf
    Y_nlf = (v_orig - cy_nlf) * Z / f_nlf

    j3d_nlf = np.stack([X_nlf, Y_nlf, Z], axis=-1).astype(np.float32)

    depth_scale = float(f_nlf / f_m)
    j3d_nlf *= depth_scale

    j3d_nlf *= METRES_TO_MM
    return j3d_nlf


class SapiensPromptHMRToNLFNode:
    """Convert unified POSES to SCAIL-Pose NLF/DW format."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses": ("POSES",),
            },
        }

    RETURN_TYPES = ("NLFPRED", "DWPOSES", "INT", "INT")
    RETURN_NAMES = ("nlf_poses", "dw_poses", "width", "height")
    FUNCTION = "convert"
    CATEGORY = "4dhumans"

    def convert(self, poses):
        n_persons = poses["n_persons"]
        B = poses["n_frames"]
        img_h = poses["img_h"]
        img_w = poses["img_w"]
        fps = poses["fps"]

        # Only process visible persons
        visible_indices = [
            p for p in range(n_persons)
            if poses["persons"][p].get("visible", True)
        ]
        n_visible = len(visible_indices)

        K_nlf = _nlf_intrinsic(img_h, img_w)

        pbar = comfy.utils.ProgressBar(B)

        # ----- Per-person timelines (source fps) -----
        per_person_3d = [[None] * B for _ in range(n_visible)]
        per_person_body = [[None] * B for _ in range(n_visible)]
        per_person_fh = [[None] * B for _ in range(n_visible)]

        for t in range(B):
            cam_int_t = poses["cam_int"][t]
            scale_t = poses["scale"][t]
            offset_t = poses["offset"][t]

            for vi, p_idx in enumerate(visible_indices):
                person = poses["persons"][p_idx]

                # --- NLF 3D ---
                j3d_smpl = person["smpl_j3d"][t]
                if j3d_smpl is not None and cam_int_t is not None:
                    j3d_nlf = _transform_j3d_to_nlf_camera(
                        j3d_smpl, cam_int_t, scale_t, offset_t, K_nlf
                    )
                    per_person_3d[vi][t] = j3d_nlf

                # --- DW body ---
                j2d = person["body_joints2d"][t]
                if j2d is not None:
                    candidate, subset = openpose25_to_dwpose_body(
                        j2d, img_w, img_h
                    )
                    per_person_body[vi][t] = (candidate, subset)

                # --- DW face + hands ---
                sapiens_kp = person["keypoints"][t]
                if sapiens_kp is not None:
                    face, rhand, lhand = coco_wb133_to_dwpose_face_hands(
                        sapiens_kp, img_w, img_h
                    )
                    per_person_fh[vi][t] = (face, rhand, lhand)

            pbar.update(1)

        # ----- Resample to 16fps -----
        do_resample = fps > 0 and abs(fps - TARGET_FPS) > 0.1

        if do_resample:
            resampled_3d = []
            src_indices = None
            for vi in range(n_visible):
                r, idx = resample_keypoints(per_person_3d[vi], fps, TARGET_FPS)
                resampled_3d.append(r)
                if src_indices is None:
                    src_indices = idx
            n_out = len(src_indices)

            resampled_body_cand = []
            resampled_body_sub = []
            for vi in range(n_visible):
                cand_tl = [x[0] if x is not None else None for x in per_person_body[vi]]
                sub_tl = [x[1] if x is not None else None for x in per_person_body[vi]]
                r_c, _ = resample_keypoints(cand_tl, fps, TARGET_FPS)
                r_s, _ = resample_keypoints(sub_tl, fps, TARGET_FPS)
                resampled_body_cand.append(r_c)
                resampled_body_sub.append(r_s)

            resampled_face = []
            resampled_rhand = []
            resampled_lhand = []
            for vi in range(n_visible):
                face_tl = [x[0] if x is not None else None for x in per_person_fh[vi]]
                rhand_tl = [x[1] if x is not None else None for x in per_person_fh[vi]]
                lhand_tl = [x[2] if x is not None else None for x in per_person_fh[vi]]
                r_f, _ = resample_keypoints(face_tl, fps, TARGET_FPS)
                r_r, _ = resample_keypoints(rhand_tl, fps, TARGET_FPS)
                r_l, _ = resample_keypoints(lhand_tl, fps, TARGET_FPS)
                resampled_face.append(r_f)
                resampled_rhand.append(r_r)
                resampled_lhand.append(r_l)
        else:
            n_out = B
            resampled_3d = per_person_3d
            resampled_body_cand = [
                [x[0] if x is not None else None for x in per_person_body[vi]]
                for vi in range(n_visible)
            ]
            resampled_body_sub = [
                [x[1] if x is not None else None for x in per_person_body[vi]]
                for vi in range(n_visible)
            ]
            resampled_face = [
                [x[0] if x is not None else None for x in per_person_fh[vi]]
                for vi in range(n_visible)
            ]
            resampled_rhand = [
                [x[1] if x is not None else None for x in per_person_fh[vi]]
                for vi in range(n_visible)
            ]
            resampled_lhand = [
                [x[2] if x is not None else None for x in per_person_fh[vi]]
                for vi in range(n_visible)
            ]

        # ----- Build NLF output -----
        nlf_poses = []
        for t in range(n_out):
            frame_joints = []
            for vi in range(n_visible):
                j = resampled_3d[vi][t]
                if j is not None:
                    frame_joints.append(
                        torch.from_numpy(j).float()
                        if isinstance(j, np.ndarray)
                        else j.clone().float()
                    )
                else:
                    frame_joints.append(
                        torch.zeros((24, 3), dtype=torch.float32)
                    )
            if frame_joints:
                nlf_poses.append(torch.stack(frame_joints, dim=0))
            else:
                nlf_poses.append(torch.zeros((1, 24, 3), dtype=torch.float32))

        # ----- Build DW output -----
        dw_frames = []
        zero_body = np.zeros((18, 2), dtype=np.float32)
        zero_subset = np.full(18, -1.0, dtype=np.float32)
        zero_face = np.zeros((68, 2), dtype=np.float32)
        zero_hand = np.zeros((21, 2), dtype=np.float32)

        for t in range(n_out):
            candidates = []
            subsets = []
            faces = []
            hands = []

            for vi in range(n_visible):
                cand = resampled_body_cand[vi][t]
                sub = resampled_body_sub[vi][t]
                candidates.append(cand if cand is not None else zero_body.copy())
                subsets.append(sub if sub is not None else zero_subset.copy())

                face = resampled_face[vi][t]
                faces.append(face if face is not None else zero_face.copy())

                rhand = resampled_rhand[vi][t]
                lhand = resampled_lhand[vi][t]
                hands.append(rhand if rhand is not None else zero_hand.copy())
                hands.append(lhand if lhand is not None else zero_hand.copy())

            dw_frames.append({
                "bodies": {
                    "candidate": np.array(candidates, dtype=np.float32) if candidates else np.zeros((1, 18, 2), dtype=np.float32),
                    "subset": np.array(subsets, dtype=np.float32) if subsets else np.full((1, 18), -1.0, dtype=np.float32),
                },
                "hands": np.array(hands, dtype=np.float32) if hands else np.zeros((2, 21, 2), dtype=np.float32),
                "faces": np.array(faces, dtype=np.float32) if faces else np.zeros((1, 68, 2), dtype=np.float32),
            })

        dw_poses = {"poses": dw_frames, "swap_hands": False}

        return (nlf_poses, dw_poses, img_w, img_h)
