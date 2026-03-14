"""
Sapiens PromptHMR to NLF Poses node.

Converts PromptHMR 3D pose (POSE_3D) and Sapiens 2D pose into formats
compatible with SCAIL-Pose's "Render NLF Poses" node.

Outputs:
  - nlf_poses (NLFPRED): list of per-frame tensors (n_persons, 24, 3)
    in SMPL 24-joint format for 3D skeleton rendering.
  - dw_poses (DWPOSES): dict with per-frame body/face/hands in DWPose
    format for 2D overlay rendering.

Internally runs Sapiens 2D inference and interpolates all results to 16fps.

Camera transform: PromptHMR's 3D joints are in the model's padded/scaled
camera space. This node transforms them to the NLF camera space (55 deg FOV)
so that 3D rendering projects correctly onto the original image.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox
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

    For a 3D point P = (X, Y, Z) in PromptHMR's modified camera:
      pixel_modified = K_phmr @ (P / Z)
      pixel_original = (pixel_modified - offset) / scale

    For NLF rendering at pixel_original:
      pixel_original = K_nlf @ (P_nlf / Z_nlf)

    We keep Z_nlf = Z and solve for X_nlf, Y_nlf.

    Parameters
    ----------
    j3d : (24, 3) ndarray
    cam_int : (3, 3) ndarray – PromptHMR modified cam intrinsic
    scale : float
    offset : (2,) ndarray
    K_nlf : (3, 3) ndarray – NLF intrinsic matrix

    Returns
    -------
    j3d_nlf : (24, 3) ndarray
    """
    f_m = cam_int[0, 0]
    cx_m = cam_int[0, 2]
    cy_m = cam_int[1, 2]

    f_nlf = K_nlf[0, 0]
    cx_nlf = K_nlf[0, 2]
    cy_nlf = K_nlf[1, 2]

    X, Y, Z = j3d[:, 0], j3d[:, 1], j3d[:, 2]

    # Project to 2D in modified space, then unscale to original image coords
    u_orig = (f_m * X / Z + cx_m - offset[0]) / scale
    v_orig = (f_m * Y / Z + cy_m - offset[1]) / scale

    # Back-project from original image coords using NLF camera, keeping Z
    X_nlf = (u_orig - cx_nlf) * Z / f_nlf
    Y_nlf = (v_orig - cy_nlf) * Z / f_nlf

    j3d_nlf = np.stack([X_nlf, Y_nlf, Z], axis=-1).astype(np.float32)
    # Convert from metres (SMPL convention) to millimetres (NLF convention)
    j3d_nlf *= METRES_TO_MM
    return j3d_nlf


class SapiensPromptHMRToNLFNode:
    """Convert Sapiens 2D + PromptHMR 3D to SCAIL-Pose NLF/DW format."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sapiens": ("SAPIENS",),
                "images": ("IMAGE",),
                "pose_3d": ("POSE_3D",),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": "Source video FPS.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("NLFPRED", "DWPOSES")
    RETURN_NAMES = ("nlf_poses", "dw_poses")
    FUNCTION = "convert"
    CATEGORY = "4dhumans"

    def convert(self, sapiens, images, pose_3d, fps):
        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape

        n_persons = pose_3d["n_persons"]
        whole_bbox = np.array([0, 0, img_w, img_h], dtype=np.float32)

        # NLF camera intrinsic (55 deg FOV)
        K_nlf = _nlf_intrinsic(img_h, img_w)

        pbar = comfy.utils.ProgressBar(B)

        # ----- Per-person timelines (source fps) -----
        per_person_3d = [[None] * B for _ in range(n_persons)]
        per_person_body = [[None] * B for _ in range(n_persons)]
        per_person_fh = [[None] * B for _ in range(n_persons)]

        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()

            # Run Sapiens to get COCO WholeBody 133 keypoints
            sapiens_result = run_sapiens_on_bbox(img_np, whole_bbox, sapiens)
            sapiens_kp = (
                sapiens_result["pixel_kp"] if sapiens_result is not None
                else None
            )

            # Camera info for this frame
            cam_int_t = pose_3d["cam_int"][t]
            scale_t = pose_3d["scale"][t]
            offset_t = pose_3d["offset"][t]

            for p_idx in range(n_persons):
                person = pose_3d["persons"][p_idx]

                # --- NLF 3D (SMPL 24 joints, transformed to NLF camera) ---
                j3d_smpl = person["smpl_j3d"][t]
                if j3d_smpl is not None and cam_int_t is not None:
                    j3d_nlf = _transform_j3d_to_nlf_camera(
                        j3d_smpl, cam_int_t, scale_t, offset_t, K_nlf
                    )
                    per_person_3d[p_idx][t] = j3d_nlf

                # --- DW body (from PromptHMR 2D) ---
                j2d = person["body_joints2d"][t]
                if j2d is not None:
                    candidate, subset = openpose25_to_dwpose_body(
                        j2d, img_w, img_h
                    )
                    per_person_body[p_idx][t] = (candidate, subset)

                # --- DW face + hands (from Sapiens) ---
                if sapiens_kp is not None:
                    face, rhand, lhand = coco_wb133_to_dwpose_face_hands(
                        sapiens_kp, img_w, img_h
                    )
                    per_person_fh[p_idx][t] = (face, rhand, lhand)

            pbar.update(1)

        # ----- Resample to 16fps -----
        do_resample = fps > 0 and abs(fps - TARGET_FPS) > 0.1

        if do_resample:
            resampled_3d = []
            src_indices = None
            for p in range(n_persons):
                r, idx = resample_keypoints(per_person_3d[p], fps, TARGET_FPS)
                resampled_3d.append(r)
                if src_indices is None:
                    src_indices = idx
            n_out = len(src_indices)

            resampled_body_cand = []
            resampled_body_sub = []
            for p in range(n_persons):
                cand_tl = [x[0] if x is not None else None for x in per_person_body[p]]
                sub_tl = [x[1] if x is not None else None for x in per_person_body[p]]
                r_c, _ = resample_keypoints(cand_tl, fps, TARGET_FPS)
                r_s, _ = resample_keypoints(sub_tl, fps, TARGET_FPS)
                resampled_body_cand.append(r_c)
                resampled_body_sub.append(r_s)

            resampled_face = []
            resampled_rhand = []
            resampled_lhand = []
            for p in range(n_persons):
                face_tl = [x[0] if x is not None else None for x in per_person_fh[p]]
                rhand_tl = [x[1] if x is not None else None for x in per_person_fh[p]]
                lhand_tl = [x[2] if x is not None else None for x in per_person_fh[p]]
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
                [x[0] if x is not None else None for x in per_person_body[p]]
                for p in range(n_persons)
            ]
            resampled_body_sub = [
                [x[1] if x is not None else None for x in per_person_body[p]]
                for p in range(n_persons)
            ]
            resampled_face = [
                [x[0] if x is not None else None for x in per_person_fh[p]]
                for p in range(n_persons)
            ]
            resampled_rhand = [
                [x[1] if x is not None else None for x in per_person_fh[p]]
                for p in range(n_persons)
            ]
            resampled_lhand = [
                [x[2] if x is not None else None for x in per_person_fh[p]]
                for p in range(n_persons)
            ]

        # ----- Build NLF output: list of (n_persons, 24, 3) tensors -----
        nlf_poses = []
        for t in range(n_out):
            frame_joints = []
            for p in range(n_persons):
                j = resampled_3d[p][t]
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
            nlf_poses.append(torch.stack(frame_joints, dim=0))

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

            for p in range(n_persons):
                cand = resampled_body_cand[p][t]
                sub = resampled_body_sub[p][t]
                candidates.append(cand if cand is not None else zero_body.copy())
                subsets.append(sub if sub is not None else zero_subset.copy())

                face = resampled_face[p][t]
                faces.append(face if face is not None else zero_face.copy())

                rhand = resampled_rhand[p][t]
                lhand = resampled_lhand[p][t]
                hands.append(rhand if rhand is not None else zero_hand.copy())
                hands.append(lhand if lhand is not None else zero_hand.copy())

            dw_frames.append({
                "bodies": {
                    "candidate": np.array(candidates, dtype=np.float32),
                    "subset": np.array(subsets, dtype=np.float32),
                },
                "hands": np.array(hands, dtype=np.float32),
                "faces": np.array(faces, dtype=np.float32),
            })

        dw_poses = {"poses": dw_frames, "swap_hands": False}

        return (nlf_poses, dw_poses)
