# NPZ Pose Data Format Specification

## Overview

The `.npz` file is a compressed NumPy archive (`np.savez_compressed`) containing per-frame, per-person 2D and 3D human pose data extracted by the ComfyUI-4DHumans pipeline. All arrays use the key naming conventions documented below.

## Global Metadata

| Key | Type | Description |
|-----|------|-------------|
| `n_persons` | `int32` scalar | Number of persons tracked in the video |
| `n_frames` | `int32` scalar | Number of frames in the video |
| `img_h` | `int32` scalar | Original video frame height (pixels) |
| `img_w` | `int32` scalar | Original video frame width (pixels) |
| `fps` | `float32` scalar | Source video frame rate |

## Optional Filter Parameters

Saved by the Pose Editor when temporal filtering has been applied. These allow restoring the filter state on reload.

| Key | Type | Description |
|-----|------|-------------|
| `filter_velocity_threshold` | `float32` scalar | Velocity outlier threshold multiplier (e.g. 3.0) |
| `filter_smooth_sigma` | `float32` scalar | Gaussian smoothing sigma (0 = no smoothing) |

## Per-Person Visibility

| Key Pattern | Type | Description |
|-------------|------|-------------|
| `person_{i}_visible` | `bool` scalar | Whether person `i` is visible (not deleted by editor). `i` ∈ `[0, n_persons)` |

## Per-Person, Per-Frame 2D Keypoints (Sapiens)

| Key Pattern | Shape | Dtype | Description |
|-------------|-------|-------|-------------|
| `p2d_p{i}_f{j}` | `(133, 3)` | `float32` | Active/filtered COCO-WholeBody 133 keypoints for person `i`, frame `j`. Columns: `[x_pixel, y_pixel, confidence]`. May be absent if person not detected in that frame. |
| `p2d_raw_p{i}_f{j}` | `(133, 3)` | `float32` | Original unfiltered keypoints (before temporal outlier repair). Used for non-destructive restore. Absent if no filtering was applied. |

### COCO-WholeBody 133 Joint Layout

| Index Range | Count | Region |
|-------------|-------|--------|
| 0–16 | 17 | Body |
| 17–22 | 6 | Feet |
| 23–90 | 68 | Face |
| 91–111 | 21 | Left hand |
| 112–132 | 21 | Right hand |

Coordinates are in **pixel space** of the original image (`img_w × img_h`). Confidence is typically `[0, 1]` from the Sapiens model; 0 means not detected.

## Per-Person, Per-Frame 3D Pose (PromptHMR)

| Key Pattern | Shape | Dtype | Description |
|-------------|-------|-------|-------------|
| `p3d_p{i}_body_joints2d_f{j}` | `(25, 2)` | `float32` | OpenPose 25-joint 2D keypoints in original image pixel coordinates. Unscaled from PromptHMR's padded space. |
| `p3d_p{i}_body_joints_f{j}` | `(25, 3)` | `float32` | OpenPose 25-joint 3D keypoints in PromptHMR's modified camera space (metres). |
| `p3d_p{i}_smpl_j3d_f{j}` | `(24, 3)` | `float32` | SMPL 24-joint 3D keypoints from J_regressor, in PromptHMR's modified camera space (metres). |

All 3D keys may be absent if the person's mask was empty in that frame.

### OpenPose 25 Joint Order

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | Nose | 13 | L_Knee |
| 1 | Neck | 14 | L_Ankle |
| 2 | R_Shoulder | 15 | R_Eye |
| 3 | R_Elbow | 16 | L_Eye |
| 4 | R_Wrist | 17 | R_Ear |
| 5 | L_Shoulder | 18 | L_Ear |
| 6 | L_Elbow | 19 | L_BigToe |
| 7 | L_Wrist | 20 | L_SmallToe |
| 8 | MidHip | 21 | L_Heel |
| 9 | R_Hip | 22 | R_BigToe |
| 10 | R_Knee | 23 | R_SmallToe |
| 11 | R_Ankle | 24 | R_Heel |
| 12 | L_Hip | | |

### SMPL 24 Joint Order

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | Pelvis | 12 | Neck |
| 1 | L_Hip | 13 | Head |
| 2 | R_Hip | 14 | Head_top |
| 3 | Spine1 | 15 | L_Collar |
| 4 | L_Knee | 16 | L_Shoulder |
| 5 | R_Knee | 17 | R_Shoulder |
| 6 | Spine2 | 18 | L_Elbow |
| 7 | L_Ankle | 19 | R_Elbow |
| 8 | R_Ankle | 20 | L_Wrist |
| 9 | Spine3 | 21 | R_Wrist |
| 10 | L_Foot | 22 | L_Hand |
| 11 | R_Foot | 23 | R_Hand |

## Per-Frame Camera Parameters

| Key Pattern | Shape | Dtype | Description |
|-------------|-------|-------|-------------|
| `cam_int_f{j}` | `(3, 3)` | `float64` | PromptHMR's modified camera intrinsic matrix for frame `j`. `[0,0]` and `[1,1]` are focal lengths, `[0,2]` and `[1,2]` are principal points. Already includes padding/scaling. |
| `scale_f{j}` | `float64` scalar | Padding scale factor applied by PromptHMR's `prepare_batch`. |
| `offset_f{j}` | `(2,)` | `float64` | Padding offset `[dx, dy]` applied by PromptHMR's `prepare_batch`. |

These are needed to transform `smpl_j3d` from PromptHMR's camera space to any other camera model. The transform from PromptHMR modified pixel space to original image pixel space is:

```
u_original = (u_modified - offset[0]) / scale
v_original = (v_modified - offset[1]) / scale
```

Camera parameters may be absent for frames where no person was detected.

## Loading Example (Python)

```python
import numpy as np

npz = np.load("pose_data_00001.npz", allow_pickle=False)

n_persons = int(npz["n_persons"])
n_frames = int(npz["n_frames"])
fps = float(npz["fps"])
img_w, img_h = int(npz["img_w"]), int(npz["img_h"])

for i in range(n_persons):
    visible = bool(npz[f"person_{i}_visible"])
    print(f"Person {i}: visible={visible}")

    for j in range(n_frames):
        # 2D keypoints
        key_2d = f"p2d_p{i}_f{j}"
        if key_2d in npz:
            kp_2d = npz[key_2d]  # (133, 3)
            nose_x, nose_y, nose_conf = kp_2d[0]

        # 3D SMPL joints
        key_3d = f"p3d_p{i}_smpl_j3d_f{j}"
        if key_3d in npz:
            j3d = npz[key_3d]  # (24, 3) metres
            pelvis_xyz = j3d[0]

        # Camera
        cam_key = f"cam_int_f{j}"
        if cam_key in npz:
            K = npz[cam_key]        # (3, 3)
            scale = float(npz[f"scale_f{j}"])
            offset = npz[f"offset_f{j}"]  # (2,)
```

## Key Naming Summary

All keys follow these patterns (where `i` = person index, `j` = frame index):

```
Global:        n_persons, n_frames, img_h, img_w, fps
Filter:        filter_velocity_threshold, filter_smooth_sigma
Visibility:    person_{i}_visible
Sapiens 2D:    p2d_p{i}_f{j}           → (133, 3)
Sapiens raw:   p2d_raw_p{i}_f{j}       → (133, 3)
PromptHMR 2D:  p3d_p{i}_body_joints2d_f{j}  → (25, 2)
PromptHMR 3D:  p3d_p{i}_body_joints_f{j}    → (25, 3)
SMPL 3D:       p3d_p{i}_smpl_j3d_f{j}       → (24, 3)
Camera:        cam_int_f{j} → (3,3), scale_f{j} → scalar, offset_f{j} → (2,)
```

Missing frames have no corresponding key (check with `key in npz`).
