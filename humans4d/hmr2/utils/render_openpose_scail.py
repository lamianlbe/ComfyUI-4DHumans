"""
Render SCAIL-style pose images from SMPLest-X 137-joint keypoints.

Visual style matches WAN-SCAIL ControlNet expectations:
  - Body limbs: 3D cylinder rendering via ray marching (SDF + Blinn-Phong)
    with SCAIL color scheme (warm=right side, cool=left side)
  - Hands: DWPose-style HSV rainbow thin lines + red dots
  - Face: white dots
"""
import math
import cv2
import numpy as np
import torch

# ── SMPLest-X 25-body → SCAIL limb mapping ──────────────────────────────────
# Our body joints: 0=Pelvis 1=L.Hip 2=R.Hip 3=L.Knee 4=R.Knee 5=L.Ankle
# 6=R.Ankle 7=Neck 8=L.Shoulder 9=R.Shoulder 10=L.Elbow 11=R.Elbow
# 12=L.Wrist 13=R.Wrist 14-16=L.Foot 17-19=R.Foot 20=L.Ear 21=R.Ear
# 22=L.Eye 23=R.Eye 24=Nose

SCAIL_LIMBS = [
    (7, 9),    # 0  Neck → R.Shoulder
    (7, 8),    # 1  Neck → L.Shoulder
    (9, 11),   # 2  R.Shoulder → R.Elbow
    (11, 13),  # 3  R.Elbow → R.Wrist
    (8, 10),   # 4  L.Shoulder → L.Elbow
    (10, 12),  # 5  L.Elbow → L.Wrist
    (7, 2),    # 6  Neck → R.Hip
    (2, 4),    # 7  R.Hip → R.Knee
    (4, 6),    # 8  R.Knee → R.Ankle
    (7, 1),    # 9  Neck → L.Hip
    (1, 3),    # 10 L.Hip → L.Knee
    (3, 5),    # 11 L.Knee → L.Ankle
    (7, 24),   # 12 Neck → Nose
    (24, 23),  # 13 Nose → R.Eye
    (23, 21),  # 14 R.Eye → R.Ear
    (24, 22),  # 15 Nose → L.Eye
    (22, 20),  # 16 L.Eye → L.Ear
]

# Draw order: expanding outward from the proximal end
SCAIL_DRAW_SEQ = [
    0, 2, 3,     # Neck → R.Shoulder → R.Elbow → R.Wrist
    1, 4, 5,     # Neck → L.Shoulder → L.Elbow → L.Wrist
    6, 7, 8,     # Neck → R.Hip → R.Knee → R.Ankle
    9, 10, 11,   # Neck → L.Hip → L.Knee → L.Ankle
    12,          # Neck → Nose
    13, 14,      # Nose → R.Eye → R.Ear
    15, 16,      # Nose → L.Eye → L.Ear
]

# SCAIL color scheme (RGB 0-255): warm=right, cool=left
SCAIL_LIMB_COLORS_255 = [
    (255, 0, 0),       # 0  Red
    (0, 255, 255),     # 1  Cyan
    (255, 85, 0),      # 2  Orange
    (255, 170, 0),     # 3  Golden Orange
    (0, 170, 255),     # 4  Sky Blue
    (0, 85, 255),      # 5  Medium Blue
    (180, 255, 0),     # 6  Yellow-Green
    (0, 255, 0),       # 7  Bright Green
    (0, 255, 85),      # 8  Light Green-Blue
    (0, 0, 255),       # 9  Pure Blue
    (85, 0, 255),      # 10 Purple-Blue
    (170, 0, 255),     # 11 Medium Purple
    (150, 150, 150),   # 12 Grey
    (255, 0, 170),     # 13 Pink-Magenta
    (50, 0, 255),      # 14 Dark Violet
    (255, 0, 170),     # 15 Pink-Magenta
    (50, 0, 255),      # 16 Dark Violet
]

# Normalized RGBA colors for 3D renderer (same formula as SCAIL reference)
SCAIL_LIMB_COLORS_NORM = [
    [c / 300 + 0.15 for c in rgb] + [0.8]
    for rgb in SCAIL_LIMB_COLORS_255
]

# Hand edges (wrist → 4 joints per finger × 5 fingers = 20 edges)
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def _hsv_to_rgb(h, s, v):
    """Convert a single HSV value to RGB tuple (0-255)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── 3D Cylinder Renderer (PyTorch ray marching) ─────────────────────────────

def _render_cylinders_torch(cylinder_specs, H, W, fx, fy, cx, cy,
                            radius=21.5, device=None):
    """
    Render 3D cylinders via SDF ray marching with Blinn-Phong shading.

    Args:
        cylinder_specs: list of (start_3d, end_3d, color_rgba) tuples
        H, W: output image dimensions
        fx, fy, cx, cy: camera intrinsics
        radius: cylinder radius in 3D units
        device: torch device

    Returns:
        (H, W, 4) uint8 RGBA image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(cylinder_specs) == 0:
        return np.zeros((H, W, 4), dtype=np.uint8)

    starts_np = np.array([s for s, e, c in cylinder_specs], dtype=np.float32)
    ends_np = np.array([e for s, e, c in cylinder_specs], dtype=np.float32)
    colors_np = np.array([c for s, e, c in cylinder_specs], dtype=np.float32)

    # Skip zero-length cylinders
    lengths = np.linalg.norm(ends_np - starts_np, axis=1)
    valid = lengths > 0.01
    if not valid.any():
        return np.zeros((H, W, 4), dtype=np.uint8)
    starts_np = starts_np[valid]
    ends_np = ends_np[valid]
    colors_np = colors_np[valid]

    curr_starts = torch.from_numpy(starts_np).to(device).float()
    curr_ends = torch.from_numpy(ends_np).to(device).float()
    curr_colors = torch.from_numpy(colors_np).to(device).float()

    z_min_val = min(starts_np[:, 2].min(), ends_np[:, 2].min())
    z_max_val = max(starts_np[:, 2].max(), ends_np[:, 2].max())

    znear = 0.1
    zfar = max(min(z_max_val, 25000), 10000)

    # Ray grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device).float(),
        torch.arange(W, device=device).float(),
        indexing="ij",
    )
    u = (x_coords - cx) / fx
    v = (y_coords - cy) / fy
    z = torch.ones_like(u)
    ray_dirs = torch.stack([u, v, z], dim=-1)
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

    flat_ray_dirs = ray_dirs.view(-1, 3)
    flat_ray_origins = torch.zeros((H * W, 3), device=device)

    light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Cylinder precomputation
    ba = curr_ends - curr_starts
    ba_len = torch.sqrt((ba * ba).sum(dim=1))
    ba_norm = ba / ba_len.unsqueeze(1)

    MAX_STEPS = 100
    EPSILON = 1e-3
    N_pix = H * W

    flat_t = torch.ones(N_pix, device=device) * znear
    flat_active = torch.ones(N_pix, dtype=torch.bool, device=device)
    flat_hit = torch.zeros(N_pix, dtype=torch.bool, device=device)
    flat_hit_color = torch.zeros((N_pix, 4), device=device)
    flat_hit_pos = torch.zeros((N_pix, 3), device=device)

    depth_near = max(z_min_val, 0.1)
    depth_far = min(z_max_val + 6000, 20000)

    for step in range(MAX_STEPS):
        if not flat_active.any():
            break

        p = flat_ray_origins + flat_ray_dirs * flat_t.unsqueeze(1)
        active_indices = torch.nonzero(flat_active).squeeze()
        if active_indices.numel() == 0:
            break

        p_active = p[active_indices]
        pa = p_active.unsqueeze(1) - curr_starts.unsqueeze(0)
        proj = (pa * ba_norm.unsqueeze(0)).sum(dim=-1)
        proj_clamped = proj.clamp(min=0.0).min(ba_len.unsqueeze(0))
        closest_on_axis = (curr_starts.unsqueeze(0)
                           + proj_clamped.unsqueeze(-1) * ba_norm.unsqueeze(0))
        dist_vec = p_active.unsqueeze(1) - closest_on_axis
        dist_euc = torch.norm(dist_vec, dim=-1)
        sdf = dist_euc - radius
        min_sdf, min_idx = sdf.min(dim=1)

        current_t_vals = flat_t[active_indices]
        hit_cond = min_sdf < EPSILON
        miss_cond = current_t_vals > zfar
        new_hits = hit_cond & (~miss_cond)

        hit_global_idx = active_indices[new_hits]
        if hit_global_idx.numel() > 0:
            flat_hit[hit_global_idx] = True
            flat_active[hit_global_idx] = False
            flat_hit_pos[hit_global_idx] = p_active[new_hits]
            flat_hit_color[hit_global_idx] = curr_colors[min_idx[new_hits]]

        miss_global_idx = active_indices[miss_cond]
        if miss_global_idx.numel() > 0:
            flat_active[miss_global_idx] = False

        still_active_local = ~(hit_cond | miss_cond)
        if still_active_local.any():
            step_dist = min_sdf[still_active_local]
            step_dist = torch.max(step_dist, torch.tensor(1e-4, device=device))
            flat_t[active_indices[still_active_local]] += step_dist

    # Shading
    hit_indices = torch.nonzero(flat_hit).squeeze()
    if hit_indices.numel() > 0:
        p_hit = flat_hit_pos[hit_indices]
        hit_cols = flat_hit_color[hit_indices]

        e = 1e-3

        def get_sdf_batch(points):
            pa = points.unsqueeze(1) - curr_starts.unsqueeze(0)
            proj = (pa * ba_norm.unsqueeze(0)).sum(dim=-1)
            proj_clamped = proj.clamp(min=0.0).min(ba_len.unsqueeze(0))
            closest = (curr_starts.unsqueeze(0)
                       + proj_clamped.unsqueeze(-1) * ba_norm.unsqueeze(0))
            dist = torch.norm(points.unsqueeze(1) - closest, dim=-1)
            return (dist - radius).min(dim=1)[0]

        def get_normal_batch(points):
            dx = (get_sdf_batch(points + torch.tensor([e, 0, 0], device=device))
                  - get_sdf_batch(points - torch.tensor([e, 0, 0], device=device)))
            dy = (get_sdf_batch(points + torch.tensor([0, e, 0], device=device))
                  - get_sdf_batch(points - torch.tensor([0, e, 0], device=device)))
            dz = (get_sdf_batch(points + torch.tensor([0, 0, e], device=device))
                  - get_sdf_batch(points - torch.tensor([0, 0, e], device=device)))
            n = torch.stack([dx, dy, dz], dim=-1)
            return n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)

        normals = get_normal_batch(p_hit)
        view_dir = -flat_ray_dirs[hit_indices]
        view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)

        diff = torch.clamp((normals * (-light_dir)).sum(dim=-1), min=0.0)
        half_dir = (view_dir + (-light_dir)).float()
        half_dir = half_dir / (torch.norm(half_dir, dim=-1, keepdim=True) + 1e-8)
        spec = torch.clamp((normals * half_dir).sum(dim=-1), min=0.0) ** 32

        z_vals = p_hit[:, 2]
        depth_factor = (1.0 - (z_vals - depth_near) / (depth_far - znear)).clamp(0.0, 1.0)

        diffuse_term = 0.3 + 0.7 * diff
        base_rgb = hit_cols[:, :3] * diffuse_term.unsqueeze(-1) * depth_factor.unsqueeze(-1)
        highlight = (torch.tensor([1.0, 1.0, 1.0], device=device)
                     * (0.5 * spec.unsqueeze(-1)) * depth_factor.unsqueeze(-1))
        final_rgb = base_rgb + highlight

        flat_hit_color[hit_indices, :3] = final_rgb
        flat_hit_color[hit_indices, 3] = hit_cols[:, 3]

    frame_img = flat_hit_color.view(H, W, 4)
    return (frame_img.clamp(0, 1) * 255).byte().cpu().numpy()


# ── Public API ───────────────────────────────────────────────────────────────

def build_cylinder_specs(joints_3d, threshold_2d=None, keypoints_2d=None):
    """
    Build cylinder specs from SMPLest-X 3D body joints.

    Args:
        joints_3d: (137, 3) or (25, 3) absolute 3D joints in camera space
        threshold_2d: confidence threshold (only used with keypoints_2d)
        keypoints_2d: optional (137, 3) with confidence in column 2

    Returns:
        list of (start_3d, end_3d, color_rgba) tuples
    """
    specs = []
    for limb_idx in SCAIL_DRAW_SEQ:
        a, b = SCAIL_LIMBS[limb_idx]
        if a >= joints_3d.shape[0] or b >= joints_3d.shape[0]:
            continue

        # Skip if 2D confidence too low
        if keypoints_2d is not None and threshold_2d is not None:
            if (keypoints_2d[a, 2] < threshold_2d
                    or keypoints_2d[b, 2] < threshold_2d):
                continue

        start = joints_3d[a].tolist()
        end = joints_3d[b].tolist()

        # Skip degenerate
        if (abs(start[0]) + abs(start[1]) + abs(start[2]) < 0.01
                or abs(end[0]) + abs(end[1]) + abs(end[2]) < 0.01):
            continue

        color = SCAIL_LIMB_COLORS_NORM[limb_idx]
        specs.append((start, end, color))
    return specs


def render_scail_pose(img, keypoints, threshold=0.1,
                      joint_cam_3d=None, root_cam=None, cfg=None,
                      inv_trans=None):
    """
    Render SCAIL-style pose on image from 137-joint SMPLest-X keypoints.

    When 3D data (joint_cam_3d, root_cam, cfg, inv_trans) is provided,
    body limbs are rendered as 3D cylinders with Blinn-Phong shading.
    Otherwise falls back to 2D ellipse rendering.

    Args:
        img: (H, W, 3) uint8 canvas (typically black).
        keypoints: (137, 3) array with (x, y, confidence) per joint.
        threshold: Minimum confidence to draw.
        joint_cam_3d: (137, 3) root-relative 3D joints (optional).
        root_cam: (3,) absolute root position in camera space (optional).
        cfg: SMPLest-X config with model.focal, model.princpt, etc. (optional).
        inv_trans: (2, 3) affine from input_img_shape → original image (optional).

    Returns:
        (H, W, 3) uint8 image with SCAIL-style skeleton.
    """
    img = np.ascontiguousarray(img.copy())
    h, w = img.shape[:2]

    use_3d = (joint_cam_3d is not None and root_cam is not None
              and cfg is not None and inv_trans is not None)

    def _valid(idx):
        return keypoints[idx, 2] > threshold

    def _pt(idx):
        return (int(round(keypoints[idx, 0])), int(round(keypoints[idx, 1])))

    # ── Body limbs ────────────────────────────────────────────────────────
    if use_3d:
        # 3D cylinder rendering
        joints_abs = joint_cam_3d + root_cam[None, :]  # absolute camera coords

        # Build cylinder specs from 3D body joints
        specs = build_cylinder_specs(joints_abs, threshold, keypoints)

        if specs:
            # Compute intrinsics for input_img_shape space
            focal = cfg.model.focal
            princpt = cfg.model.princpt
            input_body = cfg.model.input_body_shape
            input_img = cfg.model.input_img_shape

            # Scale focal/princpt from input_body_shape to input_img_shape
            scale_x = input_img[1] / input_body[1]
            scale_y = input_img[0] / input_body[0]
            render_fx = focal[0] * scale_x
            render_fy = focal[1] * scale_y
            render_cx = princpt[0] * scale_x
            render_cy = princpt[1] * scale_y

            render_h = int(input_img[0])
            render_w = int(input_img[1])

            # Render 3D cylinders at input_img_shape resolution
            rendered_rgba = _render_cylinders_torch(
                specs, render_h, render_w,
                fx=render_fx, fy=render_fy,
                cx=render_cx, cy=render_cy,
                radius=21.5,
            )

            # Warp rendered body to original image space via inv_trans
            rendered_rgb = rendered_rgba[:, :, :3]
            rendered_alpha = rendered_rgba[:, :, 3]

            body_warped = cv2.warpAffine(
                rendered_rgb, inv_trans, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            alpha_warped = cv2.warpAffine(
                rendered_alpha, inv_trans, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            # Composite body onto canvas
            mask = alpha_warped > 10
            img[mask] = body_warped[mask]
    else:
        # 2D fallback: ellipse-fill
        stickwidth = max(2, int(min(h, w) / 200))

        def _xy(idx):
            return keypoints[idx, 0], keypoints[idx, 1]

        body_layer = np.zeros_like(img)
        for i, (a, b) in enumerate(SCAIL_LIMBS):
            if not (_valid(a) and _valid(b)):
                continue
            x1, y1 = _xy(a)
            x2, y2 = _xy(b)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if length < 1:
                continue
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            polygon = cv2.ellipse2Poly(
                (int(mx), int(my)),
                (int(length / 2), stickwidth),
                int(angle), 0, 360, 1,
            )
            cv2.fillConvexPoly(body_layer, polygon, SCAIL_LIMB_COLORS_255[i])

        body_layer = (body_layer * 0.6).astype(np.uint8)
        mask = body_layer > 0
        img[mask] = body_layer[mask]

    # ── Hands (DWPose HSV rainbow style) ──────────────────────────────────
    hand_thick = max(1, int(min(h, w) / 300))
    hand_dot_rad = max(1, hand_thick)

    for hand_start, wrist_idx in [(25, 12), (45, 13)]:  # left, right
        hand_pts = np.zeros((21, 3), dtype=np.float32)
        hand_pts[0] = keypoints[wrist_idx]
        hand_pts[1:] = keypoints[hand_start:hand_start + 20]

        for ie, (ea, eb) in enumerate(_HAND_EDGES):
            if hand_pts[ea, 2] > threshold and hand_pts[eb, 2] > threshold:
                p1 = (int(round(hand_pts[ea, 0])), int(round(hand_pts[ea, 1])))
                p2 = (int(round(hand_pts[eb, 0])), int(round(hand_pts[eb, 1])))
                color = _hsv_to_rgb(ie / len(_HAND_EDGES), 1.0, 1.0)
                cv2.line(img, p1, p2, color, hand_thick, cv2.LINE_AA)

        for j in range(21):
            if hand_pts[j, 2] > threshold:
                pt = (int(round(hand_pts[j, 0])), int(round(hand_pts[j, 1])))
                cv2.circle(img, pt, hand_dot_rad, (0, 0, 255), thickness=-1)

    # ── Face (white dots, small radius) ───────────────────────────────────
    face_rad = max(1, int(min(h, w) / 500))
    for j in range(65, min(137, keypoints.shape[0])):
        if _valid(j):
            cv2.circle(img, _pt(j), face_rad, (255, 255, 255), -1, cv2.LINE_AA)

    return img
