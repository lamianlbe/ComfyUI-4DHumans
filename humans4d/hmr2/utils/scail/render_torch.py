import torch
import numpy as np
import random
import math


def flatten_specs(specs_list):
    """
    Flatten specs_list into numpy arrays + index tables.
    Returns:
        starts: (N, 3) float32
        ends:   (N, 3) float32
        colors: (N, 4) float32
        frame_offset: (num_frames,) int32
        frame_count:  (num_frames,) int32
    """
    starts, ends, colors = [], [], []
    frame_offset, frame_count = [], []
    offset = 0
    for specs in specs_list:
        frame_offset.append(offset)
        frame_count.append(len(specs))
        for s, e, c in specs:
            starts.append(s)
            ends.append(e)
            colors.append(c)
        offset += len(specs)

    # Handle empty case
    if len(starts) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.array(frame_offset, dtype=np.int32),
            np.array(frame_count, dtype=np.int32),
        )

    return (
        np.array(starts, dtype=np.float32),
        np.array(ends, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(frame_offset, dtype=np.int32),
        np.array(frame_count, dtype=np.int32),
    )


def _render_frame_torch(
    curr_starts, curr_ends, curr_colors, ba, ba_len, ba_norm,
    flat_ray_dirs, flat_ray_origins, H, W, radius, znear, zfar,
    z_min_val, z_max_val, light_dir, device,
):
    """Render a single frame's cylinders. Returns (H, W, 4) uint8 numpy."""
    n_pix = H * W

    # Smart t initialization: start rays near the scene instead of at znear
    # Ray z-component determines how fast we advance in z.
    # For each pixel, t_start = max(z_min / ray_dir_z, znear)
    ray_z = flat_ray_dirs[:, 2].clamp(min=1e-6)
    t_start = ((z_min_val - radius * 2) / ray_z).clamp(min=znear)

    flat_t = t_start.clone()
    flat_active = torch.ones(n_pix, dtype=torch.bool, device=device)
    flat_hit = torch.zeros(n_pix, dtype=torch.bool, device=device)
    flat_hit_color = torch.zeros((n_pix, 4), device=device)
    flat_hit_pos = torch.zeros((n_pix, 3), device=device)

    depth_near = max(z_min_val, 0.1)
    depth_far = min(z_max_val + 6000, 20000)

    MAX_STEPS = 64
    EPSILON = 1e-3

    for step in range(MAX_STEPS):
        active_indices = torch.nonzero(flat_active).reshape(-1)
        if active_indices.numel() == 0:
            break

        p_active = flat_ray_origins[active_indices] + \
            flat_ray_dirs[active_indices] * flat_t[active_indices].unsqueeze(1)

        pa = p_active.unsqueeze(1) - curr_starts.unsqueeze(0)
        proj = (pa * ba_norm.unsqueeze(0)).sum(dim=-1)
        proj_clamped = proj.clamp(min=0.0).min(ba_len.unsqueeze(0))
        closest_on_axis = curr_starts.unsqueeze(0) + \
            proj_clamped.unsqueeze(-1) * ba_norm.unsqueeze(0)
        dist_euc = torch.norm(p_active.unsqueeze(1) - closest_on_axis, dim=-1)
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
            step_dist = min_sdf[still_active_local].clamp(min=1e-4)
            flat_t[active_indices[still_active_local]] += step_dist

    # --- Shading ---
    hit_indices = torch.nonzero(flat_hit).reshape(-1)
    if hit_indices.numel() > 0:
        p_hit = flat_hit_pos[hit_indices]
        hit_cols = flat_hit_color[hit_indices]

        e = 1e-3

        def get_sdf_batch(points):
            pa = points.unsqueeze(1) - curr_starts.unsqueeze(0)
            proj = (pa * ba_norm.unsqueeze(0)).sum(dim=-1)
            proj_clamped = proj.clamp(min=0.0).min(ba_len.unsqueeze(0))
            closest = curr_starts.unsqueeze(0) + \
                proj_clamped.unsqueeze(-1) * ba_norm.unsqueeze(0)
            dist = torch.norm(points.unsqueeze(1) - closest, dim=-1)
            return (dist - radius).min(dim=1)[0]

        offsets = [
            torch.tensor([e, 0, 0], device=device),
            torch.tensor([0, e, 0], device=device),
            torch.tensor([0, 0, e], device=device),
        ]
        grads = []
        for off in offsets:
            grads.append(get_sdf_batch(p_hit + off) - get_sdf_batch(p_hit - off))
        normals = torch.stack(grads, dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

        view_dir = -flat_ray_dirs[hit_indices]
        view_dir = view_dir / (torch.norm(view_dir, dim=-1, keepdim=True) + 1e-8)

        diff = torch.clamp((normals * (-light_dir)).sum(dim=-1), min=0.0)
        half_dir = (view_dir + (-light_dir)).float()
        half_dir = half_dir / (torch.norm(half_dir, dim=-1, keepdim=True) + 1e-8)
        spec = torch.clamp((normals * half_dir).sum(dim=-1), min=0.0) ** 32

        z_vals = p_hit[:, 2]
        depth_factor = (1.0 - (z_vals - depth_near) / (depth_far - znear)).clamp(0.0, 1.0)

        diffuse_term = 0.3 + 0.7 * diff
        base_rgb = hit_cols[:, :3] * diffuse_term.unsqueeze(-1) * depth_factor.unsqueeze(-1)
        highlight = 0.5 * spec.unsqueeze(-1) * depth_factor.unsqueeze(-1)

        flat_hit_color[hit_indices, :3] = base_rgb + highlight
        flat_hit_color[hit_indices, 3] = hit_cols[:, 3]

    frame_img = flat_hit_color.view(H, W, 4)
    return (frame_img.clamp(0, 1) * 255).byte().cpu().numpy()


def render_whole(
    specs_list, H=480, W=640, fx=500, fy=500, cx=240, cy=320, radius=21.5, device=None
):
    """
    Render cylinders using PyTorch ray marching (single set of intrinsics).
    """
    return render_whole_batch(
        specs_list, H=H, W=W, radius=radius, device=device,
        intrinsics_list=[(fx, fy, cx, cy)] * len(specs_list),
    )


def render_whole_batch(
    specs_list, H=480, W=640, radius=21.5, device=None,
    intrinsics_list=None,
):
    """
    Render cylinders with per-frame camera intrinsics.
    Geometry is uploaded to GPU once; ray grid is recomputed per-frame only
    when intrinsics differ from the previous frame.

    Args:
        specs_list: list of cylinder specs per frame
        intrinsics_list: list of (fx, fy, cx, cy) tuples, one per frame
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_frames = len(specs_list)

    starts_np, ends_np, colors_np, frame_offset_np, frame_count_np = flatten_specs(
        specs_list
    )

    if len(starts_np) == 0:
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(n_frames)]

    # Upload all geometry to GPU once
    all_starts = torch.from_numpy(starts_np).to(device).float()
    all_ends = torch.from_numpy(ends_np).to(device).float()
    all_colors = torch.from_numpy(colors_np).to(device).float()

    z_min_val = min(starts_np[:, 2].min(), ends_np[:, 2].min())
    z_max_val = max(starts_np[:, 2].max(), ends_np[:, 2].max())
    znear = 0.1
    zfar = max(min(z_max_val, 25000), 10000)

    light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Pixel coordinate grid (computed once, reused)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device).float(),
        torch.arange(W, device=device).float(),
        indexing="ij",
    )
    flat_ray_origins = torch.zeros((H * W, 3), device=device)

    # Cache ray dirs for repeated intrinsics
    prev_intrinsics = None
    flat_ray_dirs = None

    rendered_frames = []
    for i in range(n_frames):
        start_idx = frame_offset_np[i]
        count = frame_count_np[i]

        if count == 0:
            rendered_frames.append(np.zeros((H, W, 4), dtype=np.uint8))
            continue

        fx, fy, cx, cy = intrinsics_list[i]

        # Recompute ray directions only if intrinsics changed
        if (fx, fy, cx, cy) != prev_intrinsics:
            u = (x_coords - cx) / fx
            v = (y_coords - cy) / fy
            z = torch.ones_like(u)
            ray_dirs = torch.stack([u, v, z], dim=-1)
            ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
            flat_ray_dirs = ray_dirs.view(-1, 3)
            prev_intrinsics = (fx, fy, cx, cy)

        curr_starts = all_starts[start_idx: start_idx + count]
        curr_ends = all_ends[start_idx: start_idx + count]
        curr_colors = all_colors[start_idx: start_idx + count]

        ba = curr_ends - curr_starts
        ba_len = torch.sqrt((ba * ba).sum(dim=1))
        ba_norm = ba / ba_len.unsqueeze(1)

        frame_np = _render_frame_torch(
            curr_starts, curr_ends, curr_colors, ba, ba_len, ba_norm,
            flat_ray_dirs, flat_ray_origins, H, W, radius, znear, zfar,
            z_min_val, z_max_val, light_dir, device,
        )
        rendered_frames.append(frame_np)

    return rendered_frames


def random_cylinder():
    """Generate a random cylinder (start, end, color)."""
    # Start point [-200,200]^2, z in [300,400]
    ax = random.uniform(-200, 200)
    ay = random.uniform(-200, 200)
    az = random.uniform(300, 400)
    start = [ax, ay, az]

    # Random direction and length
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(-math.pi / 4, math.pi / 4)  # Tilt angle
    L = 100
    dx = math.cos(phi) * math.cos(theta)
    dy = math.cos(phi) * math.sin(theta)
    dz = math.sin(phi)
    end = [ax + dx * L, ay + dy * L, az + dz * L]

    # Random color (RGB + alpha=1)
    color = [random.random(), random.random(), random.random(), 1.0]

    return (start, end, color)


def generate_specs_list(num_frames=120, min_cyl=10, max_cyl=120):
    """Generate specs_list, each frame has several random cylinders."""
    specs_list = []
    for _ in range(num_frames):
        n_cyl = random.randint(min_cyl, max_cyl)
        specs = [random_cylinder() for _ in range(n_cyl)]
        specs_x_shift = [
            (
                [spec[0][0] + 50, spec[0][1], spec[0][2]],
                [spec[1][0] + 50, spec[1][1], spec[1][2]],
                spec[2],
            )
            for spec in specs
        ]
        specs_list.append(specs)
        specs_list.append(specs_x_shift)
    return specs_list
