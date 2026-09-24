#!/usr/bin/env python3
"""
Textured orthographic preview renderer for the kitty GLB.

No GPU / OpenGL dependency (this sandbox has neither a display nor a
working PyOpenGL build): this is a small software rasterizer (z-buffered,
numpy-vectorized per triangle) that samples the model's actual PBR base
color texture via UV coordinates and applies simple Lambert shading, so
preview renders show real fur color / belly patch / face markings instead
of a flat fill color (round 1's previews were untextured -- advisory
finding #16).

Camera model: orthographic, parameterized by azimuth (degrees, rotation
around the model's Y-up axis) and an optional zoom/crop for close-ups.
Azimuth 0 = the model's own verified forward axis pointing at the camera
("front"); the mapping from azimuth to world axes was re-derived from
this round's regenerated mesh (see keyframe_wave.py's docstring for how
the +X forward axis was confirmed), not assumed from round 1. Azimuth 180
= back, +/-90 = side views, +/-45/135 = the "3/4" views the work order
asks for. The renderer never mirrors the projection (round 1's advisory
finding #16) -- azimuth sweeps consistently the same rotational direction
for every view, and a front-view azimuth=0 render puts screen-left on the
character's own left, matching a viewer looking at the character (i.e. a
"mirror" convention, standard for character turnarounds and consistent
with the original reference/kitty-*.png views).

Usage:
    python3 render_preview_textured.py <glb> <out.png> [--anim NAME]
        [--time T] [--azimuth DEG] [--elevation DEG] [--zoom Z]
        [--focus-y-frac LO HI] [--width W] [--height H] [--title T]
"""
import sys
import os
import io
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene

BG = (255, 255, 255)


def get_base_color_image(scene: GLTFScene):
    g = scene.g
    for m in g.materials:
        if m.pbrMetallicRoughness and m.pbrMetallicRoughness.baseColorTexture is not None:
            tex_idx = m.pbrMetallicRoughness.baseColorTexture.index
            image_idx = g.textures[tex_idx].source
            im = g.images[image_idx]
            bv = g.bufferViews[im.bufferView]
            data = scene.blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.asarray(img, dtype=np.float32) / 255.0
    return None


def camera_basis(azimuth_deg, elevation_deg=0.0):
    """Camera looks toward the origin from direction (az, el). Azimuth 0
    points along +X (this model's verified forward axis) toward the
    origin -- i.e. the camera sits on the +X side for azimuth=0, matching
    the front view used throughout this pipeline. Azimuth increases
    counter-clockwise viewed from above (+Y)."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    # camera position direction (unit vector from origin toward camera)
    cam_dir = np.array([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)])
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, cam_dir)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.cross(cam_dir, right)
    up = up / np.linalg.norm(up)
    return cam_dir, right, up


def render(scene: GLTFScene, anim_name, t, out_path, azimuth=0.0, elevation=0.0,
           width=640, height=640, title=None, focus_y_frac=None, zoom=1.0, bg=BG,
           fixed_xlim=None, fixed_ylim=None):
    if anim_name is not None:
        anim = scene.get_animation(anim_name)
        positions = scene.skinned_positions_at(anim, t)
    else:
        positions = scene.get_positions()

    indices = scene.get_indices()
    tris = indices.reshape(-1, 3)
    prim = scene.get_mesh_primitive()
    uvs = scene.read_accessor(prim.attributes.TEXCOORD_0)
    tex = get_base_color_image(scene)
    tex_h, tex_w = (tex.shape[0], tex.shape[1]) if tex is not None else (0, 0)

    cam_dir, right_v, up_v = camera_basis(azimuth, elevation)
    proj_x = positions @ right_v
    proj_y = positions @ up_v
    depth_val = positions @ cam_dir  # larger = nearer camera

    v0 = positions[tris[:, 0]]
    v1 = positions[tris[:, 1]]
    v2 = positions[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1e-9
    normals = normals / norm_len
    light_dir = (cam_dir * 0.75 + up_v * 0.45 + right_v * 0.2)
    light_dir = light_dir / np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.35, 1.0)

    # framing: full model bbox unless a face-focus y-fraction window is given,
    # or an explicit fixed frame (so a sequence of frames -- e.g. jump's
    # crouch/apex/land -- can share one camera window and show the body
    # actually moving within it, instead of auto-fitting to each frame's own
    # bbox and hiding the motion).
    if fixed_xlim is not None and fixed_ylim is not None:
        xlim = fixed_xlim
        ylim = fixed_ylim
    else:
        all_mins = positions.min(axis=0)
        all_maxs = positions.max(axis=0)
        model_h = all_maxs[1] - all_mins[1]
        if focus_y_frac is not None:
            lo, hi = focus_y_frac
            y_lo = all_mins[1] + lo * model_h
            y_hi = all_mins[1] + hi * model_h
            vmask = (positions[:, 1] >= y_lo) & (positions[:, 1] <= y_hi)
            if vmask.sum() < 3:
                vmask = np.ones(len(positions), dtype=bool)
            xr = proj_x[vmask]
            yr = proj_y[vmask]
        else:
            xr = proj_x
            yr = proj_y
        xmin, xmax = xr.min(), xr.max()
        ymin, ymax = yr.min(), yr.max()
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_w = (xmax - xmin) / 2 / zoom * 1.15
        half_h = (ymax - ymin) / 2 / zoom * 1.15
        half = max(half_w, half_h)
        xlim = (cx - half, cx + half)
        ylim = (cy - half, cy + half)

    def to_pixel(x, y):
        px = (x - xlim[0]) / (xlim[1] - xlim[0]) * width
        py = height - (y - ylim[0]) / (ylim[1] - ylim[0]) * height
        return px, py

    px_all, py_all = to_pixel(proj_x, proj_y)

    framebuffer = np.ones((height, width, 3), dtype=np.float32) * (np.array(bg) / 255.0)
    zbuffer = np.full((height, width), -np.inf, dtype=np.float32)

    face_depth = depth_val[tris].mean(axis=1)
    order = np.argsort(face_depth)  # far to near; z-buffer also guards correctness

    for ti in order:
        tri = tris[ti]
        xs = px_all[tri]
        ys = py_all[tri]
        d = depth_val[tri]
        x_lo = max(int(np.floor(xs.min())), 0)
        x_hi = min(int(np.ceil(xs.max())), width - 1)
        y_lo = max(int(np.floor(ys.min())), 0)
        y_hi = min(int(np.ceil(ys.max())), height - 1)
        if x_hi < x_lo or y_hi < y_lo:
            continue
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        gx = gx.astype(np.float32) + 0.5
        gy = gy.astype(np.float32) + 0.5
        # standard barycentric weights: wA multiplies vertex tri[0], etc.
        wA = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / area
        wB = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / area
        wC = 1.0 - wA - wB
        inside = (wA >= -1e-4) & (wB >= -1e-4) & (wC >= -1e-4)
        if not inside.any():
            continue
        depth_interp = wA * d[0] + wB * d[1] + wC * d[2]
        sub_y, sub_x = np.where(inside)
        if len(sub_y) == 0:
            continue
        py_idx = y_lo + sub_y
        px_idx = x_lo + sub_x
        d_vals = depth_interp[sub_y, sub_x]
        cur_z = zbuffer[py_idx, px_idx]
        closer = d_vals > cur_z
        if not closer.any():
            continue
        py_idx = py_idx[closer]
        px_idx = px_idx[closer]
        wA_s = wA[sub_y, sub_x][closer]
        wB_s = wB[sub_y, sub_x][closer]
        wC_s = wC[sub_y, sub_x][closer]
        d_vals = d_vals[closer]

        if tex is not None:
            uvA, uvB, uvC = uvs[tri[0]], uvs[tri[1]], uvs[tri[2]]
            u = wA_s * uvA[0] + wB_s * uvB[0] + wC_s * uvC[0]
            v = wA_s * uvA[1] + wB_s * uvB[1] + wC_s * uvC[1]
            tx = np.clip((u * tex_w).astype(np.int32), 0, tex_w - 1)
            ty = np.clip((v * tex_h).astype(np.int32), 0, tex_h - 1)
            color = tex[ty, tx]
        else:
            color = np.tile(np.array([0.95, 0.65, 0.20]), (len(py_idx), 1))

        shade = ndotl[ti]
        color = color * shade

        zbuffer[py_idx, px_idx] = d_vals
        framebuffer[py_idx, px_idx] = color

    img_out = np.clip(framebuffer, 0, 1)
    img = Image.fromarray((img_out * 255).astype(np.uint8), mode="RGB")
    if title:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, 22], fill=(255, 255, 255))
        draw.text((6, 4), title, fill=(0, 0, 0))
    img.save(out_path)
    return out_path


def compute_fixed_frame(scene: GLTFScene, anim_name, times, azimuth=0.0, elevation=0.0, zoom=1.0, pad=0.2):
    """Union the projected bbox of the model across several sampled times of
    one clip, so a fixed camera window can be shared by a sequence of
    frames (e.g. jump's crouch/apex/land) and show the body actually
    moving within a stable frame."""
    cam_dir, right_v, up_v = camera_basis(azimuth, elevation)
    anim = scene.get_animation(anim_name)
    xs, ys = [], []
    for t in times:
        positions = scene.skinned_positions_at(anim, t)
        xs.append(positions @ right_v)
        ys.append(positions @ up_v)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_w = (xmax - xmin) / 2 / zoom * (1 + pad)
    half_h = (ymax - ymin) / 2 / zoom * (1 + pad)
    half = max(half_w, half_h)
    return (cx - half, cx + half), (cy - half, cy + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_path")
    ap.add_argument("out_path")
    ap.add_argument("--anim", default=None)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--focus-y-frac", type=float, nargs=2, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.glb_path)
    render(scene, args.anim, args.time, args.out_path, azimuth=args.azimuth,
           elevation=args.elevation, width=args.width, height=args.height,
           title=args.title, focus_y_frac=args.focus_y_frac, zoom=args.zoom)
    print("wrote", args.out_path)


if __name__ == "__main__":
    main()
