#!/usr/bin/env python3
"""
Dominant-joint colour render, round 3.

Colours every rendered vertex by which skin joint DOMINATES its
WEIGHTS_0 (the same "dominant joint" definition the manager's numpy LBS
audit and scripts/pipeline/verify_kitty_v2.py both use), so a reviewer
can see at a glance whether each arm is ONE continuous limb -- a single
unbroken colour run from shoulder to paw -- or whether a stray lobe
elsewhere on the body carries an arm-chain colour it should not.

Colour key (fixed, side-paired so left/right are visually comparable):
  LeftShoulder/RightShoulder   dark blue / dark red   (clavicle)
  Left*Arm*/Right*Arm*         blue / red              (upper arm + twists)
  Left*ForeArm*/Right*ForeArm* cyan / orange           (forearm + twists)
  LeftHand/RightHand           light blue / yellow     (paw)
  everything else              light grey              (body -- not arm)

Renders at:
  * the BIND pose (no animation sampled -- scene.get_positions()), and
  * the "wave" clip's own measured peak-hand-height frame (t=2.30s this
    round, matching verify_kitty_v2.py's own peak-frame detection) so the
    arm's colour run can be checked while it is actually raised, not just
    at rest.

Usage:
    python3 dominant_joint_render.py <glb> <out.png> [--pose bind|wave]
        [--time T] [--azimuth DEG] [--width W] [--height H] [--title T]
"""
import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene
from render_preview_textured import camera_basis

COLOR_MAP = {
    "LeftShoulder": (30, 30, 140),
    "LeftArm": (50, 90, 230),
    "LeftArmTwist1": (50, 90, 230),
    "LeftArmTwist2": (50, 90, 230),
    "LeftForeArm": (60, 190, 210),
    "LeftForeArmTwist1": (60, 190, 210),
    "LeftForeArmTwist2": (60, 190, 210),
    "LeftHand": (140, 220, 255),
    "RightShoulder": (140, 20, 20),
    "RightArm": (230, 60, 60),
    "RightArmTwist1": (230, 60, 60),
    "RightArmTwist2": (230, 60, 60),
    "RightForeArm": (240, 150, 40),
    "RightForeArmTwist1": (240, 150, 40),
    "RightForeArmTwist2": (240, 150, 40),
    "RightHand": (250, 230, 60),
}
BODY_COLOR = (210, 210, 210)
BG = (255, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_path")
    ap.add_argument("out_path")
    ap.add_argument("--pose", choices=["bind", "anim"], default="bind")
    ap.add_argument("--anim", default=None)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.glb_path)
    g = scene.g

    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]
    joints0, weights0 = scene.get_skin_weights()
    dominant_slot = joints0[np.arange(len(joints0)), np.argmax(weights0, axis=1)]
    dominant_name = np.array([joint_names[s] for s in dominant_slot])

    vertex_color = np.tile(np.array(BODY_COLOR, dtype=np.float32), (len(dominant_name), 1))
    for name, col in COLOR_MAP.items():
        vertex_color[dominant_name == name] = np.array(col, dtype=np.float32)
    vertex_color /= 255.0

    if args.pose == "anim" and args.anim:
        anim = scene.get_animation(args.anim)
        positions = scene.skinned_positions_at(anim, args.time)
    else:
        positions = scene.get_positions()

    indices = scene.get_indices()
    tris = indices.reshape(-1, 3)

    cam_dir, right_v, up_v = camera_basis(args.azimuth, 0.0)
    proj_x = positions @ right_v
    proj_y = positions @ up_v
    depth_val = positions @ cam_dir

    v0 = positions[tris[:, 0]]
    v1 = positions[tris[:, 1]]
    v2 = positions[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(normals, axis=1, keepdims=True)
    nl[nl == 0] = 1e-9
    normals = normals / nl
    light_dir = (cam_dir * 0.75 + up_v * 0.45 + right_v * 0.2)
    light_dir = light_dir / np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.45, 1.0)

    xmin, xmax = proj_x.min(), proj_x.max()
    ymin, ymax = proj_y.min(), proj_y.max()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * 1.15
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    width, height = args.width, args.height

    def to_pixel(x, y):
        px = (x - xlim[0]) / (xlim[1] - xlim[0]) * width
        py = height - (y - ylim[0]) / (ylim[1] - ylim[0]) * height
        return px, py

    px_all, py_all = to_pixel(proj_x, proj_y)
    framebuffer = np.ones((height, width, 3), dtype=np.float32) * (np.array(BG) / 255.0)
    zbuffer = np.full((height, width), -np.inf, dtype=np.float32)

    face_depth = depth_val[tris].mean(axis=1)
    order = np.argsort(face_depth)

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
        wA = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / area
        wB = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / area
        wC = 1.0 - wA - wB
        inside = (wA >= -1e-4) & (wB >= -1e-4) & (wC >= -1e-4)
        if not inside.any():
            continue
        depth_interp = wA * d[0] + wB * d[1] + wC * d[2]
        sub_y, sub_x = np.where(inside)
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

        cA, cB, cC = vertex_color[tri[0]], vertex_color[tri[1]], vertex_color[tri[2]]
        color = (wA_s[:, None] * cA + wB_s[:, None] * cB + wC_s[:, None] * cC)
        color = color * ndotl[ti]

        zbuffer[py_idx, px_idx] = d_vals
        framebuffer[py_idx, px_idx] = color

    img_out = np.clip(framebuffer, 0, 1)
    img = Image.fromarray((img_out * 255).astype(np.uint8), mode="RGB")
    if args.title:
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, 22], fill=(255, 255, 255))
        draw.text((6, 4), args.title, fill=(0, 0, 0))
    img.save(args.out_path)
    print("wrote", args.out_path)


if __name__ == "__main__":
    main()
