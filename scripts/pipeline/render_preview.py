#!/usr/bin/env python3
"""
Lightweight orthographic preview renderer for the kitty GLB.

No GPU / OpenGL dependency (this sandbox has neither a display nor a
working PyOpenGL build): this rasterises the *actual* skinned mesh
(evaluated at a requested animation + time via gltf_anim.GLTFScene) as a
flat/Lambert-shaded, back-to-front painter's-algorithm triangle render
using matplotlib. It's for reviewer-facing sanity previews (pose,
silhouette, torso stability), not a production renderer.
"""
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene

FUR_COLOR = np.array([0.95, 0.65, 0.20])
BELLY_HINT = np.array([0.98, 0.93, 0.80])


def render(scene: GLTFScene, anim_name, t, out_path, view="front", width=640, height=640,
            title=None):
    positions = scene.get_positions() if anim_name is None else None
    if anim_name is not None:
        anim = scene.get_animation(anim_name)
        positions = scene.skinned_positions_at(anim, t)
    else:
        positions = scene.get_positions()

    indices = scene.get_indices()
    tris = indices.reshape(-1, 3)

    # camera basis per view: model is Y-up; empirically (checked against the
    # mesh's own left-right vs front-back vertex-cloud symmetry) the model's
    # front-facing axis is X (depth) with left-right along Z.
    if view == "front":
        right, up, depth, mirror, depth_sign = 2, 1, 0, False, +1
    elif view == "back":
        right, up, depth, mirror, depth_sign = 2, 1, 0, True, -1
    elif view == "left":
        right, up, depth, mirror, depth_sign = 0, 1, 2, False, +1
    elif view == "right":
        right, up, depth, mirror, depth_sign = 0, 1, 2, True, -1
    else:
        raise ValueError(view)

    P = positions
    proj_x = P[:, right] * (-1 if mirror else 1)
    proj_y = P[:, up]
    depth_val = P[:, depth] * depth_sign

    v0 = P[tris[:, 0]]
    v1 = P[tris[:, 1]]
    v2 = P[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1e-9
    normals = normals / norm_len

    light_dir = np.zeros(3)
    light_dir[right] = 0.35
    light_dir[up] = 0.55
    light_dir[depth] = 0.9 * depth_sign
    light_dir = light_dir / np.linalg.norm(light_dir)
    ndotl = np.clip(normals @ light_dir, 0.05, 1.0)

    face_depth = depth_val[tris].mean(axis=1)
    # No explicit backface culling: for a closed, non-self-intersecting mesh a
    # strict far-to-near painter's-algorithm sort already hides backfaces
    # behind the corresponding front faces. Convention: camera sits on the
    # +depth side looking toward -depth, so larger depth_val = nearer camera
    # -> ascending sort draws far first, near last (on top).
    order = np.argsort(face_depth)

    polys = []
    colors = []
    for ti in order:
        tri = tris[ti]
        poly = np.stack([proj_x[tri], proj_y[tri]], axis=1)
        polys.append(poly)
        shade = ndotl[ti]
        col = FUR_COLOR * shade + (1 - shade) * 0.15
        colors.append(np.clip(col, 0, 1))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    coll = PolyCollection(polys, facecolors=colors, edgecolors=colors, linewidths=0.4,
                           antialiased=False)
    ax.add_collection(coll)
    pad = 0.08
    xmin, xmax = proj_x.min(), proj_x.max()
    ymin, ymax = proj_y.min(), proj_y.max()
    xr = xmax - xmin
    yr = ymax - ymin
    ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
    ax.set_ylim(ymin - pad * yr, ymax + pad * yr)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_path")
    ap.add_argument("out_path")
    ap.add_argument("--anim", default=None)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--view", default="front", choices=["front", "back", "left", "right"])
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.glb_path)
    render(scene, args.anim, args.time, args.out_path, view=args.view, title=args.title)
    print("wrote", args.out_path)


if __name__ == "__main__":
    main()
