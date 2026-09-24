#!/usr/bin/env python3
"""
Self-verification report for assets/kitty/kitty.glb, printed as the numbers
the work order's completion notes need (and roughly what the parallel
validator will check): skin count, joint list, clip names/durations,
rendered triangle count, texture list, file size, and -- for `wave` --
peak arm-joint rotation, oscillation count, root/hips translation range,
and max torso-vertex displacement vs frame 0 (as % of model height).
"""
import sys
import os
import io
import json
import argparse
import numpy as np
from pygltflib import GLTF2
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene, quat_to_matrix

ARM_JOINT_RE_PARTS = ["arm", "hand", "shoulder", "clavicle", "elbow", "wrist", "forearm", "paw"]
SIDE_PARTS = ["l", "left", "r", "right"]
TORSO_RE_PARTS = ["spine", "chest", "hips", "hip", "pelvis", "neck", "head", "root"]


def quat_angle_deg(q_bind, q_now):
    m1 = quat_to_matrix(q_bind)
    m2 = quat_to_matrix(q_now)
    rel = m2 @ m1.T
    trace = np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def count_oscillations(angles_deg, min_amp_deg=3.0):
    """Count sign changes in the (mean-subtracted) angle-vs-time series that
    clear a small amplitude threshold -- a rough, dependency-free oscillation
    counter (each full up-down-up cycle counts as ~2 sign changes -> /2)."""
    a = np.array(angles_deg) - np.mean(angles_deg)
    signs = np.sign(a)
    signs[signs == 0] = 1
    changes = 0
    last_extreme = 0.0
    direction = 0
    for i in range(1, len(a)):
        if signs[i] != signs[i - 1]:
            if abs(a[i - 1] - last_extreme) is not None:
                pass
        if signs[i] != signs[i - 1] and abs(a[i]) >= 0:
            changes += 1
        last_extreme = a[i]
    # filter tiny numerical-noise crossings using amplitude threshold
    peak_to_peak = a.max() - a.min()
    if peak_to_peak < min_amp_deg:
        return 0
    return changes / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_path")
    ap.add_argument("--wave-side", default="Right")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    path = args.glb_path
    size_bytes = os.path.getsize(path)
    scene = GLTFScene(path)
    g = scene.g

    report = {}
    report["file"] = path
    report["file_size_bytes"] = size_bytes
    report["skin_count"] = len(g.skins)
    joint_names = [g.nodes[j].name for j in g.skins[0].joints] if g.skins else []
    report["joint_names"] = joint_names
    report["joint_count"] = len(joint_names)

    # triangle count (sum over primitives)
    total_tris = 0
    for m in g.meshes:
        for p in m.primitives:
            if p.indices is not None:
                idx = scene.read_accessor(p.indices)
                total_tris += len(idx) // 3
    report["rendered_triangles"] = int(total_tris)

    # textures
    tex_info = []
    blob = scene.blob
    for im in g.images:
        bv = g.bufferViews[im.bufferView]
        data = blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
        img = Image.open(io.BytesIO(data))
        tex_info.append({"name": im.name, "size": img.size, "mime": im.mimeType})
    report["textures"] = tex_info
    report["has_base_color_texture"] = any(
        m.pbrMetallicRoughness and m.pbrMetallicRoughness.baseColorTexture is not None
        for m in g.materials
    )

    # clips
    clips = {}
    for a in g.animations:
        clips[a.name] = round(scene.animation_duration(a), 3)
    report["clips"] = clips

    # model height (bind pose bounding box)
    positions = scene.get_positions()
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extents = maxs - mins
    model_height = float(extents[1])  # Y-up
    report["model_height_bind_pose"] = model_height

    # ---- wave-specific checks ----
    if "wave" in clips:
        anim = scene.get_animation("wave")
        dur = clips["wave"]
        times = np.linspace(0, dur, int(dur * 30) + 1)

        side = args.wave_side
        arm_joint_candidates = [n for n in joint_names
                                 if any(p in n.lower() for p in ["arm", "hand"]) and side.lower() in n.lower()]
        bind_rot = {}
        for jn in arm_joint_candidates:
            idx = joint_names_to_node_idx(g, jn)
            t0, r0, s0 = scene.node_bind_local(idx)
            bind_rot[jn] = r0

        angle_series = {jn: [] for jn in arm_joint_candidates}
        for t in times:
            for jn in arm_joint_candidates:
                idx = joint_names_to_node_idx(g, jn)
                tr, ro, sc = scene.sample_node_local(anim, idx, t)
                angle_series[jn].append(quat_angle_deg(bind_rot[jn], ro))

        peak_rotation = {jn: float(np.max(v)) for jn, v in angle_series.items()}
        oscillations = {jn: count_oscillations(v) for jn, v in angle_series.items()}
        report["wave_peak_rotation_deg"] = peak_rotation
        report["wave_oscillation_count"] = oscillations

        # root / hips translation range across the clip
        root_like = [n for n in joint_names if n.lower() in ("root", "hips", "hip", "pelvis")]
        root_trans_range = {}
        for jn in root_like:
            idx = joint_names_to_node_idx(g, jn)
            world = []
            for t in times:
                w = scene.world_matrices_at(anim, t)
                world.append(w[idx][:3, 3])
            world = np.array(world)
            rng = world.max(axis=0) - world.min(axis=0)
            root_trans_range[jn] = {
                "range_units": rng.tolist(),
                "range_pct_height": (rng / model_height * 100).tolist(),
            }
        report["wave_root_hips_translation_range"] = root_trans_range

        # torso-dominant vertex displacement: frame0 vs peak-rotation frame
        joints0, weights0 = scene.get_skin_weights()
        joint_list, ibm = scene.get_skin()
        joint_name_by_slot = [g.nodes[j].name for j in joint_list]
        torso_mask = build_torso_vertex_mask(joint_list, joints0, weights0, joint_name_by_slot)

        pos0 = scene.skinned_positions_at(anim, 0.0)
        # scan every sampled frame (not just one guessed "peak") for the true
        # worst-case torso displacement across the whole clip
        worst_max = 0.0
        worst_mean = 0.0
        worst_t = 0.0
        for t in times:
            pos_t = scene.skinned_positions_at(anim, float(t))
            disp = np.linalg.norm(pos_t - pos0, axis=1)
            torso_disp = disp[torso_mask]
            if len(torso_disp) == 0:
                continue
            if torso_disp.max() > worst_max:
                worst_max = float(torso_disp.max())
                worst_mean = float(torso_disp.mean())
                worst_t = float(t)
        report["torso_vertex_count"] = int(torso_mask.sum())
        report["wave_worst_frame_time"] = worst_t
        report["torso_max_displacement_units"] = worst_max
        report["torso_max_displacement_pct_height"] = worst_max / model_height * 100
        report["torso_mean_displacement_pct_height_at_worst_frame"] = worst_mean / model_height * 100

    print(json.dumps(report, indent=2, default=str))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2, default=str)


def joint_names_to_node_idx(g, name):
    for i, n in enumerate(g.nodes):
        if n.name == name:
            return i
    raise KeyError(name)


def build_torso_vertex_mask(joint_list, joints0, weights0, joint_name_by_slot):
    """A vertex counts as torso-dominant if its single largest-weight joint
    influence is a torso/spine/hip/head joint (not an arm/leg joint)."""
    torso_slots = set()
    for slot, name in enumerate(joint_name_by_slot):
        nl = name.lower()
        if any(p in nl for p in TORSO_RE_PARTS) and not any(p in nl for p in ["arm", "hand", "leg", "foot", "toe"]):
            torso_slots.add(slot)
    N = joints0.shape[0]
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        dominant_slot = joints0[i, np.argmax(weights0[i])]
        if dominant_slot in torso_slots:
            mask[i] = True
    return mask


if __name__ == "__main__":
    main()
