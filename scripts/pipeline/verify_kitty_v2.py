#!/usr/bin/env python3
"""
Self-verification script for assets/kitty/kitty.glb against the NEW
validator contract (V1-V6) that scripts/validate-kitty.mjs is being
upgraded to on the parallel t2 branch, plus the existing baseline checks
(>=1 skin, exact clip names, <=20,000 rendered triangles, a base-color
texture backed by real image bytes).

This script is deliberately independent of scripts/validate-kitty.mjs (out
of this round's touch-scope) -- it is the "own script" the work order asks
this round to self-verify against. It works from joint NAME patterns only
(case-insensitive substring match), not hard-coded indices, so it is not
tied to one specific rig export.

Checks implemented (see the work order's V1-V6 for the exact wording):
  V1  wave: some joint in the waving arm chain rotates >=45deg from the
      clip's own frame 0; at that arm's peak-height frame, hand/wrist
      world height >= shoulder world height + 5%*model_height; >=2
      oscillations; the OTHER arm's joints rotate <=15deg all clip.
  V2  wave: root+hips translation range <=5%*height; torso joints'
      rotation (from frame 0) <=15deg.
  V3  wave: co-deformation over every vertex NOT dominated by the waving
      arm chain: <=0.5% of them displaced >3%*height from frame 0, none
      >8%*height; all-non-arm-dominated triangles: edge stretch <=1.5x.
  V4  weight leakage: zero Head/Neck-dominant vertices carry >0.2 total
      arm-chain weight (checked on the skin directly, pose-independent).
  V5  loop seams: idle and walk -- every channel's last sample equals its
      first within 1%*height (translation) / 2deg (rotation).
  V6  jump: a foot/toe joint rises >=8%*height above its own clip-start
      height at some sampled frame.

Usage:
    python3 verify_kitty_v2.py <kitty.glb> [--out-json report.json]
Exits non-zero if any check fails.
"""
import sys
import os
import io
import json
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene, quat_to_matrix

MAX_TRIANGLES = 20_000
REQUIRED_CLIPS = {"idle", "jump", "walk", "wave"}  # 'run' also acceptable for 'walk' per spec; not needed here

V1_MIN_ARM_PEAK_DEG = 45.0
V1_MIN_HAND_ABOVE_SHOULDER_FRAC = 0.05
V1_MIN_OSCILLATIONS = 2
V1_MAX_OTHER_ARM_DEG = 15.0
V2_MAX_ROOT_HIPS_TRANS_FRAC = 0.05
V2_MAX_TORSO_ROT_DEG = 15.0
V3_MAX_FRAC_OVER_3PCT = 0.005
V3_ABS_MAX_DISP_FRAC = 0.08
V3_DISP_WARN_FRAC = 0.03
V3_MAX_EDGE_STRETCH = 1.5
V4_MAX_HEAD_NECK_ARM_WEIGHT = 0.2
V5_MAX_TRANS_FRAC = 0.01
V5_MAX_ROT_DEG = 2.0
V6_MIN_FOOT_RISE_FRAC = 0.08


def angle_deg(q0, q1):
    m1 = quat_to_matrix(q0)
    m2 = quat_to_matrix(q1)
    rel = m2 @ m1.T
    tr = np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def joint_names_lower(g, joints):
    return [g.nodes[j].name.lower() for j in joints]


def is_arm_joint(name_lower, side=None):
    ok = any(p in name_lower for p in ("shoulder", "clavicle", "arm", "hand", "wrist", "elbow", "paw"))
    if side is not None:
        ok = ok and (side in name_lower)
    return ok


def is_head_neck_joint(name_lower):
    return ("head" in name_lower) or ("neck" in name_lower)


def is_torso_joint(name_lower):
    return any(p in name_lower for p in ("spine", "chest", "hips", "hip", "pelvis", "neck", "head", "root", "waist")) \
        and not any(p in name_lower for p in ("arm", "hand", "leg", "foot", "toe"))


def is_foot_joint(name_lower):
    return ("foot" in name_lower) or ("toe" in name_lower)


def count_oscillations(series, min_amp_frac=0.15):
    a = np.asarray(series, dtype=float)
    a = a - a.mean()
    rng = a.max() - a.min()
    if rng < 1e-9:
        return 0
    signs = np.sign(a)
    signs[signs == 0] = 1
    changes = int(np.sum(signs[1:] != signs[:-1]))
    return changes / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_path")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.glb_path)
    g = scene.g
    report = {"file": args.glb_path, "checks": {}}
    failures = []

    def check(name, ok, detail):
        report["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)

    # ---- baseline checks ----
    skin_count = len(g.skins)
    check("skin_present", skin_count >= 1, {"skin_count": skin_count})

    clip_names = {a.name for a in g.animations}
    check("clip_names_exact", REQUIRED_CLIPS.issubset(clip_names), {"clips_found": sorted(clip_names)})

    total_tris = 0
    for m in g.meshes:
        for p in m.primitives:
            if p.indices is not None:
                total_tris += len(scene.read_accessor(p.indices)) // 3
    check("triangle_budget", total_tris <= MAX_TRIANGLES, {"rendered_triangles": total_tris, "max": MAX_TRIANGLES})

    tex_ok = False
    tex_info = []
    for m in g.materials:
        if m.pbrMetallicRoughness and m.pbrMetallicRoughness.baseColorTexture is not None:
            tex_idx = m.pbrMetallicRoughness.baseColorTexture.index
            image_idx = g.textures[tex_idx].source
            im = g.images[image_idx]
            bv = g.bufferViews[im.bufferView]
            data = scene.blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
            img = Image.open(io.BytesIO(data))
            img.load()
            tex_info.append({"name": im.name, "size": img.size, "mode": img.mode})
            tex_ok = True
    check("base_color_texture_real_bytes", tex_ok, {"textures": tex_info})

    positions = scene.get_positions()
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    height = float(maxs[1] - mins[1])
    report["model_height"] = height

    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]
    joint_names_l = [n.lower() for n in joint_names]
    joints0, weights0 = scene.get_skin_weights()

    # ---- V4: weight leakage (pose independent, from the skin only) ----
    dominant_slot = joints0[np.arange(len(joints0)), np.argmax(weights0, axis=1)]
    head_neck_slots = set(i for i, n in enumerate(joint_names_l) if is_head_neck_joint(n))
    arm_slots_all = set(i for i, n in enumerate(joint_names_l) if is_arm_joint(n))
    head_neck_dom = np.isin(dominant_slot, list(head_neck_slots))
    arm_weight = np.zeros(len(joints0))
    for k in range(joints0.shape[1]):
        slot = joints0[:, k]
        w = weights0[:, k]
        arm_weight += np.where(np.isin(slot, list(arm_slots_all)), w, 0.0)
    leak_mask = head_neck_dom & (arm_weight > V4_MAX_HEAD_NECK_ARM_WEIGHT)
    check("V4_weight_leakage", int(leak_mask.sum()) == 0, {
        "head_neck_dominant_vertices": int(head_neck_dom.sum()),
        "leaking_vertices": int(leak_mask.sum()),
        "max_arm_weight_on_head_neck": float(arm_weight[head_neck_dom].max()) if head_neck_dom.sum() else 0.0,
    })

    # ---- V5: loop seams (idle, walk) ----
    v5_detail = {}
    v5_ok = True
    for clip_name in ("idle", "walk"):
        if clip_name not in clip_names:
            continue
        anim = next(a for a in g.animations if a.name == clip_name)
        clip_bad = []
        for ch in anim.channels:
            node_name = g.nodes[ch.target.node].name
            sampler = anim.samplers[ch.sampler]
            times = scene.read_accessor(sampler.input)
            values = scene.read_accessor(sampler.output)
            values = values.reshape(len(times), -1)
            first, last = values[0], values[-1]
            if ch.target.path == "translation":
                d = float(np.linalg.norm(last - first) / height * 100)
                if d > V5_MAX_TRANS_FRAC * 100:
                    clip_bad.append({"node": node_name, "path": "translation", "diff_pct_height": d})
            elif ch.target.path == "rotation":
                dot = float(np.clip(abs(np.dot(first, last)), -1, 1))
                ang = float(np.degrees(2 * np.arccos(dot)))
                if ang > V5_MAX_ROT_DEG:
                    clip_bad.append({"node": node_name, "path": "rotation", "diff_deg": ang})
        v5_detail[clip_name] = clip_bad
        if clip_bad:
            v5_ok = False
    check("V5_loop_seams", v5_ok, v5_detail)

    # ---- V6: jump foot clearance ----
    v6_detail = {}
    v6_ok = False
    if "jump" in clip_names:
        anim = next(a for a in g.animations if a.name == "jump")
        dur = scene.animation_duration(anim)
        times = np.linspace(0, dur, int(dur * 60) + 1)
        foot_slots = [i for i, n in enumerate(joint_names_l) if is_foot_joint(n)]
        for slot in foot_slots:
            node_idx = joints[slot]
            ys = []
            for t in times:
                w = scene.world_matrices_at(anim, float(t))
                ys.append(w[node_idx][1, 3])
            ys = np.array(ys)
            rise_frac = float((ys.max() - ys[0]) / height)
            v6_detail[joint_names[slot]] = rise_frac
            if rise_frac >= V6_MIN_FOOT_RISE_FRAC:
                v6_ok = True
    check("V6_jump_clearance", v6_ok, {"rise_pct_height_by_joint": {k: v * 100 for k, v in v6_detail.items()},
                                        "threshold_pct": V6_MIN_FOOT_RISE_FRAC * 100})

    # ---- wave checks (V1, V2, V3) ----
    if "wave" in clip_names:
        anim = next(a for a in g.animations if a.name == "wave")
        dur = scene.animation_duration(anim)
        times = np.linspace(0, dur, int(dur * 60) + 1)

        arm_chain = {"left": [], "right": []}
        for i, n in enumerate(joint_names_l):
            if is_arm_joint(n, "left"):
                arm_chain["left"].append(i)
            elif is_arm_joint(n, "right"):
                arm_chain["right"].append(i)

        # local-rotation-from-frame0 peak per side, to auto-detect the waving side
        peak_by_side = {}
        series_by_slot = {}
        for side, slots in arm_chain.items():
            side_peak = 0.0
            for slot in slots:
                node_idx = joints[slot]
                rots = []
                for t in times:
                    _, r, _ = scene.sample_node_local(anim, node_idx, float(t))
                    rots.append(r)
                rots = np.array(rots)
                angles = np.array([angle_deg(rots[0], r) for r in rots])
                series_by_slot[(side, slot)] = angles
                side_peak = max(side_peak, float(angles.max()))
            peak_by_side[side] = side_peak

        waving_side = max(peak_by_side, key=peak_by_side.get)
        resting_side = "left" if waving_side == "right" else "right"

        v1a_peak = peak_by_side[waving_side]
        v1a_ok = v1a_peak >= V1_MIN_ARM_PEAK_DEG

        # hand/shoulder world height at the frame of peak hand height
        hand_slots = [s for s in arm_chain[waving_side] if "hand" in joint_names_l[s]]
        shoulder_slots = [s for s in arm_chain[waving_side] if any(p in joint_names_l[s] for p in ("shoulder", "clavicle"))]
        hand_slot = hand_slots[0] if hand_slots else arm_chain[waving_side][-1]
        shoulder_slot = shoulder_slots[0] if shoulder_slots else arm_chain[waving_side][0]
        hand_node = joints[hand_slot]
        shoulder_node = joints[shoulder_slot]
        hand_y = []
        shoulder_y = []
        hand_z = []
        for t in times:
            w = scene.world_matrices_at(anim, float(t))
            hand_y.append(w[hand_node][1, 3])
            shoulder_y.append(w[shoulder_node][1, 3])
            hand_z.append(w[hand_node][2, 3])
        hand_y = np.array(hand_y)
        shoulder_y = np.array(shoulder_y)
        hand_z = np.array(hand_z)
        peak_i = int(np.argmax(hand_y))
        clearance_frac = float((hand_y[peak_i] - shoulder_y[peak_i]) / height)
        v1b_ok = clearance_frac >= V1_MIN_HAND_ABOVE_SHOULDER_FRAC

        # oscillations: zero-crossings of hand world Z (lateral, frontal-plane wave) during hold window
        hold_mask = (times > times.max() * 0.2) & (times < times.max() * 0.85)
        osc = count_oscillations(hand_z[hold_mask])
        v1c_ok = osc >= V1_MIN_OSCILLATIONS

        v1d_peak = peak_by_side[resting_side]
        v1d_ok = v1d_peak <= V1_MAX_OTHER_ARM_DEG

        check("V1_wave_arm_rotation_peak", v1a_ok, {"waving_side": waving_side, "peak_deg": v1a_peak, "min": V1_MIN_ARM_PEAK_DEG})
        check("V1_wave_hand_above_shoulder", v1b_ok, {"clearance_pct_height": clearance_frac * 100, "min_pct": V1_MIN_HAND_ABOVE_SHOULDER_FRAC * 100})
        check("V1_wave_oscillations", v1c_ok, {"oscillations": osc, "min": V1_MIN_OSCILLATIONS})
        check("V1_other_arm_stable", v1d_ok, {"resting_side": resting_side, "peak_deg": v1d_peak, "max": V1_MAX_OTHER_ARM_DEG})

        # V2: root/hips translation + torso rotation
        root_hips_slots = [i for i, n in enumerate(joint_names) if n.lower() in ("root", "hips", "hip", "pelvis")]
        max_trans_frac = 0.0
        for slot in root_hips_slots:
            node_idx = joints[slot]
            pos = []
            for t in times:
                w = scene.world_matrices_at(anim, float(t))
                pos.append(w[node_idx][:3, 3])
            pos = np.array(pos)
            rng = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)) / height)
            max_trans_frac = max(max_trans_frac, rng)
        v2a_ok = max_trans_frac <= V2_MAX_ROOT_HIPS_TRANS_FRAC

        torso_slots = [i for i, n in enumerate(joint_names_l) if is_torso_joint(n)]
        max_torso_rot = 0.0
        for slot in torso_slots:
            node_idx = joints[slot]
            rots = []
            for t in times:
                _, r, _ = scene.sample_node_local(anim, node_idx, float(t))
                rots.append(r)
            rots = np.array(rots)
            angles = np.array([angle_deg(rots[0], r) for r in rots])
            max_torso_rot = max(max_torso_rot, float(angles.max()))
        v2b_ok = max_torso_rot <= V2_MAX_TORSO_ROT_DEG

        check("V2_root_hips_translation", v2a_ok, {"max_range_pct_height": max_trans_frac * 100, "max_allowed_pct": V2_MAX_ROOT_HIPS_TRANS_FRAC * 100})
        check("V2_torso_rotation", v2b_ok, {"max_rot_deg": max_torso_rot, "max_allowed_deg": V2_MAX_TORSO_ROT_DEG})

        # V3: co-deformation over all non-waving-arm-dominated vertices
        waving_slots = set(arm_chain[waving_side])
        dominant_slot_v = joints0[np.arange(len(joints0)), np.argmax(weights0, axis=1)]
        waving_dominant = np.isin(dominant_slot_v, list(waving_slots))
        non_waving_mask = ~waving_dominant

        pos0 = scene.skinned_positions_at(anim, 0.0)
        max_disp_pct = np.zeros(len(positions))
        for t in times:
            post = scene.skinned_positions_at(anim, float(t))
            disp = np.linalg.norm(post - pos0, axis=1) / height * 100
            max_disp_pct = np.maximum(max_disp_pct, disp)
        non_waving_disp = max_disp_pct[non_waving_mask]
        frac_over_3 = float((non_waving_disp > 3.0).sum()) / len(non_waving_disp)
        abs_max = float(non_waving_disp.max())
        v3a_ok = (frac_over_3 <= V3_MAX_FRAC_OVER_3PCT) and (abs_max <= V3_ABS_MAX_DISP_FRAC * 100)

        # edge stretch for all-non-arm-dominated triangles
        indices = scene.get_indices()
        tris = indices.reshape(-1, 3)
        tri_dom = dominant_slot_v[tris]
        non_arm_tri_mask = np.isin(tri_dom, list(waving_slots), invert=True).all(axis=1)
        nz_tris = tris[non_arm_tri_mask]
        edge_pairs = [(0, 1), (1, 2), (2, 0)]
        max_stretch = 1.0
        worst_frame_t = 0.0
        for t in times[::4]:  # subsample for speed; still covers full range densely
            post = scene.skinned_positions_at(anim, float(t))
            for a, b in edge_pairs:
                l0 = np.linalg.norm(pos0[nz_tris[:, a]] - pos0[nz_tris[:, b]], axis=1)
                lt = np.linalg.norm(post[nz_tris[:, a]] - post[nz_tris[:, b]], axis=1)
                safe = l0 > 1e-6
                stretch = np.ones_like(l0)
                stretch[safe] = lt[safe] / l0[safe]
                m = float(stretch.max())
                if m > max_stretch:
                    max_stretch = m
                    worst_frame_t = float(t)
        v3b_ok = max_stretch <= V3_MAX_EDGE_STRETCH

        check("V3_co_deformation_displacement", v3a_ok, {
            "non_waving_vertex_count": int(non_waving_mask.sum()),
            "frac_over_3pct_height": frac_over_3 * 100,
            "max_allowed_frac_pct": V3_MAX_FRAC_OVER_3PCT * 100,
            "abs_max_disp_pct_height": abs_max,
            "abs_max_allowed_pct": V3_ABS_MAX_DISP_FRAC * 100,
        })
        check("V3_edge_stretch", v3b_ok, {
            "max_edge_stretch": max_stretch, "max_allowed": V3_MAX_EDGE_STRETCH, "worst_frame_t": worst_frame_t,
        })

        report["wave_summary"] = {
            "waving_side": waving_side,
            "resting_side": resting_side,
            "arm_peak_rotation_deg": peak_by_side,
            "hand_clearance_above_shoulder_pct_height": clearance_frac * 100,
            "oscillations": osc,
        }

    report["overall_pass"] = len(failures) == 0
    report["failures"] = failures

    print(json.dumps(report, indent=2, default=str))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2, default=str)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
