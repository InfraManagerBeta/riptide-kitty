#!/usr/bin/env python3
"""
scripts/pipeline/own_edge_check.py

Round-4 (t1) OWN verification script -- SCOPE item 2 of the work order:
"Verify with your own ALL-EDGES check. In every clip, vs bind and vs clip
start: report the max edge ratio and the count > 1.5x. Also seam-group
crack max (% h) per clip, and arm-dominated vertex count above the neck."

Deliberately INDEPENDENT of scripts/validate-kitty.mjs (out of this
round's touch-scope) and of the reviewer's own review tooling -- this is
the asset's own accompanying check, over MESH TRIANGLE EDGES the real
validator's V3 does NOT look at (V3 restricts its edge-strain check to
triangles whose three vertices are ALL non-waving-arm-dominated, which is
exactly why N1's arm<->body boundary tearing slipped past a PASSing V3:
an edge with one arm-dominated endpoint and one body-dominated endpoint
is invisible to that filter). This script checks EVERY triangle edge,
including the arm<->body boundary, against TWO references per clip:

  * "vs bind"       -- the raw (un-skinned) POSITION accessor, i.e. the
                        rest/bind pose shape.
  * "vs clip start"  -- the clip's own frame-0 skinned pose (what
                        scripts/validate-kitty.mjs's V3 also uses).

...plus:
  * seam-group crack -- for every co-located (weld) vertex group (see
    mesh_graph.find_weld_groups), the max pairwise distance between its
    members' skinned positions at any sampled frame of the clip, as a
    fraction of model height. N1 reported these open up to 5.0% h in
    wave / 3.4% h in idle when a weld group's members carried DIFFERENT
    skin weights (scope item 1c's fix).
  * arm-dominated vertex count above the neck joint's bind height --
    pose-independent (a property of the skin weights alone, not of any
    one clip's animation), reported identically on every clip's row for
    the table the work order asks for; this is N2's own headline number
    (RightShoulder/LeftShoulder dominant above the neck).

Usage:
    python3 own_edge_check.py <kitty.glb> [--out-json report.json]
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene
from mesh_graph import find_weld_groups

ARM_NAME_TOKENS = ("arm", "hand", "shoulder", "clavicle", "elbow", "wrist", "paw")
HEAD_NECK_TOKENS = ("head", "neck")


def fast_skinned_positions(positions, joints0, weights0, joint_node_idx, ibm, world):
    """Same math as GLTFScene.skinned_positions_at, but takes pre-loaded
    POSITION/JOINTS_0/WEIGHTS_0 arrays and a pre-computed `world` dict
    (node_idx -> 4x4) instead of re-reading raw accessor bytes on every
    call -- re-parsing 13,186 vertices' worth of struct.unpack per SAMPLED
    FRAME (thousands of frames at 60Hz) is the difference between seconds
    and hours here."""
    skin_mats = np.stack([world[joint_node_idx[j]] @ ibm[j] for j in range(len(joint_node_idx))])
    N = positions.shape[0]
    pos_h = np.concatenate([positions, np.ones((N, 1))], axis=1)
    out = np.zeros((N, 4))
    for k in range(4):
        jslot = joints0[:, k]
        w = weights0[:, k:k + 1]
        m = skin_mats[jslot]
        transformed = np.einsum("nij,nj->ni", m, pos_h)
        out += w * transformed
    return out[:, :3]


def sample_times(duration, rate=60, max_samples=3601):
    n = min(max_samples, max(2, int(np.ceil(duration * rate)) + 1))
    return np.linspace(0, duration, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_path")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--clips", nargs="*", default=None, help="restrict to these clip names")
    args = ap.parse_args()

    scene = GLTFScene(args.glb_path)
    g = scene.g
    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]

    positions = scene.get_positions()
    indices = scene.get_indices()
    tris = indices.reshape(-1, 3)
    N = positions.shape[0]
    height = float(positions[:, 1].max() - positions[:, 1].min())

    joints0, weights0 = scene.get_skin_weights()

    # unique undirected edges from the triangle list
    edge_set = set()
    for a, b, c in tris:
        a, b, c = int(a), int(b), int(c)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add((u, v) if u < v else (v, u))
    edges = np.array(sorted(edge_set), dtype=np.int64)
    ea, eb = edges[:, 0], edges[:, 1]

    weld_groups = find_weld_groups(positions)

    # Flatten every weld group into its pairwise (i,j) combinations, each
    # tagged with a group id, so a single frame's seam-crack distances can
    # be computed with one vectorized call instead of a Python loop over
    # thousands of small groups per sampled frame.
    pair_a, pair_b, pair_group = [], [], []
    for gi, grp in enumerate(weld_groups):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                pair_a.append(grp[i])
                pair_b.append(grp[j])
                pair_group.append(gi)
    pair_a = np.array(pair_a, dtype=np.int64)
    pair_b = np.array(pair_b, dtype=np.int64)
    pair_group = np.array(pair_group, dtype=np.int64)
    n_groups = len(weld_groups)

    dom = joints0[np.arange(N), np.argmax(weights0, axis=1)]
    arm_slots = set(i for i, n in enumerate(joint_names)
                    if any(tok in n.lower() for tok in ARM_NAME_TOKENS))
    cache = {}
    neck_slot = next(i for i, n in enumerate(joint_names) if n == "Neck")
    neck_y = float(scene.node_bind_world(joints[neck_slot], cache)[1, 3])
    arm_dom_above_neck = int(np.sum(np.isin(dom, list(arm_slots)) & (positions[:, 1] > neck_y)))

    bind_len = np.linalg.norm(positions[ea] - positions[eb], axis=1)
    bind_safe = bind_len > 1e-9

    clip_names = args.clips or [a.name for a in g.animations]
    report = {
        "file": args.glb_path,
        "model_height": height,
        "arm_dominated_vertices_above_neck_joint": arm_dom_above_neck,
        "neck_joint_y": neck_y,
        "weld_groups_total": len(weld_groups),
        "clips": {},
    }

    for clip_name in clip_names:
        anim = next((a for a in g.animations if a.name == clip_name), None)
        if anim is None:
            continue
        dur = scene.animation_duration(anim)
        times = sample_times(dur)

        pos0 = None
        max_ratio_vs_bind = 1.0
        max_ratio_vs_start = 1.0
        count_over_vs_bind = 0
        count_over_vs_start = 0
        worst_edge_vs_bind = None
        worst_edge_vs_start = None
        seam_crack_max_pct = 0.0
        worst_seam_group = None

        for ti, t in enumerate(times):
            world = scene.world_matrices_at(anim, float(t))
            post = fast_skinned_positions(positions, joints0, weights0, joints, ibm, world)
            if pos0 is None:
                pos0 = post.copy()
                start_len = np.linalg.norm(pos0[ea] - pos0[eb], axis=1)
                start_safe = start_len > 1e-9

            lt = np.linalg.norm(post[ea] - post[eb], axis=1)

            ratio_bind = np.ones_like(bind_len)
            ratio_bind[bind_safe] = lt[bind_safe] / bind_len[bind_safe]
            idx = int(np.argmax(ratio_bind))
            if ratio_bind[idx] > max_ratio_vs_bind:
                max_ratio_vs_bind = float(ratio_bind[idx])
                worst_edge_vs_bind = (int(ea[idx]), int(eb[idx]), float(t))
            count_over_vs_bind = max(count_over_vs_bind, int((ratio_bind > 1.5).sum()))

            ratio_start = np.ones_like(start_len)
            ratio_start[start_safe] = lt[start_safe] / start_len[start_safe]
            idx2 = int(np.argmax(ratio_start))
            if ratio_start[idx2] > max_ratio_vs_start:
                max_ratio_vs_start = float(ratio_start[idx2])
                worst_edge_vs_start = (int(ea[idx2]), int(eb[idx2]), float(t))
            count_over_vs_start = max(count_over_vs_start, int((ratio_start > 1.5).sum()))

            if n_groups:
                pd = np.linalg.norm(post[pair_a] - post[pair_b], axis=1)
                group_max = np.zeros(n_groups)
                np.maximum.at(group_max, pair_group, pd)
                gi = int(np.argmax(group_max))
                m = float(group_max[gi]) / height * 100
                if m > seam_crack_max_pct:
                    seam_crack_max_pct = m
                    worst_seam_group = weld_groups[gi]

        def edge_label(e):
            if e is None:
                return None
            a, b, t = e
            return {"vertex_a": a, "vertex_b": b, "at_t": t,
                    "joint_a": joint_names[dom[a]] if dom[a] < len(joint_names) else None,
                    "joint_b": joint_names[dom[b]] if dom[b] < len(joint_names) else None}

        report["clips"][clip_name] = {
            "duration_s": dur,
            "samples": len(times),
            "all_edges_total": len(edges),
            "vs_bind": {
                "max_ratio": max_ratio_vs_bind,
                "count_over_1.5x": count_over_vs_bind,
                "worst_edge": edge_label(worst_edge_vs_bind),
            },
            "vs_clip_start": {
                "max_ratio": max_ratio_vs_start,
                "count_over_1.5x": count_over_vs_start,
                "worst_edge": edge_label(worst_edge_vs_start),
            },
            "seam_group_crack_max_pct_height": seam_crack_max_pct,
            "arm_dominated_vertices_above_neck_joint": arm_dom_above_neck,
        }
        print(f"[{clip_name}] dur={dur:.3f}s samples={len(times)} "
              f"vs_bind: max={max_ratio_vs_bind:.3f}x count>1.5x={count_over_vs_bind} | "
              f"vs_start: max={max_ratio_vs_start:.3f}x count>1.5x={count_over_vs_start} | "
              f"seam_crack_max={seam_crack_max_pct:.3f}%h")

    print(f"\narm-dominated vertices above the neck joint (y > {neck_y:.4f}): {arm_dom_above_neck}")
    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
