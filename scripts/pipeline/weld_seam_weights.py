#!/usr/bin/env python3
"""
scripts/pipeline/weld_seam_weights.py

FIX ROUND 4 (t1) skin-weight rebuild -- part (c) of the work order's SCOPE
item 1: "UV-seam duplicates: weld-by-position groups (co-located within
1e-5) must carry IDENTICAL JOINTS/WEIGHTS; average them."

N1 found 190 co-located UV-seam duplicate vertex groups carrying
DIFFERENT skin weights, cracking apart up to 5.0% h (wave) / 3.4% h
(idle) -- every texture-chart boundary in this mesh duplicates a vertex
(same 3D position, different TEXCOORD_0/NORMAL, hence a different index),
and nothing in the prior pipeline ever enforced that duplicates of the
SAME point carry the SAME skin weights, so two mesh-adjacent triangles on
opposite sides of a UV seam could legitimately end up skinned to
different joints and visibly split apart under animation even though
they share every vertex position.

This is the LAST step of the round-4 rebuild (run after
rebuild_face_weights.py and rebuild_arm_body_weights.py): both those
passes smooth per-region weight FIELDS over mesh adjacency, but adjacency
built from triangle indices alone does not connect a seam-duplicate pair
to each other directly (they are only "the same point," not mesh
neighbors), so smoothing can still leave a weld group's members with
slightly different results even when they started identical. Welding
LAST guarantees the final committed asset has zero such groups,
regardless of what happened upstream.

For every weld group (>=2 vertices co-located within 1e-5):
  1. Build the union of (joint slot -> weight) across all members,
     summing each member's contribution and dividing by the group size
     (a true average, not just "first member wins").
  2. Cap to the largest 4 joints (mesh_graph.pack_top4), renormalize.
  3. Assign the IDENTICAL (JOINTS_0, WEIGHTS_0) row to every member of
     the group.

FINAL V4 SAFETY NET: averaging two group members can, in principle,
recombine a vertex the arm-body/face passes had already safety-clamped
under V4's 0.2 arm-weight bound with an UNCLAMPED seam-duplicate on a
different UV island that was never inside either pass's band -- the
average of "safe" and "unsafe" is not guaranteed safe. So after welding,
this script re-scans the WHOLE mesh (not just welded groups) and reapplies
the same clamp (arm weight capped back under a safety margin, excess
handed to the dominant head/neck joint) to any head/neck-dominant vertex
that still exceeds it -- a final, unconditional guarantee independent of
which upstream script combination produced the input.

Usage:
    python3 weld_seam_weights.py <in.glb> <out.glb> [--report report.json]
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene
from mesh_graph import find_weld_groups, pack_top4

EPS = 1e-9
V4_HARD_MAX = 0.2
V4_SAFETY_MARGIN = 0.15
HEAD_NECK_NAMES = {"Head", "Neck", "Neck1"}
ARM_NAME_TOKENS = ("arm", "hand", "shoulder", "clavicle", "elbow", "wrist", "paw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("output_glb")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.input_glb)
    positions = scene.get_positions()
    joints0, weights0 = scene.get_skin_weights()
    joints0 = joints0.astype(np.int64).copy()
    weights0 = weights0.astype(np.float64).copy()
    N = positions.shape[0]

    weld_groups = find_weld_groups(positions)

    def group_weight_dict(members):
        pool = {}
        for vi in members:
            for k in range(4):
                j = int(joints0[vi, k])
                w = float(weights0[vi, k])
                if w > 0:
                    pool[j] = pool.get(j, 0.0) + w
        return {j: w / len(members) for j, w in pool.items()}

    n_already_identical = 0
    n_welded = 0
    max_weight_delta_before = 0.0

    for grp in weld_groups:
        rows = [(tuple(int(x) for x in joints0[vi]), tuple(round(float(x), 6) for x in weights0[vi]))
                for vi in grp]
        if len(set(rows)) == 1:
            n_already_identical += 1
            continue

        # measure how different they were, for the report (max, over all
        # pairs in the group, of the per-joint weight vector L1 distance
        # expressed on the UNION of joints each carries)
        pool_per_member = []
        for vi in grp:
            d = {}
            for k in range(4):
                j = int(joints0[vi, k]); w = float(weights0[vi, k])
                if w > 0:
                    d[j] = d.get(j, 0.0) + w
            pool_per_member.append(d)
        for i in range(len(pool_per_member)):
            for j in range(i + 1, len(pool_per_member)):
                keys = set(pool_per_member[i]) | set(pool_per_member[j])
                delta = sum(abs(pool_per_member[i].get(k, 0.0) - pool_per_member[j].get(k, 0.0)) for k in keys)
                max_weight_delta_before = max(max_weight_delta_before, delta)

        combined = group_weight_dict(grp)
        new_j, new_w = pack_top4(combined)
        for vi in grp:
            joints0[vi] = new_j
            weights0[vi] = new_w
        n_welded += 1

    sums = weights0.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > 1e-4)[0]
    if len(bad):
        raise SystemExit(f"internal error: {len(bad)} vertex rows do not sum to 1 (first: {bad[:5]})")

    # ---- final, unconditional V4 safety net (see module docstring) ----
    g = scene.g
    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]
    name_to_slot = {n: i for i, n in enumerate(joint_names)}
    head_neck_slots = set(name_to_slot[n] for n in HEAD_NECK_NAMES if n in name_to_slot)
    arm_slots = set(i for i, n in enumerate(joint_names) if any(tok in n.lower() for tok in ARM_NAME_TOKENS))

    dominant = joints0[np.arange(N), np.argmax(weights0, axis=1)]
    head_dom_mask = np.isin(dominant, list(head_neck_slots))
    arm_w = np.zeros(N)
    for k in range(4):
        arm_w += np.where(np.isin(joints0[:, k], list(arm_slots)), weights0[:, k], 0.0)
    violation_idxs = np.where(head_dom_mask & (arm_w > V4_HARD_MAX))[0]

    n_final_clamped = 0
    for vi in violation_idxs:
        pool = {}
        for k in range(4):
            j = int(joints0[vi, k]); w = float(weights0[vi, k])
            if w > 0:
                pool[j] = pool.get(j, 0.0) + w
        dom_slot = int(dominant[vi])
        cur_arm_w = sum(w for j, w in pool.items() if j in arm_slots)
        if cur_arm_w > V4_SAFETY_MARGIN:
            scale = V4_SAFETY_MARGIN / cur_arm_w
            excess = 0.0
            for j in list(pool.keys()):
                if j in arm_slots:
                    reduced = pool[j] * scale
                    excess += pool[j] - reduced
                    pool[j] = reduced
            pool[dom_slot] = pool.get(dom_slot, 0.0) + excess
            new_j, new_w = pack_top4(pool)
            joints0[vi] = new_j
            weights0[vi] = new_w
            n_final_clamped += 1

    if n_final_clamped:
        sums = weights0.sum(axis=1)
        bad = np.where(np.abs(sums - 1.0) > 1e-4)[0]
        if len(bad):
            raise SystemExit(f"internal error after V4 clamp: {len(bad)} rows do not sum to 1")

    report = {
        "vertex_count": N,
        "weld_groups_total": len(weld_groups),
        "weld_groups_already_identical": n_already_identical,
        "weld_groups_welded_this_pass": n_welded,
        "max_per_member_weight_l1_delta_before_weld": max_weight_delta_before,
        "final_v4_safety_violations_found": int(len(violation_idxs)),
        "final_v4_safety_clamped": n_final_clamped,
    }
    print(f"weld groups: {len(weld_groups)} total, {n_already_identical} already identical, "
          f"{n_welded} welded (averaged) this pass")
    print(f"max per-member weight L1 delta seen before welding: {max_weight_delta_before:.4f}")
    print(f"final V4 safety net: {len(violation_idxs)} violation(s) found post-weld, "
          f"{n_final_clamped} clamped")

    scene.write_skin_weights(joints0, weights0)
    scene.g.save(args.output_glb)
    print(f"wrote {args.output_glb}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
