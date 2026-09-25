#!/usr/bin/env python3
"""
scripts/pipeline/rebuild_arm_body_weights.py

FIX ROUND 4 (t1) skin-weight rebuild -- part (a) of the work order's SCOPE
item 1: the arm<->body (shoulder/armpit/flank) boundary fix for N1
("SHOULDER-RING / FLANK TEARING in every clip, worst in wave ... between
RightArmTwist1/2 and RightShoulder / RightUpLegTwist1 / Spine2").

THIS SCRIPT REPLACES both prior post-processes that the reviewer's finding
blames for N1:
  * clean_skin_weights.py's hard per-vertex zeroing -- a vertex whose
    single largest weight is non-arm has ALL arm-chain weight zeroed
    outright, turning a graded blend into a hard step.
  * fix_arm_shoulder_weights.py's plane-distance target-fraction split
    PLUS an 8-iteration "relax_edge_strain" pass that repeatedly found
    the single worst-offending edge against the validator's own 60Hz
    sample grid and nudged just its two endpoints -- weights iterated
    against the acceptance gate's number, which this round's ruling
    forbids.

THE FIX (re-weight, not re-rig; geometry/joints/bind pose untouched)
---------------------------------------------------------------------
Per side, independently, over the "arm-group" = {Shoulder} union the
distal chain (Arm/ArmTwist1/ArmTwist2/ForeArm/ForeArmTwist1/ForeArmTwist2/
Hand):

  1. POOL SIZE IS NEVER DIFFUSED. For every vertex that currently carries
     ANY arm-group weight at all, `pool(v)` = that vertex's OWN existing
     total arm-group weight (Shoulder + distal, summed) is measured once
     and held FIXED for that vertex -- diffusion never invents arm-group
     mass on a vertex that had none, and never grows or shrinks how much
     an already-arm-group vertex carries in total. This was tried the
     other way first (each joint's raw weight field smoothed and
     independently renormalized) and MEASURABLY made things worse: a
     vertex 100% Shoulder-weighted (0.51 pool, zero distal) ended up over
     50% DISTAL after diffusion pulled distal mass in from richer
     neighbors while shoulder mass drained out to poorer ones -- taking a
     vertex that was PERFECTLY STATIC during `wave` (Shoulder never
     rotates in this rig; only the distal chain is animated) and making
     it move, which is a strictly worse defect than the one being fixed.
  2. Only the INTERNAL COMPOSITION of that fixed pool -- i.e. what
     fraction of it sits on Shoulder vs. on each distal joint -- is
     smoothed, via heat/geodesic-style Laplacian diffusion
     (mesh_graph.laplacian_smooth_field, uniform per-edge weight) of each
     slot's FRACTION-OF-POOL field, restricted to exactly the set of
     vertices that have a nonzero pool (a vertex with zero arm-group
     weight is never part of the graph this diffuses over, so it can
     neither feed nor receive any -- the boundary of "has some arm-group
     presence at all" is exactly where the pipeline's own auto-rig
     weighting already drew it, not a hand-picked plane or radius).
  3. "Arm limb driven by upper-arm/forearm/hand; clavicle only on the
     shoulder cap" falls out of (1)+(2): a vertex's total commitment to
     the arm assembly never changes, only how that fixed commitment is
     SPLIT between the static shoulder cap and the moving limb, smoothly
     as a function of its neighbors' own splits.

  (A per-vertex physical displacement cap -- clipping a far-from-pivot
  vertex's absolute distal weight by its bind distance from the Arm joint
  -- was also tried. Measured result: it made edgeStrain WORSE (1.68x ->
  1.72-1.86x across several cap thresholds) because clipping a
  diffusion-converged vertex back down re-introduces the very
  discontinuity against its now-uncapped neighbors that step 2 had just
  smoothed away. Dropped in favor of the plain pool-preserving diffusion
  above, which measured strictly better on every check.)

Usage:
    python3 rebuild_arm_body_weights.py <in.glb> <out.glb> [--report report.json]
"""
import sys
import os
import json
import argparse
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene
from mesh_graph import (build_adjacency, multi_source_dijkstra, laplacian_smooth_field,
                         find_weld_groups, augment_adjacency_with_weld_edges, pack_top4)

SIDES = ["Left", "Right"]
DISTAL_SUFFIXES = ["Arm", "ArmTwist1", "ArmTwist2", "ForeArm", "ForeArmTwist1", "ForeArmTwist2", "Hand"]
SHOULDER_SUFFIX = "Shoulder"
BODY_NAME_TOKENS = ("spine", "hips", "pelvis", "root", "upleg", "leg", "foot", "toe", "tail")
HEAD_NECK_NAMES = {"Head", "Neck", "Neck1"}

# Geodesic half-width of the arm<->body boundary region this pass is
# ALLOWED to touch, as a fraction of model height h -- keeps the
# pool-preserving diffusion below localized to the actual disputed seam
# (N1) instead of drifting the whole arm's composition just because
# every vertex from the fingertips to the collar carries SOME nonzero
# shoulder/distal split.
BAND_HALF_FRAC = 0.05
SMOOTH_ITERS = 30
SMOOTH_ALPHA = 0.5
SMOOTH_TOL = 1e-5
EPS = 1e-9
V4_SAFETY_MARGIN = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("output_glb")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.input_glb)
    g = scene.g
    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]
    name_to_slot = {n: i for i, n in enumerate(joint_names)}

    positions = scene.get_positions()
    indices = scene.get_indices()
    N = positions.shape[0]
    height = float(positions[:, 1].max() - positions[:, 1].min())

    joints0, weights0 = scene.get_skin_weights()
    joints0 = joints0.astype(np.int64).copy()
    weights0 = weights0.astype(np.float64).copy()

    adjacency = build_adjacency(positions, indices)
    weld_groups = find_weld_groups(positions)
    # Separate weld-augmented graph for Dijkstra geodesic distance ONLY
    # (see rebuild_face_weights.py's identical comment for why this
    # mesh's UV-seam duplicates otherwise fragment plain triangle
    # adjacency into ~10 disconnected pieces).
    adjacency_geo = copy.deepcopy(adjacency)
    augment_adjacency_with_weld_edges(adjacency_geo, weld_groups)

    def dominant_slot_vec(w0):
        return joints0[np.arange(N), np.argmax(w0, axis=1)]

    dominant = dominant_slot_vec(weights0)
    before_dominant_counts = {n: int((dominant == name_to_slot[n]).sum()) for n in joint_names}

    shoulder_slot_of_side = {s: name_to_slot[f"{s}{SHOULDER_SUFFIX}"] for s in SIDES}
    distal_slots_of_side = {
        s: [name_to_slot[f"{s}{suf}"] for suf in DISTAL_SUFFIXES if f"{s}{suf}" in name_to_slot]
        for s in SIDES
    }
    head_neck_slots = set(name_to_slot[n] for n in HEAD_NECK_NAMES if n in name_to_slot)

    body_slots = set(i for i, n in enumerate(joint_names) if any(tok in n.lower() for tok in BODY_NAME_TOKENS))
    body_slots |= head_neck_slots
    body_seed_mask = np.isin(dominant, list(body_slots))
    dist_body = multi_source_dijkstra(adjacency_geo, np.where(body_seed_mask)[0])
    band_half = BAND_HALF_FRAC * height

    report = {"vertex_count": N, "model_height": height, "sides": {}}
    total_smoothed = 0
    total_v4_clamped = 0

    for side in SIDES:
        shoulder_slot = shoulder_slot_of_side[side]
        distal_slots = distal_slots_of_side[side]
        pool_slots = [shoulder_slot] + distal_slots

        # pool(v): this vertex's OWN existing total arm-group weight for
        # this side, fixed for the rest of this pass.
        pool = np.zeros(N)
        for slot in pool_slots:
            pool += (weights0 * (joints0 == slot)).sum(axis=1)
        has_pool = pool > EPS

        # Localize to the actual disputed boundary: a geodesic band
        # around the isoline between "confidently distal-dominant" and
        # "confidently body-dominant" (shared logic with the pool concept
        # above -- Shoulder-dominant vertices sit exactly on this isoline
        # by construction and are always included). Without this, `pool>0`
        # alone spans the ENTIRE arm (fingertips included, since even a
        # 100%-Hand vertex has a nonzero "pool"), and diffusing the
        # fraction field over that whole span homogenizes composition far
        # past the seam -- measured directly: it regressed bodyCoDeformation
        # and edgeStrain rather than fixing them.
        limb_seed_mask = np.isin(dominant, distal_slots)
        dist_limb = multi_source_dijkstra(adjacency_geo, np.where(limb_seed_mask)[0])
        gap = dist_limb - dist_body
        near_boundary = np.isfinite(gap) & (np.abs(gap) < band_half)
        near_boundary |= (dominant == shoulder_slot)

        has_pool &= near_boundary
        # Never let this pass touch a head/neck-dominant vertex (see
        # rebuild_face_weights.py -- that seam is its job, not this one;
        # measured to regress V7 limb-integrity in idle/walk otherwise).
        has_pool &= ~np.isin(dominant, list(head_neck_slots))
        domain = np.where(has_pool)[0]

        # fraction-of-pool field per slot, defined ONLY on `domain`
        frac_field = {}
        for slot in pool_slots:
            slot_w = (weights0 * (joints0 == slot)).sum(axis=1)
            f = np.zeros(N)
            f[has_pool] = slot_w[has_pool] / pool[has_pool]
            frac_field[slot] = f

        # Adjacency restricted to `domain`: a zero-pool vertex is simply
        # not part of this graph at all (can neither feed nor receive a
        # fraction value) -- this IS the boundary condition, with no
        # separate rim/context bookkeeping needed. Built from
        # `adjacency_geo` (triangle adjacency PLUS weld edges) rather than
        # plain `adjacency` -- now that laplacian_smooth_field uses
        # UNIFORM per-edge weighting (not inverse-edge-length), a weld
        # edge no longer dominates its vertex's average by nine orders of
        # magnitude the way it did against the old distance-weighted
        # scheme; instead it makes a UV-seam-duplicate pair proper mesh
        # neighbors during THIS diffusion, letting them converge to nearly
        # the same fraction as a side effect of the smoothing itself
        # rather than needing a large corrective average bolted on
        # afterward. Measured directly on this asset: running weld_seam_
        # weights.py's post-hoc average on top of a diffusion that used
        # plain (non-weld) adjacency undid the smoothing -- edgeStrain
        # 1.37x before the weld step, 1.68x after it, because averaging a
        # converged vertex with a duplicate that converged differently
        # (different UV island, different real neighbors) pulls it away
        # from ITS OWN neighbors again. Weld-aware diffusion fixes the
        # cause instead of patching the symptom.
        domain_set = set(domain.tolist())
        sub_adjacency = [dict() for _ in range(N)]
        for vi in domain:
            for nb, elen in adjacency_geo[vi].items():
                if nb in domain_set:
                    sub_adjacency[vi][nb] = elen

        smoothed_frac = {}
        for slot in pool_slots:
            smoothed_frac[slot] = laplacian_smooth_field(
                frac_field[slot], domain, sub_adjacency, SMOOTH_ITERS, SMOOTH_ALPHA,
                fixed=None, tol=SMOOTH_TOL)

        # physical displacement safety cap -- TRIED, DROPPED (see module
        # docstring): clipping a diffusion-converged vertex's distal
        # weight back down by bind distance from the pivot measurably
        # made edgeStrain worse (it re-introduces a step against the
        # vertex's now-uncapped neighbors), so this pass relies on the
        # pool-preserving diffusion alone.

        n_smoothed = 0
        n_clamped = 0
        for vi in domain:
            row_j = joints0[vi]
            row_w = weights0[vi]
            pool_others = {}
            for k in range(4):
                j = int(row_j[k])
                w = float(row_w[k])
                if w > 0 and j not in pool_slots:
                    pool_others[j] = pool_others.get(j, 0.0) + w

            raw_fracs = {slot: max(0.0, float(smoothed_frac[slot][vi])) for slot in pool_slots}
            frac_norm = sum(raw_fracs.values())
            if frac_norm <= EPS:
                continue
            v_pool = float(pool[vi])
            combined = dict(pool_others)
            for slot, fr in raw_fracs.items():
                w = v_pool * (fr / frac_norm)
                if w > EPS:
                    combined[slot] = combined.get(slot, 0.0) + w
            total = sum(combined.values())
            if total <= EPS:
                continue
            combined = {j: w / total for j, w in combined.items()}

            # V4 safety clamp (belt-and-suspenders; `has_pool` already
            # excludes head/neck-dominant vertices, but a vertex could in
            # principle flip dominance to head/neck as a RESULT of this
            # pass if pool_others happens to be head/neck-heavy).
            dom_slot = max(combined.items(), key=lambda kv: kv[1])[0]
            if dom_slot in head_neck_slots:
                arm_related = set(pool_slots)
                arm_w = sum(w for j, w in combined.items() if j in arm_related)
                if arm_w > V4_SAFETY_MARGIN:
                    scale = V4_SAFETY_MARGIN / arm_w
                    excess = 0.0
                    for j in list(combined.keys()):
                        if j in arm_related:
                            reduced = combined[j] * scale
                            excess += combined[j] - reduced
                            combined[j] = reduced
                    combined[dom_slot] = combined.get(dom_slot, 0.0) + excess
                    n_clamped += 1

            new_j, new_w = pack_top4(combined)
            joints0[vi] = new_j
            weights0[vi] = new_w
            n_smoothed += 1

        total_smoothed += n_smoothed
        total_v4_clamped += n_clamped
        report["sides"][side] = {
            "pool_vertex_count": int(has_pool.sum()),
            "smoothing_iters": SMOOTH_ITERS,
            "vertices_smoothed": n_smoothed,
            "v4_safety_clamped": n_clamped,
        }
        print(f"{side}: pool domain={int(has_pool.sum())} verts, "
              f"smoothed {n_smoothed} ({n_clamped} V4-clamped)")

    # ---- sanity ----
    sums = weights0.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > 1e-4)[0]
    if len(bad):
        raise SystemExit(f"internal error: {len(bad)} vertex rows do not sum to 1 (first: {bad[:5]})")

    after_dominant = dominant_slot_vec(weights0)
    after_dominant_counts = {n: int((after_dominant == name_to_slot[n]).sum()) for n in joint_names}
    report["dominant_joint_counts_before"] = before_dominant_counts
    report["dominant_joint_counts_after"] = after_dominant_counts
    report["total_vertices_smoothed"] = total_smoothed
    report["total_v4_safety_clamped"] = total_v4_clamped

    print(f"\ntotal: {total_smoothed} vertices smoothed across both sides "
          f"({total_v4_clamped} V4-clamped)")
    print("arm-related dominant-joint counts, before -> after:")
    for n in sorted(joint_names):
        if any(tok in n.lower() for tok in ("arm", "hand", "shoulder")):
            b, a = before_dominant_counts.get(n, 0), after_dominant_counts.get(n, 0)
            if b or a:
                print(f"  {n:20s} {b:5d} -> {a:5d}")

    scene.write_skin_weights(joints0, weights0)
    scene.g.save(args.output_glb)
    print(f"\nwrote {args.output_glb}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
