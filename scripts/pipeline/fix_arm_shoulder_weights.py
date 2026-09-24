#!/usr/bin/env python3
"""
scripts/pipeline/fix_arm_shoulder_weights.py

FIX ROUND 2 (t1) post-process -- documented scripted skin-weight
re-weight, the fallback the manager's ruling for this round accepts
("a SCRIPTED, documented skin-weight post-process of the generated rig
is an acceptable fallback, committed and recorded in provenance").

THE DEFECT
----------
The manager's numpy LBS audit of round-1's asset (809ea58 merged onto the
upgraded-validator lane head 0ddda23) found that BOTH R_Clavicle and
L_Clavicle (renamed RightShoulder / LeftShoulder by
scripts/pipeline/joint_rename_map.json) own the skin over almost the
WHOLE upper arm, and part of the forearm, on both sides -- straight from
Tripo's auto-rig weighting, untouched by clean_skin_weights.py (which
only removed CROSS-contamination between the left/right arm groups and
non-arm regions; it treated "clavicle + upper arm" as a single group and
so had no way to see this defect). Concretely (dominant-joint vertex
counts, this asset before this script runs):

    RightShoulder 1610 (y 0.24-0.68)   LeftShoulder 374
    RightArm         0                  LeftArm         0
    RightArmTwist1   0                  LeftArmTwist1  56
    RightArmTwist2   2                  LeftArmTwist2  85
    RightForeArm     0                  (LeftForeArm    0)
    RightForeArmTwist1 44               LeftForeArmTwist1 55
    RightForeArmTwist2 344              LeftForeArmTwist2 113
    RightHand        603                LeftHand       557

During `wave`, only RightArm/RightForeArm (the animated joints) rotate;
the clavicle never moves. So at the wave's peak, the paw and forearm rise
while up to ~1600 clavicle-pinned "upper arm sleeve" vertices stay put --
the arm visibly tears into a raised forearm/hand and a hanging upper-arm
sleeve joined by a stretched strip. That is exactly the validator's V3
`bodyCoDeformation` FAIL (960/12,293 non-waving-arm vertices moved
>3%h, worst 23.35%h; `edgeStrain` 5.90x) -- the validator does NOT count
shoulder/clavicle-dominated skin as part of the waving arm (documented in
scripts/validate-kitty.mjs's wavingArmCoreSet): a clavicle-owned sleeve is
body, not arm, and it must stay still.

THE FIX (re-weight, not re-rig)
--------------------------------
No geometry, joint hierarchy, or bind matrix is touched -- only the
per-vertex JOINTS_0 / WEIGHTS_0 skin weights, and only for the "arm
group" of each side (Shoulder/clavicle + Arm/Upperarm + its twists +
ForeArm + its twists + Hand):

For each side (Left, Right) independently:
  1. Bind-pose plane: passes through that side's Upperarm ("...Arm")
     joint position, its normal is the (normalized) upper-arm bone
     direction Arm -> ForeArm. This is "the plane through the upper-arm
     joint perpendicular to the upper-arm bone" the work order specifies.
  2. Every vertex carrying ANY weight on the Shoulder joint OR on the
     limb chain (Arm/ArmTwist/ForeArm/ForeArmTwist/Hand) has its signed
     distance `d` to this plane converted into a continuous
     shoulder-fraction target via a linear ramp over a blend-ring
     (RING_HALF_FRAC * upper-arm-bone-length on each side of the plane):
     1.0 (all "arm group" weight stays on the clavicle) strictly medial,
     0.0 (all of it moves onto the limb chain) strictly lateral, linear
     in between -- so the shoulder seam does not carry a hard weight
     discontinuity as a function of d.
  3. That per-vertex target is then smoothed by a few iterations of
     mesh-adjacency (Laplacian) averaging over the actual triangle graph,
     restricted to the vertices this script touches. Two mesh vertices
     can sit very close in 3D (a fraction of the blend ring's own width)
     while their signed distance `d` differs by much more than that,
     wherever the local surface is not parallel to the upper-arm bone
     (found empirically at the belly/hip crease where the low-hanging
     arm passes close to the torso) -- without this step, two adjacent,
     nearly-touching vertices could land in different weight regimes and
     produce a short, artificially high-ratio "stretched" edge even
     though neither vertex individually violates the displacement cap.
     Smoothing over real mesh adjacency (not just the 1-D plane
     coordinate) removes that, whatever its geometric cause.
  4. The vertex's pre-existing "arm group" weight (shoulder + limb
     chain, whatever its current split) is redistributed to match the
     smoothed target: the shoulder slot gets target_frac * pool; the
     rest is scaled from the vertex's own EXISTING limb-chain
     proportions (reinforcing whatever the auto-rig already associated
     it with, rather than inventing a placement). A vertex whose ONLY
     arm-group weight is on the Shoulder (no existing limb-chain share
     at all to reinforce) is left untouched -- see the "gate" comment in
     the code for why. Every other joint's weight on a touched vertex
     (torso, head, leg, tail, ...) is left untouched.
  5. Per-vertex slot surgery keeps WEIGHTS_0 a true probability (sums to
     1) and JOINTS_0/WEIGHTS_0 at exactly 4 influences: existing slots
     are reused where the target joint already appears; if inserting a
     new joint would exceed 4 influences, the smallest-weight entries
     are folded (weight redistributed proportionally) into the largest.

Usage:
    python3 fix_arm_shoulder_weights.py <in.glb> <out.glb> [--report report.json]
"""
import sys
import os
import json
import argparse
import numpy as np
from pygltflib import GLTF2

sys.path.insert(0, os.path.dirname(__file__))
from gltf_anim import GLTFScene

SIDES = ["Left", "Right"]
FULL_CHAIN_SUFFIXES = ["Arm", "ArmTwist1", "ArmTwist2", "ForeArm", "ForeArmTwist1", "ForeArmTwist2", "Hand"]
RING_HALF_FRAC = 0.10     # blend-ring half-width, as a fraction of the upper-arm bone length
SMOOTH_ITERS = 6          # mesh-adjacency smoothing iterations on the target shoulder-fraction field
SMOOTH_ALPHA = 0.5        # per-iteration blend toward the neighbor average
EPS = 1e-9


def build_adjacency(indices, n):
    adj = [set() for _ in range(n)]
    tris = indices.reshape(-1, 3)
    for a, b, c in tris:
        a, b, c = int(a), int(b), int(c)
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj


def smooth_field(values, active_idxs, adjacency, iters, alpha):
    """Laplacian-smooth `values` (dict vertex-> float) over `adjacency`,
    restricted to `active_idxs`, for `iters` iterations."""
    active_set = set(active_idxs)
    cur = dict(values)
    for _ in range(iters):
        nxt = {}
        for v in active_idxs:
            neigh = [n for n in adjacency[v] if n in active_set]
            if not neigh:
                nxt[v] = cur[v]
                continue
            avg = sum(cur[n] for n in neigh) / len(neigh)
            nxt[v] = (1 - alpha) * cur[v] + alpha * avg
        cur = nxt
    return cur


def rebuild_vertex(row_joints, row_weights, remove_slots, adds):
    """Return new (joints[4], weights[4]) for one vertex: zero every slot
    in `remove_slots`, then add the (joint -> weight) pairs in `adds`
    (merging into an existing slot for that joint if present), capping at
    4 total influences by folding overflow proportionally into the 4
    largest, and renormalizing to sum 1."""
    pairs = {}
    for k in range(4):
        j = int(row_joints[k])
        w = float(row_weights[k])
        if w <= 0:
            continue
        if j in remove_slots:
            continue
        pairs[j] = pairs.get(j, 0.0) + w
    for j, w in adds.items():
        if w > EPS:
            pairs[j] = pairs.get(j, 0.0) + w

    if len(pairs) > 4:
        items = sorted(pairs.items(), key=lambda kv: -kv[1])
        kept = items[:4]
        dropped = items[4:]
        dropped_w = sum(w for _, w in dropped)
        kept_sum = sum(w for _, w in kept)
        if kept_sum > 1e-12:
            kept = [(j, w + dropped_w * (w / kept_sum)) for j, w in kept]
        pairs = dict(kept)

    total = sum(pairs.values())
    if total <= 1e-12:
        pairs = {int(row_joints[0]): 1.0}
        total = 1.0

    new_j = [0, 0, 0, 0]
    new_w = [0.0, 0.0, 0.0, 0.0]
    for idx, (j, w) in enumerate(pairs.items()):
        new_j[idx] = j
        new_w[idx] = w / total
    return new_j, new_w


def nearest_on_polyline(p, pts):
    """Nearest point on a polyline (list of np.array points). Returns
    (dist, segment_index, t) where t in [0,1] interpolates pts[i]->pts[i+1]."""
    best = None
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        l2 = float(ab @ ab)
        t = 0.0 if l2 < 1e-12 else float(np.clip((p - a) @ ab / l2, 0.0, 1.0))
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        if best is None or dist < best[0]:
            best = (dist, i, t)
    return best


def detect_waving_side(scene, anim, joints, name_to_slot):
    """Which side's Arm joint rotates most across the clip (from its own
    frame-0 pose) -- matches the validator's own side auto-detection."""
    dur = scene.animation_duration(anim)
    times = np.linspace(0, dur, 20)
    best_side, best_peak = SIDES[0], -1.0
    for side in SIDES:
        node_idx = joints[name_to_slot[f"{side}Arm"]]
        base_r = None
        peak = 0.0
        for t in times:
            _, r, _ = scene.sample_node_local(anim, node_idx, float(t))
            if base_r is None:
                base_r = r
            dot = min(1.0, abs(float(np.dot(base_r, r))))
            ang = 2 * np.degrees(np.arccos(dot))
            peak = max(peak, ang)
        if peak > best_peak:
            best_peak, best_side = peak, side
    return best_side


def vertex_chain_state(row_j, row_w, shoulder_slot, full_chain_slots):
    """(shoulder_weight, {chain_joint: weight, ...}) for one vertex row."""
    shoulder_w = 0.0
    chain = {}
    for k in range(4):
        j = int(row_j[k]); w = float(row_w[k])
        if w <= 0:
            continue
        if j == shoulder_slot:
            shoulder_w += w
        elif j in full_chain_slots:
            chain[j] = chain.get(j, 0.0) + w
    return shoulder_w, chain


def skinned_positions_with_weights(scene, anim, t, positions, joints, ibm, joints0, weights0):
    """Same as GLTFScene.skinned_positions_at, but takes the skin's
    POSITION/JOINTS_0/WEIGHTS_0 arrays explicitly instead of re-reading
    them from the (unmodified, on-disk) glTF blob -- required here
    because this script mutates joints0/weights0 in memory and only
    writes them back to the blob at the very end."""
    world = scene.world_matrices_at(anim, t)
    skin_mats = np.stack([world[joints[j]] @ ibm[j] for j in range(len(joints))])
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


def relax_edge_strain(scene, anim, joints0, weights0, positions, indices, adjacency,
                       waving_core_slots, side_of_slot, shoulder_slot_of_side,
                       full_chain_slots_of_side, chain_pts_of_side, chain_slots_of_side,
                       skin_joints, skin_ibm, head_neck_slots,
                       max_iters=25, n_sample_times=None, target_ratio=1.5, verbose=True):
    """Targeted post-pass: repeatedly find the single worst-stretching
    "body" (non-waving-arm) mesh edge over the wave clip and equalize the
    two endpoint vertices' shoulder-vs-limb-chain weight SPLIT (their
    total arm-group weight -- i.e. how much of it counts as "moving
    limb" -- is nudged to the pair's average), so a locally fine mesh
    seam this asset's own T-junction topology puts right between a
    chain-weighted and a chain-empty vertex (see the module docstring)
    does not carry an outsized short-edge stretch ratio. Only vertices
    that are part of a currently-offending edge are ever touched -- no
    blanket dilation, no invented weight on unrelated vertices."""
    N = positions.shape[0]
    dur = scene.animation_duration(anim)
    # Match the REAL validator's own wave-clip sampling exactly (60 Hz,
    # >=2 samples, capped) -- see scripts/validate-kitty.mjs sampleTimes()
    # -- so this pass optimizes against the same frames the acceptance
    # test measures, not a coarser approximation of them.
    if n_sample_times is None:
        n_sample_times = min(3601, max(2, int(np.ceil(dur * 60)) + 1))
    sample_times = np.linspace(0, dur, n_sample_times)
    tris = indices.reshape(-1, 3)
    edge_pairs = [(0, 1), (1, 2), (2, 0)]

    touched = set()
    worst_history = []

    for it in range(max_iters):
        # Frame-0 pose is itself a function of the (evolving) weights --
        # recompute every iteration.
        pos0 = skinned_positions_with_weights(scene, anim, 0.0, positions, skin_joints, skin_ibm, joints0, weights0)
        dom = joints0[np.arange(N), np.argmax(weights0, axis=1)]
        is_body = ~np.isin(dom, list(waving_core_slots))
        tri_mask = is_body[tris].all(axis=1)
        body_tris = tris[tri_mask]
        if len(body_tris) == 0:
            break

        worst = (1.0, None, None)
        for t in sample_times:
            post = skinned_positions_with_weights(scene, anim, float(t), positions, skin_joints, skin_ibm, joints0, weights0)
            for a, b in edge_pairs:
                va = body_tris[:, a]; vb = body_tris[:, b]
                l0 = np.linalg.norm(pos0[va] - pos0[vb], axis=1)
                lt = np.linalg.norm(post[va] - post[vb], axis=1)
                safe = l0 > 1e-6
                ratio = np.ones_like(l0)
                ratio[safe] = lt[safe] / l0[safe]
                idx = int(np.argmax(ratio))
                if ratio[idx] > worst[0]:
                    worst = (float(ratio[idx]), int(va[idx]), int(vb[idx]))

        worst_ratio, wa, wb = worst
        worst_history.append(worst_ratio)
        if worst_ratio <= target_ratio or wa is None:
            break

        sides = []
        for vi in (wa, wb):
            row_j = joints0[vi]
            found = None
            for k in range(4):
                j = int(row_j[k])
                if row_j[k] in side_of_slot and weights0[vi][k] > 0:
                    found = side_of_slot[j]
                    break
            sides.append(found)
        if sides[0] is None and sides[1] is None:
            # Neither endpoint is part of an arm group at all -- not
            # something this script can address; stop rather than loop.
            break

        # Compute current (shoulder_w, chain dict, side) per endpoint,
        # falling back to the OTHER endpoint's side if one has none.
        info = {}
        for vi, sd in zip((wa, wb), sides):
            side = sd or (sides[0] if vi == wb else sides[1])
            if side is None:
                continue
            sh_w, chain = vertex_chain_state(joints0[vi], weights0[vi], shoulder_slot_of_side[side], full_chain_slots_of_side[side])
            info[vi] = {"side": side, "shoulder_w": sh_w, "chain": chain}

        if len(info) < 2:
            break

        pools = {}
        cfs = {}
        for vi, d in info.items():
            chain_w = sum(d["chain"].values())
            pool = d["shoulder_w"] + chain_w
            pools[vi] = pool
            cfs[vi] = (chain_w / pool) if pool > EPS else 0.0

        target_cf = sum(cfs.values()) / len(cfs)

        for vi, d in info.items():
            pool = pools[vi]
            if pool <= EPS:
                continue
            side = d["side"]
            shoulder_slot = shoulder_slot_of_side[side]
            target_chain_w = pool * target_cf
            target_shoulder_w = pool - target_chain_w

            adds = {}
            if target_shoulder_w > EPS:
                adds[shoulder_slot] = target_shoulder_w
            if target_chain_w > EPS:
                chain = d["chain"]
                if not chain:
                    other = wb if vi == wa else wa
                    other_chain = info.get(other, {}).get("chain") or {}
                    chain = other_chain
                if chain:
                    norm = sum(chain.values())
                    for j, w in chain.items():
                        adds[j] = adds.get(j, 0.0) + target_chain_w * (w / norm)
                else:
                    p = positions[vi]
                    dist, seg_i, t = nearest_on_polyline(p, chain_pts_of_side[side])
                    slots = chain_slots_of_side[side]
                    ja, jb = slots[seg_i], slots[seg_i + 1]
                    if t <= 0.0:
                        adds[ja] = adds.get(ja, 0.0) + target_chain_w
                    elif t >= 1.0:
                        adds[jb] = adds.get(jb, 0.0) + target_chain_w
                    else:
                        adds[ja] = adds.get(ja, 0.0) + target_chain_w * (1.0 - t)
                        adds[jb] = adds.get(jb, 0.0) + target_chain_w * t

            remove_slots = full_chain_slots_of_side[side] | {shoulder_slot}
            new_j, new_w = rebuild_vertex(joints0[vi], weights0[vi], remove_slots, adds)

            # V4 safety: never let this vertex become Head/Neck-dominant
            # while still carrying >0.2 combined shoulder+chain ("arm")
            # weight -- equalizing two vertices' split must not undo the
            # main pass's own head/neck weight-leakage guarantee. If it
            # would, leave this vertex's weights untouched this
            # iteration (the OTHER endpoint may still move).
            new_dom = int(new_j[int(np.argmax(new_w))])
            if new_dom in head_neck_slots:
                arm_w = sum(w for j, w in zip(new_j, new_w) if j == shoulder_slot or j in full_chain_slots_of_side[side])
                if arm_w > 0.2:
                    continue

            joints0[vi] = new_j
            weights0[vi] = new_w
            touched.add(int(vi))

        if verbose:
            print(f"  relax iter {it}: worst edge ({wa},{wb}) ratio {worst_ratio:.2f}x -> equalized")

    return touched, worst_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("output_glb")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    scene = GLTFScene(args.input_glb)
    g = scene.g
    prim = scene.get_mesh_primitive()

    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]
    name_to_slot = {n: i for i, n in enumerate(joint_names)}

    joints0, weights0 = scene.get_skin_weights()
    joints0 = joints0.astype(np.int64).copy()
    weights0 = weights0.astype(np.float64).copy()
    positions = scene.get_positions()
    indices = scene.get_indices()
    N = joints0.shape[0]

    adjacency = build_adjacency(indices, N)

    # bind-pose world position of every joint (model space; the mesh node
    # itself has an identity transform, verified once at pipeline-build
    # time -- so joint bind-world position is directly comparable to
    # POSITION).
    cache = {}
    bind_pos = {}
    for j in joints:
        m = scene.node_bind_world(j, cache)
        bind_pos[g.nodes[j].name] = m[:3, 3]

    def dominant_slot_vec(w0):
        return joints0[np.arange(N), np.argmax(w0, axis=1)]

    before_dominant = dominant_slot_vec(weights0)
    before_counts = {n: int((before_dominant == name_to_slot[n]).sum()) for n in joint_names}
    head_neck_slots = set(i for i, n in enumerate(joint_names) if any(tok in n.lower() for tok in ("head", "neck")))

    report = {"sides": {}}
    changed_vertices = set()

    side_of_slot = {}
    shoulder_slot_of_side = {}
    full_chain_slots_of_side = {}
    chain_pts_of_side = {}
    chain_slots_of_side = {}
    CHAIN_SUFFIXES = ["Arm", "ArmTwist2", "ForeArm", "ForeArmTwist2", "Hand"]

    for side in SIDES:
        shoulder_name = f"{side}Shoulder"
        shoulder_slot = name_to_slot[shoulder_name]
        full_chain_names = [f"{side}{suf}" for suf in FULL_CHAIN_SUFFIXES]
        full_chain_slots = set(name_to_slot[n] for n in full_chain_names)
        arm_group_slots = full_chain_slots | {shoulder_slot}

        shoulder_slot_of_side[side] = shoulder_slot
        full_chain_slots_of_side[side] = full_chain_slots
        chain_slots_of_side[side] = [name_to_slot[f"{side}{suf}"] for suf in CHAIN_SUFFIXES]
        chain_pts_of_side[side] = [bind_pos[f"{side}{suf}"] for suf in CHAIN_SUFFIXES]
        for s in arm_group_slots:
            side_of_slot[s] = side

        arm_pos = bind_pos[f"{side}Arm"]
        fore_pos = bind_pos[f"{side}ForeArm"]
        axis = fore_pos - arm_pos
        arm_len = float(np.linalg.norm(axis))
        axis_n = axis / arm_len
        ring_half = RING_HALF_FRAC * arm_len

        has_arm_group = np.isin(joints0, list(arm_group_slots)) & (weights0 > 0)
        candidate_mask = has_arm_group.any(axis=1)
        candidate_idxs = np.where(candidate_mask)[0]

        # Core gate: a vertex ALREADY carrying some weight (>=
        # MIN_CHAIN_WEIGHT_TO_REINFORCE, currently "any nonzero share")
        # on the limb chain itself (Arm/ArmTwist/ForeArm/ForeArmTwist/
        # Hand) -- i.e. the auto-rig already treated it as a (perhaps
        # under-weighted) part of the moving limb, so there is something
        # concrete to reinforce rather than invent. A vertex whose ONLY
        # arm-group weight is on the Shoulder is left alone here (some
        # of those are nowhere near the actual limb, e.g. a stray
        # cosmetic Shoulder weight bleed reaching around onto the back
        # of the torso, where the plane-distance sign is uninformative);
        # any residual short-edge strain this leaves at a genuine
        # chain/no-chain mesh seam is cleaned up surgically afterward by
        # relax_edge_strain(), which touches ONLY the specific vertices
        # on an actually-offending edge.
        MIN_CHAIN_WEIGHT_TO_REINFORCE = 0.0
        core_idxs = []
        chain_existing_cache = {}
        for vi in candidate_idxs:
            row_j = joints0[vi]
            row_w = weights0[vi]
            chain_existing = {}
            for k in range(4):
                j = int(row_j[k]); w = float(row_w[k])
                if w > 0 and j in full_chain_slots:
                    chain_existing[j] = chain_existing.get(j, 0.0) + w
            if chain_existing and sum(chain_existing.values()) >= MIN_CHAIN_WEIGHT_TO_REINFORCE:
                chain_existing_cache[vi] = chain_existing
                core_idxs.append(int(vi))

        idxs = np.array(core_idxs, dtype=np.int64)

        # --- raw target shoulder-fraction from the plane distance ---
        raw_frac = {}
        for vi in idxs:
            p = positions[vi]
            d = float((p - arm_pos) @ axis_n)
            raw_frac[vi] = float(np.clip((ring_half - d) / (2 * ring_half), 0.0, 1.0))

        # --- smooth over real mesh adjacency (see module docstring #3) ---
        smoothed_frac = smooth_field(raw_frac, idxs, adjacency, SMOOTH_ITERS, SMOOTH_ALPHA)

        n_touched = 0
        n_full_transfer = 0
        n_blend = 0
        n_untouched_medial = 0
        n_reinforced_existing = 0

        for vi in idxs:
            row_j = joints0[vi]
            row_w = weights0[vi]
            shoulder_w = float(row_w[row_j == shoulder_slot].sum())
            chain_existing = chain_existing_cache[vi]
            chain_w_total = sum(chain_existing.values())
            pool = shoulder_w + chain_w_total
            if pool <= EPS:
                continue

            frac = smoothed_frac[vi]
            target_shoulder = pool * frac
            target_chain = pool - target_shoulder

            adds = {}
            if target_shoulder > EPS:
                adds[shoulder_slot] = target_shoulder
            if target_chain > EPS:
                # Reinforce the vertex's own existing chain proportions --
                # never an invented placement.
                for j, w in chain_existing.items():
                    adds[j] = adds.get(j, 0.0) + target_chain * (w / chain_w_total)
                n_reinforced_existing += 1

            # No-op guard (within float tolerance of the vertex's current
            # split) -- still counts toward "touched" bookkeeping only if
            # it actually changes something material.
            unchanged = (
                abs(target_shoulder - shoulder_w) < 1e-6 and
                all(abs(adds.get(j, 0.0) - w) < 1e-6 for j, w in chain_existing.items())
            )
            if unchanged:
                if frac >= 1.0 - 1e-9:
                    n_untouched_medial += 1
                continue

            new_j, new_w = rebuild_vertex(row_j, row_w, arm_group_slots, adds)

            # V4 safety: don't let this vertex become Head/Neck-dominant
            # while carrying >0.2 combined shoulder+chain weight (see the
            # identical guard in relax_edge_strain for the full rationale).
            new_dom = int(new_j[int(np.argmax(new_w))])
            if new_dom in head_neck_slots:
                arm_w = sum(w for j, w in zip(new_j, new_w) if j in arm_group_slots)
                if arm_w > 0.2:
                    continue

            joints0[vi] = new_j
            weights0[vi] = new_w
            changed_vertices.add(int(vi))
            n_touched += 1
            if frac <= 1e-9:
                n_full_transfer += 1
            else:
                n_blend += 1

        report["sides"][side] = {
            "shoulder_joint": shoulder_name,
            "upper_arm_bone_length": arm_len,
            "ring_half_width": ring_half,
            "min_chain_weight_to_reinforce": MIN_CHAIN_WEIGHT_TO_REINFORCE,
            "vertices_in_arm_group": int(len(idxs)),
            "vertices_untouched_medial": n_untouched_medial,
            "vertices_fully_transferred": n_full_transfer,
            "vertices_blended_at_seam": n_blend,
            "vertices_touched_total": n_touched,
            "vertices_reinforced_existing_chain_weight": n_reinforced_existing,
        }

    # ---- targeted edge-strain relaxation pass (see relax_edge_strain) ----
    wave_anim = None
    for a in g.animations:
        if a.name == "wave":
            wave_anim = a
            break
    relax_touched = set()
    relax_history = []
    if wave_anim is not None:
        waving_side = detect_waving_side(scene, wave_anim, joints, name_to_slot)
        waving_core_slots = full_chain_slots_of_side[waving_side]
        print(f"\nrelaxing edge strain (waving side: {waving_side}) ...")
        relax_touched, relax_history = relax_edge_strain(
            scene, wave_anim, joints0, weights0, positions, indices, adjacency,
            waving_core_slots, side_of_slot, shoulder_slot_of_side,
            full_chain_slots_of_side, chain_pts_of_side, chain_slots_of_side,
            joints, ibm, head_neck_slots,
        )
        changed_vertices |= relax_touched
        print(f"  relaxation touched {len(relax_touched)} vertices; "
              f"worst-edge ratio {relax_history[0]:.2f}x -> {relax_history[-1]:.2f}x"
              if relax_history else "  relaxation: no offending body edge found")
    report["edge_strain_relaxation"] = {
        "vertices_touched": len(relax_touched),
        "worst_ratio_by_iteration": relax_history,
    }

    after_dominant = dominant_slot_vec(weights0)
    after_counts = {n: int((after_dominant == name_to_slot[n]).sum()) for n in joint_names}

    # sanity: WEIGHTS_0 rows still sum to 1
    sums = weights0.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > 1e-4)[0]
    if len(bad):
        raise SystemExit(f"internal error: {len(bad)} vertex rows do not sum to 1 (first: {bad[:5]})")

    print(f"touched {len(changed_vertices)}/{N} vertices ({len(changed_vertices) / N * 100:.2f}%)")
    for side in SIDES:
        s = report["sides"][side]
        print(f"  {side}: {s['vertices_touched_total']} touched "
              f"({s['vertices_fully_transferred']} full transfer, {s['vertices_blended_at_seam']} blended)")

    arm_related = [n for n in joint_names if any(tok in n.lower() for tok in
                   ("arm", "hand", "shoulder"))]
    print("\ndominant-joint vertex counts, before -> after:")
    for n in sorted(arm_related):
        b, a = before_counts.get(n, 0), after_counts.get(n, 0)
        if b or a:
            print(f"  {n:20s} {b:5d} -> {a:5d}")

    # ---- write back JOINTS_0 (u8 VEC4) and WEIGHTS_0 (f32 VEC4) ----
    blob = bytearray(scene.blob)

    j_acc = g.accessors[prim.attributes.JOINTS_0]
    j_bv = g.bufferViews[j_acc.bufferView]
    j_stride = j_bv.byteStride or 4
    j_offset = j_bv.byteOffset or 0
    for i in range(N):
        packed = bytes(int(x) for x in joints0[i])
        blob[j_offset + i * j_stride: j_offset + i * j_stride + 4] = packed

    w_acc = g.accessors[prim.attributes.WEIGHTS_0]
    w_bv = g.bufferViews[w_acc.bufferView]
    w_stride = w_bv.byteStride or 16
    w_offset = w_bv.byteOffset or 0
    for i in range(N):
        packed = weights0[i].astype(np.float32).tobytes()
        blob[w_offset + i * w_stride: w_offset + i * w_stride + 16] = packed
    w_acc.min = [float(v) for v in weights0.min(axis=0)]
    w_acc.max = [float(v) for v in weights0.max(axis=0)]

    g.set_binary_blob(bytes(blob))
    g.save(args.output_glb)
    print(f"\nwrote {args.output_glb}")

    report["vertices_touched_total"] = len(changed_vertices)
    report["vertex_count"] = N
    report["before_dominant_counts"] = before_counts
    report["after_dominant_counts"] = after_counts
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
