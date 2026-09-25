#!/usr/bin/env python3
"""
Documented, scripted skin-weight cleanup for the auto-rigged kitty mesh.

Per the manager's ruling for this round: a regeneration whose bind pose is
close to neutral is the PREFERRED fix for a dirty auto-rig, and this
round's regenerated arms-apart input (a shallow ~35-40deg-down A-pose,
see kitty.provenance.json stage 1b) already removed the round-1 fused-arm
lobe entirely and cut arm-chain weight leakage into the head/neck region
from 5,332/8,695 head-region vertices (round 1) down to 768/5,757
Head-or-Neck-*dominant* vertices carrying >0.2 arm weight (this round's
rigged_v1.glb, before this script runs) -- but it does not reach zero.
Per the ruling, a SCRIPTED post-process is an acceptable documented
fallback for the remainder, so this script performs exactly that: it does
NOT touch geometry, joint hierarchy, or bind poses -- only the per-vertex
skin weights (JOINTS_0 / WEIGHTS_0), and only to remove cross-contamination
between the "arm chain" bone group and everything else:

  * A vertex whose SINGLE LARGEST weight is a non-arm joint (torso, head,
    neck, hips, legs, tail, ...) has ALL of its arm-chain weight (either
    arm, including clavicle/shoulder and the twist bones) zeroed and the
    remainder renormalized to sum to 1. This is what the wave clip's V4
    (zero Head/Neck-dominant vertices with >0.2 arm weight) and V3
    (co-deformation of every non-waving-arm vertex) checks need: those
    vertices can now only move with joints that never move during `wave`.
  * A vertex dominated by the LEFT arm chain has any residual RIGHT arm
    chain weight zeroed (and vice versa), so the resting arm truly cannot
    be dragged by the waving one, and its own torso/clavicle blend weight
    (which is legitimate, for a smooth shoulder seam) is left untouched.

This is a strictly vertex-local weight edit -- it cannot change which
joints influence which vertex beyond removing the listed cross-group
contamination, and it always renormalizes so WEIGHTS_0 still sums to 1
per vertex (glTF requirement).

Usage:
    python3 clean_skin_weights.py <in_rigged.glb> <out_rigged.glb>
"""
import sys
import argparse
import numpy as np
from pygltflib import GLTF2

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gltf_anim import GLTFScene

LEFT_ARM_PARTS = ["L_Clavicle", "L_Upperarm", "L_UpperarmTwist01", "L_UpperarmTwist02",
                  "L_Forearm", "L_ForearmTwist01", "L_ForearmTwist02", "L_Hand"]
RIGHT_ARM_PARTS = ["R_Clavicle", "R_Upperarm", "R_UpperarmTwist01", "R_UpperarmTwist02",
                   "R_Forearm", "R_ForearmTwist01", "R_ForearmTwist02", "R_Hand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("output_glb")
    args = ap.parse_args()

    scene = GLTFScene(args.input_glb)
    g = scene.g
    prim = scene.get_mesh_primitive()

    joints, ibm = scene.get_skin()
    joint_names = [g.nodes[j].name for j in joints]
    name_to_slot = {n: i for i, n in enumerate(joint_names)}

    left_slots = set(name_to_slot[n] for n in LEFT_ARM_PARTS if n in name_to_slot)
    right_slots = set(name_to_slot[n] for n in RIGHT_ARM_PARTS if n in name_to_slot)
    arm_slots = left_slots | right_slots

    joints0, weights0 = scene.get_skin_weights()
    joints0 = joints0.copy()
    weights0 = weights0.copy()
    N = joints0.shape[0]

    dominant_slot = joints0[np.arange(N), np.argmax(weights0, axis=1)]
    is_left_dom = np.isin(dominant_slot, list(left_slots))
    is_right_dom = np.isin(dominant_slot, list(right_slots))
    is_other_dom = ~(is_left_dom | is_right_dom)

    changed = 0
    for i in range(N):
        row_joints = joints0[i]
        row_weights = weights0[i]
        zero_mask = np.zeros(4, dtype=bool)
        if is_other_dom[i]:
            zero_mask = np.isin(row_joints, list(arm_slots))
        elif is_left_dom[i]:
            zero_mask = np.isin(row_joints, list(right_slots))
        elif is_right_dom[i]:
            zero_mask = np.isin(row_joints, list(left_slots))
        if zero_mask.any() and row_weights[zero_mask].sum() > 0:
            row_weights = row_weights.copy()
            row_weights[zero_mask] = 0.0
            total = row_weights.sum()
            if total > 1e-9:
                row_weights = row_weights / total
            else:
                # degenerate: fall back to full weight on the single largest original slot
                row_weights = np.zeros(4)
                row_weights[np.argmax(weights0[i])] = 1.0
            weights0[i] = row_weights
            changed += 1

    print(f"cleaned {changed}/{N} vertices ({changed/N*100:.2f}%)")

    # write weights back in place (float32 VEC4, same layout as source)
    acc = g.accessors[prim.attributes.WEIGHTS_0]
    bv = g.bufferViews[acc.bufferView]
    blob = bytearray(scene.blob)
    stride = bv.byteStride or 16
    offset = bv.byteOffset or 0
    for i in range(N):
        packed = weights0[i].astype(np.float32).tobytes()
        blob[offset + i * stride: offset + i * stride + 16] = packed
    acc.min = [float(v) for v in weights0.min(axis=0)]
    acc.max = [float(v) for v in weights0.max(axis=0)]

    g.set_binary_blob(bytes(blob))
    g.save(args.output_glb)
    print(f"wrote {args.output_glb}")


if __name__ == "__main__":
    main()
