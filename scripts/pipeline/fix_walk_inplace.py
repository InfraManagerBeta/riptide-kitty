#!/usr/bin/env python3
"""
Remove baked-in horizontal root motion from a retargeted `walk` clip so it
loops in place, while keeping the vertical bob (and any small lateral
sway) from the original gait.

Diagnosis (see provenance): Tripo's `animate_retarget preset:walk` bakes
the forward stride distance directly onto the Hip joint's LOCAL
translation channel (Root itself never moves -- confirmed by sampling
every channel's first vs. last keyframe: only `Hip.translation` differs
between the clip's first and last sample, by 56.9% of model height; every
rotation channel and every other translation channel already loops
perfectly). The drifting component is the joint-local axis that this
rig's root rotation maps to world -X (this model's forward axis) --
found by comparing local vs. FK world deltas, not assumed.

Fix: linearly detrend that one local-translation axis so the channel's
last sample exactly equals its first (a straight per-frame subtraction of
the line connecting the first and last keyframe values), which by
construction removes the net drift while preserving the cyclic in-between
shape (the walk gait's natural fore/aft leg-drive sway) untouched. The
other two local-translation axes (which map to world up/lateral, i.e. the
step bob and hip sway) are left alone -- their first/last samples already
match.

Usage:
    python3 fix_walk_inplace.py <in_walk.glb> <out_walk.glb> [--joint Hip]
"""
import sys
import argparse
import numpy as np
from pygltflib import GLTF2

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gltf_anim import GLTFScene

_COMP_FMT = {5126: "f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("output_glb")
    ap.add_argument("--clip-name", default="walk")
    ap.add_argument("--joint", default="Hip")
    args = ap.parse_args()

    scene = GLTFScene(args.input_glb)
    g = scene.g
    anim = scene.get_animation(args.clip_name)
    names = {n.name: i for i, n in enumerate(g.nodes) if n.name}
    joint_idx = names[args.joint]

    target_channel = None
    for ch in anim.channels:
        if ch.target.node == joint_idx and ch.target.path == "translation":
            target_channel = ch
            break
    if target_channel is None:
        raise SystemExit(f"no translation channel found for joint {args.joint!r}")

    sampler = anim.samplers[target_channel.sampler]
    times = scene.read_accessor(sampler.input)
    values = scene.read_accessor(sampler.output).reshape(len(times), 3).copy()

    delta = values[-1] - values[0]
    per_axis_drift = np.abs(delta)
    height = float(scene.get_positions()[:, 1].max() - scene.get_positions()[:, 1].min())
    drift_axes = [i for i in range(3) if per_axis_drift[i] / height > 0.01]
    print(f"{args.joint}.translation first={values[0]} last={values[-1]} delta={delta}")
    print(f"per-axis drift as %height: {per_axis_drift/height*100}")
    print(f"detrending axes {drift_axes} (>1% height drift)")

    dur = float(times[-1] - times[0])
    frac = (times - times[0]) / dur
    for ax in drift_axes:
        values[:, ax] = values[:, ax] - frac * delta[ax]

    # sanity: first must now equal last exactly (loop-safe)
    assert np.allclose(values[0], values[-1], atol=1e-6), (values[0], values[-1])

    # write the corrected values back into the accessor's buffer bytes in place
    acc = g.accessors[sampler.output]
    bv = g.bufferViews[acc.bufferView]
    blob = bytearray(scene.blob)
    offset = bv.byteOffset or 0
    stride = bv.byteStride or 12
    for i in range(len(values)):
        packed = values[i].astype(np.float32).tobytes()
        blob[offset + i * stride: offset + i * stride + 12] = packed
    acc.min = [float(v) for v in values.min(axis=0)]
    acc.max = [float(v) for v in values.max(axis=0)]

    g.set_binary_blob(bytes(blob))
    g.save(args.output_glb)
    print(f"wrote {args.output_glb}")


if __name__ == "__main__":
    main()
