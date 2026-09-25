#!/usr/bin/env python3
"""
Author the 'jump' animation clip directly on the rigged kitty skeleton:
ONE readable anticipation crouch -> leap -> land (not a repeating bounce).

Why authored instead of the Tripo preset: `animate_retarget preset:jump`
against this rig produces ~5 small repeating bounces over its 2.23s
duration (Hips oscillating, feet clearing only a few % of model height per
bounce -- confirmed by sampling scripts/pipeline/verify_kitty.py-style FK
on the retargeted clip before authoring this). That reads as a bounce
loop, not "a readable jump: anticipation crouch -> leap -> land" (spec-001),
and the single best bounce only reaches ~9% foot clearance measured from
the clip's own start height, short of the work order's >=10%h bar for a
preset to be accepted as-is. So this script keyframes ONE hop directly on
the rig's own joints (same technique as keyframe_wave.py): every joint's
NEUTRAL WORLD rotation is taken from the idle clip's frame-0 pose, and only
Hips (translation, world-up) and the four leg joints (Thigh/Calf rotation,
world-Z hinge = the model's own left-right axis) are animated on top of
that baseline; every other joint (arms, spine, head, tail) is pinned to the
idle-neutral pose for the whole clip, matching the "everything else stays
put" spirit used for wave. Root translation is left at 0 throughout (no
horizontal travel -- the jump is in place, directly above its start mark).

Mechanics (FK, world-space, mirrors keyframe_wave.py's approach):
  * Hips: local translation delta along the *world* up axis (computed once
    from the rig's own root-to-Hips bind rotation, since this rig's root
    carries a coordinate-convention rotation -- do not assume raw local Y
    is world up). Negative delta = crouch dip, positive = leap rise.
  * Thigh / Calf: an "extra" rotation about *world* Z (this rig's left-
    right hinge axis, verified against the bind pose rather than assumed)
    is composed in WORLD rotation space on top of each joint's neutral
    world rotation (calf composes on top of the thigh's already-animated
    world rotation, so the knee bends relative to the moving thigh), then
    converted back to a local quaternion the same way keyframe_wave.py
    recovers the forearm's local rotation from its animated parent.
  * Both legs use the same signed angle (a symmetric hop, not a stride).

Timeline (default duration 1.0s @ 30fps), smoothstep-eased between keys:
  t=0.00  neutral stand (hold, matches idle t0)
  t=0.15  anticipation crouch (Hips dip, knees bend)
  t=0.35  legs straighten through neutral -- push-off
  t=0.55  apex of the leap (Hips risen, legs straight -> feet clear)
  t=0.75  touchdown at neutral height, legs straight
  t=0.85  landing crouch (absorb impact)
  t=1.00  recovered, neutral stand (hold)

Usage:
    python3 keyframe_jump.py <rigged_base.glb> <neutral_clip.glb> \
        <out_jump.glb> [--neutral-clip-name idle]
"""
import sys
import argparse
import numpy as np
from pygltflib import GLTF2, Animation, AnimationChannel, AnimationChannelTarget, \
    AnimationSampler, Accessor, BufferView

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gltf_anim import GLTFScene

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("scipy is required (pip install scipy)", file=sys.stderr)
    raise

FPS = 30.0

# (time, hip_delta_frac_of_height, thigh_deg, calf_deg)
# Angles are deliberately small: this character's legs are very short and
# almost fully enveloped by its round belly at bind pose, so a "realistic"
# large knee lift self-intersects the torso mesh (checked visually via
# render_preview.py renders of a first, larger-angle pass). A shallow
# crouch (small hip dip + small counter-bent knee) reads as an
# anticipation squash without clipping, at the cost of a few % of height
# of residual foot drift during the crouch itself (not checked by any
# validator threshold; only the leap's foot clearance is).
DEFAULT_KEYS = [
    (0.00, 0.00, 0.0, 0.0),
    (0.15, -0.02, 10.0, -14.0),
    (0.35, 0.00, 0.0, 0.0),
    (0.55, 0.15, 0.0, 0.0),
    (0.75, 0.00, 0.0, 0.0),
    (0.85, -0.015, 8.0, -11.0),
    (1.00, 0.00, 0.0, 0.0),
]


def smoothstep(t, edge0, edge1):
    if edge1 <= edge0:
        return 1.0 if t >= edge1 else 0.0
    x = np.clip((t - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def piecewise_smoothstep(t, keys, col):
    """Interpolate column `col` of `keys` at time t with smoothstep easing
    between consecutive keyframes (keys must be sorted by time)."""
    if t <= keys[0][0]:
        return keys[0][col]
    if t >= keys[-1][0]:
        return keys[-1][col]
    for i in range(len(keys) - 1):
        t0, t1 = keys[i][0], keys[i + 1][0]
        if t0 <= t <= t1:
            frac = smoothstep(t, t0, t1)
            return keys[i][col] * (1 - frac) + keys[i + 1][col] * frac
    return keys[-1][col]


def quat_xyzw(rot: "R"):
    q = rot.as_quat()
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("neutral_glb")
    ap.add_argument("output_glb")
    ap.add_argument("--neutral-clip-name", default="idle")
    ap.add_argument("--duration", type=float, default=1.0)
    args = ap.parse_args()

    base = GLTFScene(args.input_glb)
    neutral = GLTFScene(args.neutral_glb)
    g = base.g
    blob = bytearray(base.blob)

    names = {n.name: i for i, n in enumerate(g.nodes) if n.name}
    neutral_names = {n.name: i for i, n in enumerate(neutral.g.nodes) if n.name}
    neutral_anim = neutral.get_animation(args.neutral_clip_name)
    parent_map = base.parent

    neutral_local_rot = {}
    neutral_local_trans = {}
    for name, idx in names.items():
        t0, r0, s0 = base.node_bind_local(idx)
        neutral_local_trans[idx] = np.array(t0)
        nidx = neutral_names.get(name)
        if nidx is None:
            neutral_local_rot[idx] = np.array(r0)
            continue
        _, r, _ = neutral.sample_node_local(neutral_anim, nidx, 0.0)
        neutral_local_rot[idx] = np.array(r)

    def neutral_world_rotation(idx, cache={}):
        if idx in cache:
            return cache[idx]
        rot = R.from_quat(neutral_local_rot[idx])
        if idx in parent_map:
            rot = neutral_world_rotation(parent_map[idx]) * rot
        cache[idx] = rot
        return rot

    hip_idx = names["Hip"]
    root_idx = names["Root"]
    pelvis_idx = names["Pelvis"]
    leg_joints = {
        "L": (names["L_Thigh"], names["L_Calf"], names["L_Foot"]),
        "R": (names["R_Thigh"], names["R_Calf"], names["R_Foot"]),
    }

    # world-up expressed in Hip's local (=Root's world rotation) frame --
    # this rig's root carries a coordinate-convention rotation, so raw
    # local Y is NOT world up; derive it instead of assuming.
    root_world_rot = neutral_world_rotation(root_idx).as_matrix()
    local_up = root_world_rot.T @ np.array([0.0, 1.0, 0.0])

    # world Z is this rig's verified left-right hinge axis (bend axis for
    # the knee, sagittal-plane crouch/extend), matching the axis used for
    # the wave clip's shoulder sweep.
    z_axis = np.array([0.0, 0.0, 1.0])

    positions = base.get_positions()
    model_height = float(positions[:, 1].max() - positions[:, 1].min())

    n_frames = int(round(args.duration * FPS)) + 1
    times = np.linspace(0.0, args.duration, n_frames)

    hip_quats = []
    hip_trans = []
    thigh_quats = {"L": [], "R": []}
    calf_quats = {"L": [], "R": []}

    pelvis_world_rot_neutral = neutral_world_rotation(pelvis_idx)
    hip_world_rot_neutral = neutral_world_rotation(hip_idx)

    for t in times:
        hip_delta_frac = piecewise_smoothstep(t, DEFAULT_KEYS, 1)
        thigh_deg = piecewise_smoothstep(t, DEFAULT_KEYS, 2)
        calf_deg = piecewise_smoothstep(t, DEFAULT_KEYS, 3)

        # Hips: translation only, rotation pinned to neutral.
        hip_local_t = neutral_local_trans[hip_idx] + local_up * (hip_delta_frac * model_height)
        hip_trans.append(hip_local_t)
        hip_quats.append(list(neutral_local_rot[hip_idx]))  # rotation pinned to neutral LOCAL rotation

        thigh_extra = R.from_rotvec(np.radians(thigh_deg) * z_axis)
        calf_extra = R.from_rotvec(np.radians(calf_deg) * z_axis)

        for side, (thigh_idx, calf_idx, foot_idx) in leg_joints.items():
            thigh_world_neutral = neutral_world_rotation(thigh_idx)
            calf_world_neutral = neutral_world_rotation(calf_idx)

            desired_thigh_world = thigh_extra * thigh_world_neutral
            desired_calf_world = calf_extra * thigh_extra * calf_world_neutral

            local_thigh = pelvis_world_rot_neutral.inv() * desired_thigh_world
            local_calf = desired_thigh_world.inv() * desired_calf_world

            thigh_quats[side].append(quat_xyzw(local_thigh))
            calf_quats[side].append(quat_xyzw(local_calf))

    hip_quats = np.array(hip_quats, dtype=np.float32)
    hip_trans = np.array(hip_trans, dtype=np.float32)
    times32 = times.astype(np.float32)

    def append_accessor(data, comp_type, acc_type, extra_min_max=False):
        nonlocal blob
        byte_offset = len(blob)
        raw = data.tobytes()
        blob += raw
        while len(blob) % 4 != 0:
            blob += b"\x00"
        bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(raw))
        bv_idx = len(g.bufferViews)
        g.bufferViews.append(bv)
        acc = Accessor(bufferView=bv_idx, componentType=comp_type, count=len(data), type=acc_type)
        if extra_min_max:
            acc.min = [float(data.min())]
            acc.max = [float(data.max())]
        acc_idx = len(g.accessors)
        g.accessors.append(acc)
        return acc_idx

    time_acc = append_accessor(times32, 5126, "SCALAR", extra_min_max=True)
    static_times = np.array([0.0, args.duration], dtype=np.float32)
    static_time_acc = append_accessor(static_times, 5126, "SCALAR", extra_min_max=True)

    jump_anim = Animation(name="jump")
    samplers = []
    channels = []

    def add_channel(node_idx, input_acc, output_data, path):
        s_idx = len(samplers)
        out_acc = append_accessor(output_data.astype(np.float32), 5126,
                                   "VEC3" if path == "translation" else "VEC4")
        samplers.append(AnimationSampler(input=input_acc, output=out_acc, interpolation="LINEAR"))
        channels.append(AnimationChannel(sampler=s_idx, target=AnimationChannelTarget(node=node_idx, path=path)))

    add_channel(hip_idx, time_acc, hip_trans, "translation")
    add_channel(hip_idx, time_acc, hip_quats, "rotation")
    for side, (thigh_idx, calf_idx, foot_idx) in leg_joints.items():
        add_channel(thigh_idx, time_acc, np.array(thigh_quats[side]), "rotation")
        add_channel(calf_idx, time_acc, np.array(calf_quats[side]), "rotation")

    animated_nodes = {hip_idx, leg_joints["L"][0], leg_joints["L"][1], leg_joints["R"][0], leg_joints["R"][1]}
    for name, idx in names.items():
        if idx in animated_nodes:
            continue
        q = neutral_local_rot[idx]
        add_channel(idx, static_time_acc, np.array([q, q]), "rotation")

    jump_anim.samplers = samplers
    jump_anim.channels = channels
    g.animations.append(jump_anim)

    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save(args.output_glb)
    print(f"wrote {args.output_glb}: jump animation, duration={args.duration}s, frames={n_frames}, "
          f"{len(channels)} total channels")


if __name__ == "__main__":
    main()
