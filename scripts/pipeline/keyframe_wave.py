#!/usr/bin/env python3
"""
Author the 'wave' animation clip directly on the rigged kitty skeleton.

Tripo's preset animation library has no "wave" motion (checked: idle, walk,
run, dive, climb, jump, slash, shoot, hurt, fall, turn, quadruped/hexapod/
octopod/serpentine/aquatic marches -- no wave). Per the work order, the wave
clip is therefore hand-keyframed on the generated rig's own arm joints
(mesh + skeleton are still 100% Tripo-generated; only the animation curves
are authored here).

Important: the rig's raw BIND pose is the arms-apart "cheer" pose used to
generate/rig the mesh (arms already raised) -- that is *not* the relaxed
standing pose the other three clips use. `animate_retarget` repose every
joint's rotation to a neutral standing pose for its presets. So this script
takes the `idle` clip's frame-0 rotation as the neutral baseline for every
joint (translations are left at the rig's bind values, which match to
within noise -- this keeps hips/root exactly at bind, i.e. not moving at
all), and only the waving arm's chain is animated on top of that baseline.
That keeps the "other arm at rest" and "torso/hips stable" requirements
true by construction, and keeps the pose continuous with idle/jump/walk.

Approach for the waving arm: forward-kinematics-consistent keyframing.
  * Compute each joint's NEUTRAL WORLD rotation by composing the idle
    clip's frame-0 local rotations from the scene root down to the joint.
  * Shoulder (Upperarm): neutral -> raised, single smoothstep ramp up,
    hold, smoothstep back down at the end -- rotation about the model's
    world Z axis (Y-up model; a world-Z rotation sweeps the arm sideways-
    and-up in the frontal plane, matching the arms-apart reference pose).
  * Forearm: rigidly follows the raising upper arm (preserving the
    neutral elbow bend) plus an additional, enveloped oscillation in world
    space once the arm is up -- this is the actual "wave".
  * Every other joint gets a constant rotation channel pinned to the idle
    neutral pose (so nothing silently reverts to the cheer bind pose).
  * Local rotation is recovered as local = inverse(parent_world) *
    desired_world, using the *animated* parent orientation for the
    forearm (whose parent, the upper arm, is itself moving).

Usage:
    python3 keyframe_wave.py <rigged_base.glb> <neutral_clip.glb> \
        <out_wave.glb> [--neutral-clip-name idle] [--side R]
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


def smoothstep(t, edge0, edge1):
    if edge1 <= edge0:
        return 1.0 if t >= edge1 else 0.0
    x = np.clip((t - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def quat_xyzw(rot: "R"):
    q = rot.as_quat()  # x,y,z,w
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb", help="base rigged model (mesh+skin+skeleton, no animation)")
    ap.add_argument("neutral_glb", help="a retargeted clip (e.g. idle) used for the neutral standing pose")
    ap.add_argument("output_glb")
    ap.add_argument("--neutral-clip-name", default="idle")
    ap.add_argument("--side", choices=["L", "R"], default="R",
                     help="which arm waves (default: R / right arm)")
    ap.add_argument("--duration", type=float, default=3.2)
    ap.add_argument("--raise-deg", type=float, default=100.0)
    ap.add_argument("--wave-amp-deg", type=float, default=24.0)
    ap.add_argument("--wave-period", type=float, default=0.8)
    args = ap.parse_args()

    base = GLTFScene(args.input_glb)
    neutral = GLTFScene(args.neutral_glb)
    g = base.g
    blob = bytearray(base.blob)

    names = {n.name: i for i, n in enumerate(g.nodes) if n.name}
    neutral_names = {n.name: i for i, n in enumerate(neutral.g.nodes) if n.name}
    neutral_anim = neutral.get_animation(args.neutral_clip_name)

    side = args.side
    clavicle_name, upperarm_name, forearm_name = f"{side}_Clavicle", f"{side}_Upperarm", f"{side}_Forearm"
    clavicle, upperarm, forearm = names[clavicle_name], names[upperarm_name], names[forearm_name]

    # --- neutral local rotation for every joint, taken from the idle clip's frame 0 ---
    neutral_local_rot = {}
    for name, idx in names.items():
        nidx = neutral_names.get(name)
        if nidx is None:
            _, r0, _ = base.node_bind_local(idx)
            neutral_local_rot[idx] = np.array(r0)
            continue
        _, r, _ = neutral.sample_node_local(neutral_anim, nidx, 0.0)
        neutral_local_rot[idx] = np.array(r)

    parent_map = base.parent

    def neutral_world_rotation(idx, cache={}):
        if idx in cache:
            return cache[idx]
        q = neutral_local_rot[idx]
        rot = R.from_quat(q)
        if idx in parent_map:
            rot = neutral_world_rotation(parent_map[idx]) * rot
        cache[idx] = rot
        return rot

    clavicle_world_neutral = neutral_world_rotation(clavicle)
    upperarm_world_neutral = neutral_world_rotation(upperarm)
    forearm_world_neutral = neutral_world_rotation(forearm)
    forearm_local_bind_offset = upperarm_world_neutral.inv() * forearm_world_neutral

    z_axis = np.array([0.0, 0.0, 1.0])
    # Right arm waves with +Z sweep; left arm mirrors with -Z so the raise
    # still goes outward/up instead of into the body.
    sign = 1.0 if side == "R" else -1.0

    n_frames = int(round(args.duration * FPS)) + 1
    times = np.linspace(0.0, args.duration, n_frames)

    t_raise_end = 0.5
    t_lower_start = args.duration - 0.6
    t_wave_fade_in_end = 0.7
    t_wave_fade_out_start = max(t_wave_fade_in_end, t_lower_start - 0.2)

    upperarm_quats = []
    forearm_quats = []
    for t in times:
        raise_env = smoothstep(t, 0.0, t_raise_end) * (1.0 - smoothstep(t, t_lower_start, args.duration))
        shoulder_angle = np.radians(args.raise_deg) * raise_env

        wave_env = smoothstep(t, t_raise_end - 0.1, t_wave_fade_in_end) * \
            (1.0 - smoothstep(t, t_wave_fade_out_start, t_lower_start + 0.1))
        wave_angle = np.radians(args.wave_amp_deg) * wave_env * \
            np.sin(2 * np.pi * (t - t_raise_end) / args.wave_period)

        extra_shoulder = R.from_rotvec(sign * shoulder_angle * z_axis)
        desired_upperarm_world = extra_shoulder * upperarm_world_neutral

        # forearm rigidly follows the upper arm (preserving the neutral elbow
        # bend) plus an additional wave wobble expressed in world space
        natural_forearm_world = desired_upperarm_world * forearm_local_bind_offset
        extra_elbow = R.from_rotvec(sign * wave_angle * z_axis)
        desired_forearm_world = extra_elbow * natural_forearm_world

        local_upperarm = clavicle_world_neutral.inv() * desired_upperarm_world
        local_forearm = desired_upperarm_world.inv() * desired_forearm_world

        upperarm_quats.append(quat_xyzw(local_upperarm))
        forearm_quats.append(quat_xyzw(local_forearm))

    upperarm_quats = np.array(upperarm_quats, dtype=np.float32)
    forearm_quats = np.array(forearm_quats, dtype=np.float32)
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
    upperarm_acc = append_accessor(upperarm_quats, 5126, "VEC4")
    forearm_acc = append_accessor(forearm_quats, 5126, "VEC4")

    # constant 2-key neutral-pose channel for every other joint (so the wave
    # clip never silently falls back to the raw arms-apart bind pose)
    static_times = np.array([0.0, args.duration], dtype=np.float32)
    static_time_acc = append_accessor(static_times, 5126, "SCALAR", extra_min_max=True)

    wave_anim = Animation(name="wave")
    samplers = []
    channels = []

    def add_channel(node_idx, input_acc, output_data):
        s_idx = len(samplers)
        out_acc = append_accessor(output_data.astype(np.float32), 5126, "VEC4")
        samplers.append(AnimationSampler(input=input_acc, output=out_acc, interpolation="LINEAR"))
        channels.append(AnimationChannel(sampler=s_idx, target=AnimationChannelTarget(node=node_idx, path="rotation")))

    add_channel(upperarm, time_acc, upperarm_quats)
    add_channel(forearm, time_acc, forearm_quats)

    animated_nodes = {upperarm, forearm}
    for name, idx in names.items():
        if idx in animated_nodes:
            continue
        q = neutral_local_rot[idx]
        add_channel(idx, static_time_acc, np.array([q, q]))

    wave_anim.samplers = samplers
    wave_anim.channels = channels
    g.animations.append(wave_anim)

    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save(args.output_glb)
    print(f"wrote {args.output_glb}: wave animation on {upperarm_name} (node {upperarm}) "
          f"and {forearm_name} (node {forearm}), duration={args.duration}s, frames={n_frames}, "
          f"{len(channels)} total channels (neutral pose pinned on {len(channels)-2} other joints)")


if __name__ == "__main__":
    main()
