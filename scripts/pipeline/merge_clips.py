#!/usr/bin/env python3
"""
Merge separately-generated animation clips onto ONE base rigged model,
producing a single GLB with one skeleton + one skin carrying all requested
animations (named exactly as required).

Each source clip GLB (idle/jump/walk from Tripo's animate_retarget preset
pipeline, plus the hand-authored wave clip) was exported from the *same*
Tripo rig task, so they share an identical node hierarchy/order; channels
are still matched by joint NAME (not raw index) for safety, per the work
order.

Usage:
  python3 merge_clips.py <base_rigged.glb> <out.glb> \
      idle=<idle.glb> jump=<jump.glb> walk=<walk.glb> wave=<wave.glb> \
      [--rename-map name_map.json]
"""
import sys
import json
import argparse
import numpy as np
from pygltflib import GLTF2, Animation, AnimationChannel, AnimationChannelTarget, \
    AnimationSampler, Accessor, BufferView

_COMP_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path):
    g = GLTF2().load(path)
    blob = bytearray(g.binary_blob())
    return g, blob


def node_name_map(g):
    return {n.name: i for i, n in enumerate(g.nodes) if n.name}


def read_accessor_raw_bytes(g, blob, acc_idx):
    """Return (raw_bytes, component_type, type_str, count, min, max) for an accessor,
    reading directly from its bufferView (assumes no interleaving/stride tricks
    beyond simple tight packing, true for Tripo/pygltflib-authored exports here)."""
    acc = g.accessors[acc_idx]
    bv = g.bufferViews[acc.bufferView]
    comp_fmt = _COMP_FMT[acc.componentType]
    ncomp = _NCOMP[acc.type]
    import struct
    comp_size = struct.calcsize(comp_fmt)
    elem_size = comp_size * ncomp
    stride = bv.byteStride or elem_size
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    if stride == elem_size:
        raw = bytes(blob[offset:offset + elem_size * acc.count])
    else:
        raw = bytearray()
        for i in range(acc.count):
            raw += blob[offset + i * stride: offset + i * stride + elem_size]
        raw = bytes(raw)
    return raw, acc.componentType, acc.type, acc.count, acc.min, acc.max


def append_bytes_as_accessor(dst_g, dst_blob, raw, comp_type, type_str, count, mn, mx):
    byte_offset = len(dst_blob)
    dst_blob += raw
    while len(dst_blob) % 4 != 0:
        dst_blob += b"\x00"
    bv = BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(raw))
    bv_idx = len(dst_g.bufferViews)
    dst_g.bufferViews.append(bv)
    acc = Accessor(bufferView=bv_idx, componentType=comp_type, count=count, type=type_str)
    if mn is not None:
        acc.min = mn
    if mx is not None:
        acc.max = mx
    acc_idx = len(dst_g.accessors)
    dst_g.accessors.append(acc)
    return acc_idx


def copy_animation_into(dst_g, dst_blob, src_g, src_blob, clip_name, dst_name_map, src_name_map):
    # find the animation in source (by name if present, else assume single/first)
    src_anim = None
    for a in src_g.animations:
        if a.name == clip_name or len(src_g.animations) == 1:
            src_anim = a
            break
    if src_anim is None:
        raise ValueError(f"no animation found for clip {clip_name!r} in source")

    new_anim = Animation(name=clip_name)
    new_samplers = []
    new_channels = []

    # map: source sampler index -> new sampler index (samplers may be reused
    # across channels in the source; preserve that as an optimisation but it's
    # also correct to just duplicate -- we duplicate for simplicity/safety)
    for ch in src_anim.channels:
        src_node_idx = ch.target.node
        src_node_name = src_g.nodes[src_node_idx].name
        if src_node_name not in dst_name_map:
            # target joint doesn't exist on the destination skeleton -- skip
            # (shouldn't happen: all clips share the same rig)
            print(f"  [warn] {clip_name}: node {src_node_name!r} not found on base rig, skipping channel")
            continue
        dst_node_idx = dst_name_map[src_node_name]

        sampler = src_anim.samplers[ch.sampler]
        in_raw, in_ct, in_t, in_n, in_min, in_max = read_accessor_raw_bytes(src_g, src_blob, sampler.input)
        out_raw, out_ct, out_t, out_n, out_min, out_max = read_accessor_raw_bytes(src_g, src_blob, sampler.output)

        in_acc = append_bytes_as_accessor(dst_g, dst_blob, in_raw, in_ct, in_t, in_n, in_min, in_max)
        out_acc = append_bytes_as_accessor(dst_g, dst_blob, out_raw, out_ct, out_t, out_n, out_min, out_max)

        new_sampler_idx = len(new_samplers)
        new_samplers.append(AnimationSampler(input=in_acc, output=out_acc,
                                              interpolation=sampler.interpolation or "LINEAR"))
        new_channels.append(AnimationChannel(
            sampler=new_sampler_idx,
            target=AnimationChannelTarget(node=dst_node_idx, path=ch.target.path)))

    new_anim.samplers = new_samplers
    new_anim.channels = new_channels
    dst_g.animations.append(new_anim)
    print(f"  merged clip {clip_name!r}: {len(new_channels)} channels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_glb")
    ap.add_argument("out_glb")
    ap.add_argument("clips", nargs="+", help="name=path.glb pairs")
    ap.add_argument("--rename-map", default=None, help="optional JSON old_name->new_name node rename map")
    args = ap.parse_args()

    dst_g, dst_blob = load(args.base_glb)
    dst_names = node_name_map(dst_g)

    for pair in args.clips:
        clip_name, path = pair.split("=", 1)
        src_g, src_blob = load(path)
        src_names = node_name_map(src_g)
        copy_animation_into(dst_g, dst_blob, src_g, src_blob, clip_name, dst_names, src_names)

    if args.rename_map:
        with open(args.rename_map) as f:
            rename = json.load(f)
        renamed = 0
        for n in dst_g.nodes:
            if n.name in rename:
                n.name = rename[n.name]
                renamed += 1
        print(f"renamed {renamed} nodes for validator-friendly joint names")

    dst_g.buffers[0].byteLength = len(dst_blob)
    dst_g.set_binary_blob(bytes(dst_blob))
    dst_g.save(args.out_glb)
    print(f"wrote {args.out_glb} with {len(dst_g.animations)} animations: "
          f"{[a.name for a in dst_g.animations]}")


if __name__ == "__main__":
    main()
