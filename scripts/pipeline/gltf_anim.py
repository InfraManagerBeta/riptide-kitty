#!/usr/bin/env python3
"""
Minimal, dependency-light glTF skinning/animation evaluator used to:
  - sample joint-local transforms of an animation at an arbitrary time,
  - compute forward-kinematics world transforms,
  - apply linear-blend skinning (LBS) to the mesh at a given time,

so the pipeline can self-verify wave-clip numbers (arm rotation, hips
translation, torso vertex displacement) and render simple preview frames
without needing a full 3D engine / GPU.

Only the subset of glTF needed for this asset is supported (single mesh,
single skin, TRS node animation, up to 4 joint influences per vertex).
"""
import struct
import numpy as np
from pygltflib import GLTF2

_COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
         5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def quat_to_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    wx, wy, wz = w * X, w * Y, w * Z
    xx, xy, xz = x * X, x * Y, x * Z
    yy, yz, zz = y * Y, y * Z, z * Z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def trs_to_matrix(t, r, s):
    T = np.eye(4)
    if t is not None:
        T[:3, 3] = t
    R = np.eye(4)
    if r is not None:
        R[:3, :3] = quat_to_matrix(r)
    S = np.eye(4)
    if s is not None:
        S[0, 0], S[1, 1], S[2, 2] = s
    return T @ R @ S


class GLTFScene:
    def __init__(self, path):
        self.g = GLTF2().load(path)
        self.blob = self.g.binary_blob()
        self.parent = {}
        for i, n in enumerate(self.g.nodes):
            for c in (n.children or []):
                self.parent[c] = i
        self.roots = [i for i in range(len(self.g.nodes)) if i not in self.parent]

    # ---- low level accessor reading ----
    def read_accessor(self, idx):
        g = self.g
        acc = g.accessors[idx]
        bv = g.bufferViews[acc.bufferView]
        fmt, comp_size = _COMP[acc.componentType]
        ncomp = _NCOMP[acc.type]
        stride = bv.byteStride or (comp_size * ncomp)
        offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        out = np.empty((acc.count, ncomp), dtype=np.float64)
        for i in range(acc.count):
            vals = struct.unpack_from("<" + fmt * ncomp, self.blob, offset + i * stride)
            out[i] = vals
        if acc.normalized and acc.componentType in (5121, 5123):
            maxval = 255.0 if acc.componentType == 5121 else 65535.0
            out = out / maxval
        return out if ncomp > 1 else out[:, 0]

    # ---- node default (bind) TRS ----
    def node_bind_local(self, idx):
        n = self.g.nodes[idx]
        t = n.translation if n.translation is not None else [0, 0, 0]
        r = n.rotation if n.rotation is not None else [0, 0, 0, 1]
        s = n.scale if n.scale is not None else [1, 1, 1]
        return t, r, s

    def node_bind_world(self, idx, cache=None):
        if cache is None:
            cache = {}
        if idx in cache:
            return cache[idx]
        t, r, s = self.node_bind_local(idx)
        m = trs_to_matrix(t, r, s)
        if idx in self.parent:
            m = self.node_bind_world(self.parent[idx], cache) @ m
        cache[idx] = m
        return m

    # ---- animation sampling ----
    def get_animation(self, name):
        for a in self.g.animations:
            if a.name == name:
                return a
        raise KeyError(f"no animation named {name!r}")

    def animation_duration(self, anim):
        max_t = 0.0
        for s in anim.samplers:
            t = self.read_accessor(s.input)
            max_t = max(max_t, float(np.max(t)))
        return max_t

    def _sample_channel(self, anim, node_idx, path):
        for ch in anim.channels:
            if ch.target.node == node_idx and ch.target.path == path:
                sampler = anim.samplers[ch.sampler]
                times = self.read_accessor(sampler.input)
                values = self.read_accessor(sampler.output)
                return times, values
        return None

    def sample_node_local(self, anim, node_idx, t):
        """Local TRS of node_idx at time t under animation `anim`, falling back
        to the bind-pose default for any channel the node doesn't have."""
        t0, r0, s0 = self.node_bind_local(node_idx)
        translation = np.array(t0, dtype=np.float64)
        rotation = np.array(r0, dtype=np.float64)
        scale = np.array(s0, dtype=np.float64)

        for path, default in (("translation", translation), ("rotation", rotation), ("scale", scale)):
            ch = self._sample_channel(anim, node_idx, path)
            if ch is None:
                continue
            times, values = ch
            values = values.reshape(len(times), -1)
            if t <= times[0]:
                v = values[0]
            elif t >= times[-1]:
                v = values[-1]
            else:
                i = int(np.searchsorted(times, t)) - 1
                i = max(0, min(i, len(times) - 2))
                t0_, t1_ = times[i], times[i + 1]
                frac = 0.0 if t1_ <= t0_ else (t - t0_) / (t1_ - t0_)
                if path == "rotation":
                    v = slerp(values[i], values[i + 1], frac)
                else:
                    v = values[i] * (1 - frac) + values[i + 1] * frac
            if path == "translation":
                translation = v
            elif path == "rotation":
                rotation = v
            else:
                scale = v
        return translation, rotation, scale

    def world_matrices_at(self, anim, t):
        """Return dict node_idx -> 4x4 world matrix at time t for this animation."""
        cache = {}

        def compute(idx):
            if idx in cache:
                return cache[idx]
            tr, ro, sc = self.sample_node_local(anim, idx, t)
            m = trs_to_matrix(tr, ro, sc)
            if idx in self.parent:
                m = compute(self.parent[idx]) @ m
            cache[idx] = m
            return m

        for i in range(len(self.g.nodes)):
            compute(i)
        return cache

    # ---- mesh + skin ----
    def get_mesh_primitive(self):
        mesh = self.g.meshes[0]
        return mesh.primitives[0]

    def get_positions(self):
        prim = self.get_mesh_primitive()
        return self.read_accessor(prim.attributes.POSITION)

    def get_indices(self):
        prim = self.get_mesh_primitive()
        if prim.indices is None:
            return None
        return self.read_accessor(prim.indices).astype(np.int64)

    def get_skin(self):
        skin = self.g.skins[0]
        joints = skin.joints
        ibm = self.read_accessor(skin.inverseBindMatrices).reshape(-1, 4, 4)
        # glTF matrices are column-major stored flat; reshape gives rows of 4
        # each representing a column -> transpose to get row-major numpy matrix
        ibm = np.transpose(ibm, (0, 2, 1))
        return joints, ibm

    def get_skin_weights(self):
        prim = self.get_mesh_primitive()
        joints0 = self.read_accessor(prim.attributes.JOINTS_0).astype(np.int64)
        weights0 = self.read_accessor(prim.attributes.WEIGHTS_0)
        return joints0, weights0

    def write_skin_weights(self, joints0, weights0):
        """Write JOINTS_0 (u8 VEC4) / WEIGHTS_0 (f32 VEC4) back into this
        scene's in-memory blob (call .save(path) afterward to persist).
        Shared by every round-4 skin-weight post-process script
        (rebuild_face_weights.py, rebuild_arm_body_weights.py,
        weld_seam_weights.py) so the byte-packing logic lives in exactly
        one place. `joints0`/`weights0` must already be (N,4) with
        WEIGHTS_0 rows summing to 1 (glTF requirement) -- callers are
        responsible for renormalizing before calling this."""
        prim = self.get_mesh_primitive()
        g = self.g
        N = joints0.shape[0]
        blob = bytearray(self.blob)

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

        self.blob = bytes(blob)
        g.set_binary_blob(self.blob)

    def skinned_positions_at(self, anim, t):
        positions = self.get_positions()  # (N,3) mesh/model space
        joints, ibm = self.get_skin()  # joints: list of node idx; ibm: (J,4,4)
        joints0, weights0 = self.get_skin_weights()  # (N,4) each

        world = self.world_matrices_at(anim, t)
        # skin matrix per joint-slot j: world[joints[j]] @ ibm[j]
        skin_mats = np.stack([world[joints[j]] @ ibm[j] for j in range(len(joints))])  # (J,4,4)

        N = positions.shape[0]
        pos_h = np.concatenate([positions, np.ones((N, 1))], axis=1)  # (N,4)
        out = np.zeros((N, 4))
        for k in range(4):
            jslot = joints0[:, k]
            w = weights0[:, k:k + 1]
            m = skin_mats[jslot]  # (N,4,4)
            transformed = np.einsum("nij,nj->ni", m, pos_h)
            out += w * transformed
        return out[:, :3]


def slerp(q0, q1, frac):
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + frac * (q1 - q0)
        return result / np.linalg.norm(result)
    theta0 = np.arccos(dot)
    theta = theta0 * frac
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)
