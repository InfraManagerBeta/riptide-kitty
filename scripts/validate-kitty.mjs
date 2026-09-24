#!/usr/bin/env node
/**
 * validate-kitty.mjs — spec-001 asset validator for the Riptide Kitty.
 *
 * Validates `assets/kitty/kitty.glb` (preferred single-file layout) or the
 * spec-allowed four-file fallback layout
 * (`kitty-idle.glb` / `kitty-jump.glb` / `kitty-walk.glb`|`kitty-run.glb` /
 * `kitty-wave.glb`) against the spec-001 acceptance checks:
 *
 *   a. >= 1 skin
 *   b. the four clips present by exact name (`walk` OR `run` accepted)
 *   c. rendered triangles <= 20,000 (summed per node instance, all scenes)
 *   d. a PBR baseColorTexture with present image data on a material used by
 *      a skinned mesh primitive
 *   e. the wave-arm check (arm rotation, oscillations, root stability,
 *      torso rotation stability, torso co-deformation via linear blend
 *      skinning)
 *
 * Node >= 20, ESM, ZERO runtime dependencies: the GLB container, glTF JSON,
 * accessors (incl. sparse, interleaved, normalized), animation samplers
 * (STEP / LINEAR / CUBICSPLINE) and linear blend skinning are implemented
 * here directly against the glTF 2.0 specification.
 *
 * CLI:
 *   node scripts/validate-kitty.mjs [path]
 *     [path] may be a .glb file, an asset directory, or a repository root
 *     (default: ./assets/kitty relative to the current working directory).
 *   Prints a PASS/FAIL table with measured values; exits non-zero on any
 *   failure, including a missing asset.
 *
 * API:
 *   import { validate } from './validate-kitty.mjs';
 *   const report = validate({ path })            // from disk
 *   const report = validate({ files: {...} })    // in-memory GLB bytes:
 *     { single: Uint8Array }  or
 *     { idle, jump, walk|run, wave: Uint8Array }
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/* ========================================================================== *
 * Documented constants — thresholds and naming patterns.
 * ========================================================================== */

/**
 * All numeric thresholds used by the checks. Values come from
 * `specs/001-kitty-animated.md` and the work order for task fb572e15.
 */
export const THRESHOLDS = {
  /** c. Maximum rendered triangles, summed over every primitive of every
   *  mesh, once per node instance that references it (spec-001). */
  MAX_TRIANGLES: 20_000,

  /** e. Animation sampling rate for the wave analysis (samples per second,
   *  over the full clip duration, correct per-sampler interpolation). */
  WAVE_SAMPLES_PER_SECOND: 60,
  /** Safety cap on total wave samples (caps pathological clip durations). */
  WAVE_MAX_SAMPLES: 3_601,

  /** e.i  Minimum peak rotation (degrees, relative to rest pose) of at least
   *  one joint in one arm chain. */
  WAVE_MIN_ARM_PEAK_DEG: 45,

  /** e.ii Minimum oscillation count of the waving arm (an oscillation is a
   *  full back-and-forth: two direction reversals of the dominant rotation
   *  component, or two zero-crossings about its mean). */
  WAVE_MIN_OSCILLATIONS: 2,
  /** Hysteresis floor for oscillation detection: a monotone swing counts
   *  only if it spans at least this many degrees ... */
  WAVE_OSC_MIN_SWING_DEG: 10,
  /** ... or this fraction of the signal's full range, whichever is larger
   *  (rejects numeric jitter without missing small-but-real waves). */
  WAVE_OSC_MIN_SWING_FRAC: 0.2,

  /** e.iii Maximum root/hips world-translation range during `wave`,
   *  as a fraction of model height (bbox diagonal of sampled positions). */
  WAVE_MAX_ROOT_TRANSLATION_FRAC: 0.05,

  /** e.iv Maximum torso joint rotation range during `wave` (degrees,
   *  measured as the largest angle between any sampled frame's local
   *  rotation and the clip's first frame). */
  WAVE_MAX_TORSO_ROTATION_DEG: 15,

  /** e.v  Torso co-deformation: for vertices whose dominant skin weight is a
   *  torso joint, the 95th-percentile of per-vertex maximum displacement
   *  from the clip's first frame (linear blend skinning, world space) must
   *  be <= this fraction of model height. */
  WAVE_MAX_TORSO_P95_DISPLACEMENT_FRAC: 0.03,
};

/**
 * JOINT_NAME_PATTERNS — the ONE place where skeleton joint names are
 * classified. Tripo3D auto-rig joint names are unknown until the asset
 * lands, so classification is by case-insensitive TOKEN matching:
 *
 *   A joint name is tokenized on non-alphanumeric separators (`_`, `.`,
 *   `:`, `-`, spaces) AND camelCase / letter-digit boundaries, then
 *   lowercased. A category matches when any token EXACTLY equals one of its
 *   keywords. Exact-token matching (not substring) is deliberate:
 *   "Armature" must NOT match the arm category.
 *
 *   Examples: "mixamorig:LeftArm" -> [mixamorig, left, arm] (arm, left);
 *   "L_hand" -> [l, hand] (arm, left); "hand.R.001" -> [hand, r, 001]
 *   (arm, right); "UpperArm_L" -> [upper, arm, l] (arm, left);
 *   "Spine02" -> [spine, 02] (torso).
 *
 * A later merge order can extend the keyword lists here without touching
 * any classification logic.
 */
export const JOINT_NAME_PATTERNS = {
  /** Arm/hand joints — seeds of the "arm chains" (descendants included). */
  arm: ['arm', 'hand', 'shoulder', 'clavicle', 'elbow', 'wrist', 'forearm', 'paw'],
  /** Left / right side tokens (prefixes/suffixes like `L_`, `_l`, `.L`,
   *  `Left`, `mixamorig:LeftArm` all tokenize to these). */
  left: ['l', 'left'],
  right: ['r', 'right'],
  /** Root / hips joints (fallback: the skeleton's top joint). */
  root: ['root', 'hips', 'pelvis'],
  /** Torso joints for the rotation-stability check (e.iv). */
  torso: ['spine', 'chest', 'torso', 'neck', 'hips', 'pelvis'],
  /** Head/tail joints — excluded (with descendants) from the torso
   *  co-deformation vertex set (e.v). */
  head: ['head', 'skull', 'jaw', 'face', 'eye', 'ear'],
  tail: ['tail'],
};

/** Required animation clips by exact name; each inner list is "any of". */
export const REQUIRED_CLIPS = [['idle'], ['jump'], ['walk', 'run'], ['wave']];

/** Asset layout (paths relative to the repository root). */
export const ASSET_DIR = 'assets/kitty';
export const SINGLE_FILE = 'kitty.glb';
export const MULTI_FILES = {
  idle: ['kitty-idle.glb'],
  jump: ['kitty-jump.glb'],
  walk: ['kitty-walk.glb', 'kitty-run.glb'],
  wave: ['kitty-wave.glb'],
};

/* ========================================================================== *
 * Small vector / quaternion / mat4 math (column-major, glTF conventions).
 * ========================================================================== */

const DEG = 180 / Math.PI;

function quatNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
function quatDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
function quatMul(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}
function quatConj(q) { return [-q[0], -q[1], -q[2], q[3]]; }
function quatSlerp(a, b, t) {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let d = quatDot(a, b);
  if (d < 0) { d = -d; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  if (d > 0.9995) {
    return quatNormalize([
      a[0] + t * (bx - a[0]), a[1] + t * (by - a[1]),
      a[2] + t * (bz - a[2]), a[3] + t * (bw - a[3]),
    ]);
  }
  const th = Math.acos(Math.min(1, d));
  const s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [a[0] * wa + bx * wb, a[1] * wa + by * wb, a[2] * wa + bz * wb, a[3] * wa + bw * wb];
}
/** Angle (radians, [0, pi]) of the rotation q. */
function quatAngle(q) {
  const w = Math.min(1, Math.abs(q[3]) / (Math.hypot(q[0], q[1], q[2], q[3]) || 1));
  return 2 * Math.acos(w);
}
/** Rotation vector (axis * angle, radians) of q — the log map. */
function quatToRotVec(q) {
  const n = quatNormalize(q);
  const s = Math.hypot(n[0], n[1], n[2]);
  if (s < 1e-9) return [0, 0, 0];
  const ang = 2 * Math.atan2(s, n[3]);
  const k = ang / s;
  return [n[0] * k, n[1] * k, n[2] * k];
}

function mat4Identity() { return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; }
function mat4Mul(a, b) { // column-major: out = a * b
  const o = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}
function mat4FromTRS(t, q, s) {
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const [sx, sy, sz] = s;
  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    t[0], t[1], t[2], 1,
  ];
}
function mat4TransformPoint(m, p) {
  const [x, y, z] = p;
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

/* ========================================================================== *
 * GLB container + glTF accessor parsing (glTF 2.0, zero dependencies).
 * ========================================================================== */

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const COMPONENT = {
  5120: { array: Int8Array, size: 1, norm: v => Math.max(v / 127, -1) },
  5121: { array: Uint8Array, size: 1, norm: v => v / 255 },
  5122: { array: Int16Array, size: 2, norm: v => Math.max(v / 32767, -1) },
  5123: { array: Uint16Array, size: 2, norm: v => v / 65535 },
  5125: { array: Uint32Array, size: 4, norm: v => v },
  5126: { array: Float32Array, size: 4, norm: v => v },
};
const NUM_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

/** Parse a GLB byte buffer into { json, bin } (bin may be null). */
export function parseGLB(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf.byteLength < 12 || dv.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error('not a GLB file (bad magic)');
  }
  const version = dv.getUint32(4, true);
  if (version !== 2) throw new Error(`unsupported GLB version ${version} (want 2)`);
  const total = dv.getUint32(8, true);
  let off = 12;
  let json = null, bin = null;
  while (off + 8 <= Math.min(total, buf.byteLength)) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const start = off + 8;
    if (start + len > buf.byteLength) throw new Error('truncated GLB chunk');
    const chunk = buf.subarray(start, start + len);
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(chunk));
    else if (type === CHUNK_BIN) bin = chunk;
    off = start + len;
    if (off % 4) off += 4 - (off % 4); // chunks are 4-byte aligned
  }
  if (!json) throw new Error('GLB has no JSON chunk');
  return { json, bin };
}

/** A fully parsed glTF document with resolved buffers. */
class Doc {
  constructor(json, bin, baseDir, label) {
    this.json = json;
    this.label = label;
    this.baseDir = baseDir;
    this.bin = bin;
    this._buffers = new Map();
    this._accessors = new Map();
    this._parents = null;
  }

  buffer(i) {
    if (this._buffers.has(i)) return this._buffers.get(i);
    const b = (this.json.buffers || [])[i];
    if (!b) throw new Error(`buffer ${i} missing`);
    let bytes;
    if (b.uri === undefined) {
      if (!this.bin) throw new Error(`buffer ${i} refers to GLB BIN chunk but none present`);
      bytes = this.bin;
    } else if (b.uri.startsWith('data:')) {
      const comma = b.uri.indexOf(',');
      const meta = b.uri.slice(5, comma);
      const data = b.uri.slice(comma + 1);
      bytes = meta.endsWith(';base64')
        ? Uint8Array.from(Buffer.from(data, 'base64'))
        : new TextEncoder().encode(decodeURIComponent(data));
    } else {
      if (!this.baseDir) throw new Error(`buffer ${i} has external uri "${b.uri}" but document was loaded from memory`);
      const p = path.resolve(this.baseDir, decodeURIComponent(b.uri));
      if (!fs.existsSync(p)) throw new Error(`buffer uri not resolvable: ${b.uri}`);
      bytes = new Uint8Array(fs.readFileSync(p));
    }
    this._buffers.set(i, bytes);
    return bytes;
  }

  /**
   * Read accessor `i` into a dense Float64Array-like plain array of
   * count*numComponents numbers. Handles byteStride (interleaving),
   * normalized integer components, sparse accessors, and accessors without
   * a bufferView (zero-filled per spec).
   */
  accessor(i) {
    if (this._accessors.has(i)) return this._accessors.get(i);
    const a = (this.json.accessors || [])[i];
    if (!a) throw new Error(`accessor ${i} missing`);
    const nc = NUM_COMPONENTS[a.type];
    const comp = COMPONENT[a.componentType];
    if (!nc || !comp) throw new Error(`accessor ${i}: unsupported type ${a.type}/${a.componentType}`);
    const out = new Float64Array(a.count * nc);
    if (a.bufferView !== undefined) {
      const bv = this.json.bufferViews[a.bufferView];
      const bytes = this.buffer(bv.buffer);
      const base = (bv.byteOffset || 0) + (a.byteOffset || 0);
      const stride = bv.byteStride || nc * comp.size;
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const read = readerFor(dv, a.componentType);
      for (let e = 0; e < a.count; e++) {
        const eo = base + e * stride;
        for (let c = 0; c < nc; c++) {
          let v = read(eo + c * comp.size);
          if (a.normalized) v = comp.norm(v);
          out[e * nc + c] = v;
        }
      }
    }
    if (a.sparse) {
      const sp = a.sparse;
      const idxComp = COMPONENT[sp.indices.componentType];
      const iBv = this.json.bufferViews[sp.indices.bufferView];
      const iBytes = this.buffer(iBv.buffer);
      const iDv = new DataView(iBytes.buffer, iBytes.byteOffset, iBytes.byteLength);
      const iRead = readerFor(iDv, sp.indices.componentType);
      const iBase = (iBv.byteOffset || 0) + (sp.indices.byteOffset || 0);
      const vBv = this.json.bufferViews[sp.values.bufferView];
      const vBytes = this.buffer(vBv.buffer);
      const vDv = new DataView(vBytes.buffer, vBytes.byteOffset, vBytes.byteLength);
      const vRead = readerFor(vDv, a.componentType);
      const vBase = (vBv.byteOffset || 0) + (sp.values.byteOffset || 0);
      for (let k = 0; k < sp.count; k++) {
        const target = iRead(iBase + k * idxComp.size);
        for (let c = 0; c < nc; c++) {
          let v = vRead(vBase + (k * nc + c) * comp.size);
          if (a.normalized) v = comp.norm(v);
          out[target * nc + c] = v;
        }
      }
    }
    const res = { data: out, count: a.count, numComponents: nc, min: a.min, max: a.max };
    this._accessors.set(i, res);
    return res;
  }

  /** parent[i] = parent node index (or undefined for roots). */
  parents() {
    if (this._parents) return this._parents;
    const p = new Map();
    (this.json.nodes || []).forEach((n, i) => {
      for (const c of n.children || []) p.set(c, i);
    });
    this._parents = p;
    return p;
  }

  nodeName(i) {
    const n = (this.json.nodes || [])[i];
    return (n && n.name) || `node#${i}`;
  }
}

function readerFor(dv, componentType) {
  switch (componentType) {
    case 5120: return o => dv.getInt8(o);
    case 5121: return o => dv.getUint8(o);
    case 5122: return o => dv.getInt16(o, true);
    case 5123: return o => dv.getUint16(o, true);
    case 5125: return o => dv.getUint32(o, true);
    case 5126: return o => dv.getFloat32(o, true);
    default: throw new Error(`unsupported componentType ${componentType}`);
  }
}

/* ========================================================================== *
 * Joint-name classification (uses JOINT_NAME_PATTERNS — see its docs).
 * ========================================================================== */

/** Tokenize a node name: separators + camelCase + letter/digit boundaries. */
export function tokenizeName(name) {
  if (!name) return [];
  return name
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Za-z])(\d)/g, '$1 $2')
    .replace(/(\d)([A-Za-z])/g, '$1 $2')
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .map(t => t.toLowerCase());
}

function matchesCategory(name, keywords) {
  const toks = tokenizeName(name);
  return toks.some(t => keywords.includes(t));
}

function sideOfName(name) {
  const toks = tokenizeName(name);
  if (toks.some(t => JOINT_NAME_PATTERNS.left.includes(t))) return 'left';
  if (toks.some(t => JOINT_NAME_PATTERNS.right.includes(t))) return 'right';
  return null;
}

/**
 * Classify the union of all skins' joints in a document.
 * Returns {
 *   joints: Set<nodeIdx>,
 *   armChains: { left: Set, right: Set, unsided: Set },  // seeds + joint descendants
 *   armAll: Set,                // union of all arm chains
 *   headTail: Set,              // head/tail matched joints + descendants
 *   torso: Set,                 // torso-name-matched joints (e.iv)
 *   rootJoint: nodeIdx | null,  // named root/hips/pelvis, else top joint
 * }
 */
function classifyJoints(doc) {
  const joints = new Set();
  for (const skin of doc.json.skins || []) for (const j of skin.joints || []) joints.add(j);
  const parents = doc.parents();

  const descendantsWithin = seed => {
    const out = new Set([seed]);
    const stack = [seed];
    while (stack.length) {
      const n = stack.pop();
      for (const c of doc.json.nodes[n]?.children || []) {
        if (joints.has(c) && !out.has(c)) { out.add(c); stack.push(c); }
      }
    }
    return out;
  };

  const armChains = { left: new Set(), right: new Set(), unsided: new Set() };
  const headTail = new Set();
  const torso = new Set();
  let rootJoint = null;

  for (const j of joints) {
    const name = (doc.json.nodes[j] || {}).name || '';
    if (matchesCategory(name, JOINT_NAME_PATTERNS.arm)) {
      const side = sideOfName(name) || 'unsided';
      for (const d of descendantsWithin(j)) armChains[side].add(d);
    }
    if (matchesCategory(name, JOINT_NAME_PATTERNS.head) || matchesCategory(name, JOINT_NAME_PATTERNS.tail)) {
      for (const d of descendantsWithin(j)) headTail.add(d);
    }
    if (matchesCategory(name, JOINT_NAME_PATTERNS.torso)) torso.add(j);
    if (rootJoint === null && matchesCategory(name, JOINT_NAME_PATTERNS.root)) rootJoint = j;
  }

  if (rootJoint === null) {
    // Fallback: the skeleton's top joint — a joint whose parent is not a joint.
    for (const j of joints) {
      const p = parents.get(j);
      if (p === undefined || !joints.has(p)) { rootJoint = j; break; }
    }
  }

  const armAll = new Set([...armChains.left, ...armChains.right, ...armChains.unsided]);
  return { joints, armChains, armAll, headTail, torso, rootJoint };
}

/* ========================================================================== *
 * Animation sampling (STEP / LINEAR / CUBICSPLINE per glTF 2.0).
 * ========================================================================== */

function buildAnimSampler(doc, sampler, pathName) {
  const input = doc.accessor(sampler.input);
  const output = doc.accessor(sampler.output);
  const interp = sampler.interpolation || 'LINEAR';
  const nc = pathName === 'rotation' ? 4 : pathName === 'weights' ? output.numComponents : 3;
  const times = input.data;
  const values = output.data;
  const keys = input.count;
  const per = interp === 'CUBICSPLINE' ? 3 : 1; // elements per key

  function valueAt(k, elem /*0=in,1=val,2=out*/) {
    const base = interp === 'CUBICSPLINE' ? (k * 3 + elem) * nc : k * nc;
    const v = new Array(nc);
    for (let c = 0; c < nc; c++) v[c] = values[base + c];
    return v;
  }

  return {
    duration: keys ? times[keys - 1] : 0,
    sample(t) {
      if (keys === 0) return null;
      if (t <= times[0]) return valueAt(0, 1);
      if (t >= times[keys - 1]) return valueAt(keys - 1, 1);
      let lo = 0, hi = keys - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (times[mid] <= t) lo = mid; else hi = mid;
      }
      const t0 = times[lo], t1 = times[hi];
      const dt = t1 - t0 || 1e-9;
      const u = (t - t0) / dt;
      if (interp === 'STEP') return valueAt(lo, 1);
      const v0 = valueAt(lo, 1), v1 = valueAt(hi, 1);
      if (interp === 'CUBICSPLINE') {
        const b0 = valueAt(lo, 2), a1 = valueAt(hi, 0);
        const u2 = u * u, u3 = u2 * u;
        const s0 = 2 * u3 - 3 * u2 + 1, s1 = dt * (u3 - 2 * u2 + u);
        const s2 = -2 * u3 + 3 * u2, s3 = dt * (u3 - u2);
        const out = new Array(nc);
        for (let c = 0; c < nc; c++) out[c] = s0 * v0[c] + s1 * b0[c] + s2 * v1[c] + s3 * a1[c];
        return pathName === 'rotation' ? quatNormalize(out) : out;
      }
      // LINEAR
      if (pathName === 'rotation') return quatSlerp(quatNormalize(v0), quatNormalize(v1), u);
      const out = new Array(nc);
      for (let c = 0; c < nc; c++) out[c] = v0[c] + (v1[c] - v0[c]) * u;
      return out;
    },
  };
}

/** Rest-pose TRS of a node (decomposes nothing: matrix nodes are kept as matrix). */
function nodeRest(node) {
  return {
    matrix: node.matrix || null,
    t: node.translation ? [...node.translation] : [0, 0, 0],
    r: node.rotation ? quatNormalize([...node.rotation]) : [0, 0, 0, 1],
    s: node.scale ? [...node.scale] : [1, 1, 1],
  };
}

/**
 * Build a per-frame pose evaluator for one animation.
 * Morph-target (`weights`) channels are ignored: the wave checks measure
 * joint rotations and skinned vertex positions; morph animation adds no
 * skeletal motion. (Documented behavior.)
 */
function buildPoseSampler(doc, anim) {
  const tracks = new Map(); // nodeIdx -> { translation?, rotation?, scale? }
  let duration = 0;
  for (const ch of anim.channels || []) {
    const target = ch.target || {};
    if (target.node === undefined) continue;
    if (!['translation', 'rotation', 'scale'].includes(target.path)) continue;
    const s = buildAnimSampler(doc, anim.samplers[ch.sampler], target.path);
    duration = Math.max(duration, s.duration);
    if (!tracks.has(target.node)) tracks.set(target.node, {});
    tracks.get(target.node)[target.path] = s;
  }

  const nodes = doc.json.nodes || [];
  const rest = nodes.map(nodeRest);

  function localTRS(i, t) {
    const tr = tracks.get(i);
    const r = rest[i];
    if (!tr) return r;
    return {
      matrix: null, // animated nodes use TRS per glTF spec
      t: tr.translation ? tr.translation.sample(t) : r.t,
      r: tr.rotation ? quatNormalize(tr.rotation.sample(t)) : r.r,
      s: tr.scale ? tr.scale.sample(t) : r.s,
    };
  }

  /** Global (world) matrices for all nodes at time t. */
  function globalsAt(t) {
    const globals = new Array(nodes.length).fill(null);
    const parents = doc.parents();
    const localMat = i => {
      const l = localTRS(i, t);
      return l.matrix && !tracks.has(i) ? l.matrix : mat4FromTRS(l.t, l.r, l.s);
    };
    const compute = i => {
      if (globals[i]) return globals[i];
      const p = parents.get(i);
      const m = localMat(i);
      globals[i] = p === undefined ? m : mat4Mul(compute(p), m);
      return globals[i];
    };
    for (let i = 0; i < nodes.length; i++) compute(i);
    return globals;
  }

  return { duration, tracks, rest, localTRS, globalsAt };
}

/* ========================================================================== *
 * Scene traversal + triangle counting.
 * ========================================================================== */

/**
 * Node instances considered "rendered": every node reachable from ANY scene
 * in `scenes` (deduplicated — each node counts once even if several scenes
 * share roots). If the file declares no scenes, ALL nodes are counted.
 */
function renderedNodeSet(doc) {
  const json = doc.json;
  const out = new Set();
  const scenes = json.scenes || [];
  if (!scenes.length) {
    (json.nodes || []).forEach((_, i) => out.add(i));
    return out;
  }
  const stack = [];
  for (const sc of scenes) for (const r of sc.nodes || []) stack.push(r);
  while (stack.length) {
    const n = stack.pop();
    if (out.has(n)) continue;
    out.add(n);
    for (const c of json.nodes[n]?.children || []) stack.push(c);
  }
  return out;
}

/**
 * Triangles contributed by one primitive (glTF `mode`):
 *   4 TRIANGLES        -> count / 3
 *   5 TRIANGLE_STRIP   -> count - 2
 *   6 TRIANGLE_FAN     -> count - 2
 *   0-3 points/lines   -> 0
 * `count` is the indices accessor count when indexed, else the POSITION
 * accessor count. Morph targets deform the base primitive and add NO
 * triangles.
 */
function primitiveTriangles(doc, prim) {
  const mode = prim.mode === undefined ? 4 : prim.mode;
  let count = 0;
  if (prim.indices !== undefined) count = doc.json.accessors[prim.indices].count;
  else if (prim.attributes?.POSITION !== undefined) count = doc.json.accessors[prim.attributes.POSITION].count;
  if (mode === 4) return Math.floor(count / 3);
  if (mode === 5 || mode === 6) return Math.max(0, count - 2);
  return 0;
}

function countTriangles(doc) {
  const rendered = renderedNodeSet(doc);
  let total = 0;
  const perMesh = [];
  for (const n of rendered) {
    const node = doc.json.nodes[n];
    if (!node || node.mesh === undefined) continue;
    const mesh = doc.json.meshes[node.mesh];
    let t = 0;
    for (const prim of mesh.primitives || []) t += primitiveTriangles(doc, prim);
    total += t;
    perMesh.push({ node: doc.nodeName(n), mesh: mesh.name || `mesh#${node.mesh}`, triangles: t });
  }
  return { total, perMesh };
}

/* ========================================================================== *
 * Model height (bounding box from POSITION min/max).
 * ========================================================================== */

/**
 * Model height = Y extent of the union of all rendered mesh primitives'
 * POSITION bounds (accessor min/max when present, else computed). glTF is
 * Y-up by convention; if the Y extent is degenerate (~0) the largest axis
 * extent is used instead. Bounds are taken in mesh space (skinned vertices
 * are defined in mesh space per the glTF skinning equations).
 */
function modelHeight(doc) {
  const rendered = renderedNodeSet(doc);
  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const n of rendered) {
    const node = doc.json.nodes[n];
    if (!node || node.mesh === undefined) continue;
    for (const prim of doc.json.meshes[node.mesh].primitives || []) {
      const pi = prim.attributes?.POSITION;
      if (pi === undefined) continue;
      const acc = doc.json.accessors[pi];
      let lo = acc.min, hi = acc.max;
      if (!lo || !hi) {
        const a = doc.accessor(pi);
        lo = [Infinity, Infinity, Infinity]; hi = [-Infinity, -Infinity, -Infinity];
        for (let v = 0; v < a.count; v++) {
          for (let c = 0; c < 3; c++) {
            const val = a.data[v * 3 + c];
            if (val < lo[c]) lo[c] = val;
            if (val > hi[c]) hi[c] = val;
          }
        }
      }
      for (let c = 0; c < 3; c++) {
        if (lo[c] < min[c]) min[c] = lo[c];
        if (hi[c] > max[c]) max[c] = hi[c];
      }
    }
  }
  if (min[0] === Infinity) return 0;
  const ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return ext[1] > 1e-9 ? ext[1] : Math.max(...ext);
}

/* ========================================================================== *
 * Check implementations.
 * ========================================================================== */

function checkSkins(docs) {
  const rows = docs.map(d => ({ file: d.label, skins: (d.json.skins || []).length }));
  const pass = rows.every(r => r.skins >= 1);
  return {
    id: 'skin', title: `>= 1 skin`, pass,
    measured: rows.map(r => `${r.file}: ${r.skins} skin(s)`).join('; '),
    details: { rows },
  };
}

function checkClips(docs, layout) {
  const found = {}; // canonical -> {file, clip}
  const allNames = [];
  for (const d of docs) {
    for (const a of d.json.animations || []) {
      if (a.name) allNames.push(`${d.label}:${a.name}`);
    }
  }
  for (const group of REQUIRED_CLIPS) {
    for (const d of docs) {
      // in multi layout, the file dedicated to a clip must carry it; in
      // single layout any doc (there is one) may carry it.
      for (const a of d.json.animations || []) {
        if (group.includes(a.name) && !found[group[0]]) {
          found[group[0]] = { file: d.label, clip: a.name };
        }
      }
    }
  }
  const missing = REQUIRED_CLIPS.filter(g => !found[g[0]]).map(g => g.join('|'));
  const pass = missing.length === 0;
  return {
    id: 'clips', title: 'clips idle / jump / walk|run / wave present by exact name', pass,
    measured: pass
      ? REQUIRED_CLIPS.map(g => `${g[0]}: "${found[g[0]].clip}"${layout === 'multi' ? ` (${found[g[0]].file})` : ''}`).join('; ')
      : `missing: ${missing.join(', ')} — animations found: [${allNames.join(', ') || 'none'}]`,
    details: { found, missing, allNames },
  };
}

function checkTriangles(docs) {
  const rows = docs.map(d => ({ file: d.label, ...countTriangles(d) }));
  const worst = Math.max(...rows.map(r => r.total));
  const pass = worst <= THRESHOLDS.MAX_TRIANGLES;
  return {
    id: 'triangles',
    title: `rendered triangles <= ${THRESHOLDS.MAX_TRIANGLES}`,
    pass,
    measured: rows.map(r => `${r.file}: ${r.total}`).join('; ') +
      ` (limit ${THRESHOLDS.MAX_TRIANGLES}; per node instance, all scenes, morph targets add none)`,
    details: { rows, worst },
  };
}

function imageDataPresent(doc, imageIdx) {
  const img = (doc.json.images || [])[imageIdx];
  if (!img) return { present: false, why: `image ${imageIdx} missing` };
  if (img.bufferView !== undefined) {
    try {
      const bv = doc.json.bufferViews[img.bufferView];
      doc.buffer(bv.buffer);
      return { present: (bv.byteLength || 0) > 0, why: `embedded bufferView ${img.bufferView} (${bv.byteLength} bytes)` };
    } catch (e) { return { present: false, why: String(e.message) }; }
  }
  if (img.uri !== undefined) {
    if (img.uri.startsWith('data:')) return { present: true, why: 'data: URI' };
    if (!doc.baseDir) return { present: false, why: `external uri "${img.uri}" not resolvable from memory` };
    const p = path.resolve(doc.baseDir, decodeURIComponent(img.uri));
    return { present: fs.existsSync(p), why: `uri "${img.uri}"${fs.existsSync(p) ? '' : ' NOT FOUND'}` };
  }
  return { present: false, why: 'image has neither bufferView nor uri' };
}

function checkBaseColorTexture(docs) {
  const rows = [];
  let pass = true;
  for (const doc of docs) {
    let ok = false, detail = 'no skinned mesh primitive with a baseColorTexture-bearing material';
    const rendered = renderedNodeSet(doc);
    for (const n of rendered) {
      const node = doc.json.nodes[n];
      if (!node || node.mesh === undefined || node.skin === undefined) continue;
      for (const prim of doc.json.meshes[node.mesh].primitives || []) {
        if (prim.material === undefined) continue;
        const mat = doc.json.materials[prim.material];
        const bct = mat?.pbrMetallicRoughness?.baseColorTexture;
        if (!bct) continue;
        const tex = (doc.json.textures || [])[bct.index];
        const source = tex?.source ?? tex?.extensions?.KHR_texture_basisu?.source;
        if (source === undefined) { detail = `texture ${bct.index} has no image source`; continue; }
        const imgCheck = imageDataPresent(doc, source);
        if (imgCheck.present) {
          ok = true;
          detail = `material "${mat.name || prim.material}" on skinned node "${doc.nodeName(n)}" -> ${imgCheck.why}`;
          break;
        }
        detail = `image data missing: ${imgCheck.why}`;
      }
      if (ok) break;
    }
    rows.push({ file: doc.label, ok, detail });
    if (!ok) pass = false;
  }
  return {
    id: 'baseColorTexture',
    title: 'PBR baseColorTexture with present image data on a skinned mesh material',
    pass,
    measured: rows.map(r => `${r.file}: ${r.ok ? 'yes' : 'NO'} — ${r.detail}`).join('; '),
    details: { rows },
  };
}

/* ------------------------------- wave check ------------------------------ */

function countOscillations(signal) {
  const n = signal.length;
  if (n < 3) return { oscillations: 0, halfSwings: 0, crossings: 0 };
  let lo = Infinity, hi = -Infinity, mean = 0;
  for (const v of signal) { if (v < lo) lo = v; if (v > hi) hi = v; mean += v; }
  mean /= n;
  const range = hi - lo;
  const h = Math.max(THRESHOLDS.WAVE_OSC_MIN_SWING_DEG, range * THRESHOLDS.WAVE_OSC_MIN_SWING_FRAC);

  // Half-swings: monotone segments spanning >= h (hysteresis extrema walk).
  let halfSwings = 0, dir = 0, ref = signal[0];
  for (let i = 1; i < n; i++) {
    const d = signal[i] - ref;
    if (dir === 0) {
      if (Math.abs(d) >= h) { dir = Math.sign(d); halfSwings++; ref = signal[i]; }
    } else if (Math.sign(d) === dir || d === 0) {
      if (Math.sign(d) === dir) ref = signal[i];
    } else if (Math.abs(d) >= h) {
      dir = -dir; halfSwings++; ref = signal[i];
    }
  }

  // Zero-crossings about the mean (with a small dead-zone of h/4).
  let crossings = 0, state = 0;
  const dead = h / 4;
  for (const v of signal) {
    const s = v > mean + dead ? 1 : v < mean - dead ? -1 : 0;
    if (s !== 0) {
      if (state !== 0 && s !== state) crossings++;
      state = s;
    }
  }

  return {
    oscillations: Math.max(Math.floor(halfSwings / 2), Math.floor(crossings / 2)),
    halfSwings, crossings,
  };
}

function percentile(sortedAsc, p) {
  if (!sortedAsc.length) return 0;
  const idx = Math.min(sortedAsc.length - 1, Math.floor(p * (sortedAsc.length - 1) + 0.5));
  return sortedAsc[idx];
}

/**
 * The wave-arm check (e). See THRESHOLDS for every limit. Sampling:
 * WAVE_SAMPLES_PER_SECOND over the clip duration (>= 2 samples, capped at
 * WAVE_MAX_SAMPLES), with per-sampler STEP/LINEAR/CUBICSPLINE interpolation.
 */
function checkWaveArm(docs) {
  const sub = {};
  const fail = (msg) => ({
    id: 'waveArm', title: 'wave-arm check (rotation, oscillation, stability, co-deformation)',
    pass: false, measured: msg, details: { subChecks: sub, error: msg },
  });

  // Locate the doc + clip named "wave".
  let doc = null, anim = null;
  for (const d of docs) {
    for (const a of d.json.animations || []) {
      if (a.name === 'wave') { doc = d; anim = a; break; }
    }
    if (doc) break;
  }
  if (!doc) return fail('no animation named "wave" found in any file');

  const cls = classifyJoints(doc);
  if (!cls.joints.size) return fail('no skin joints in the file carrying "wave"');
  const jointNames = [...cls.joints].map(j => doc.nodeName(j));
  const height = modelHeight(doc);
  if (!(height > 0)) return fail('model height is zero — cannot scale thresholds');

  const chains = Object.entries(cls.armChains).filter(([, s]) => s.size > 0);
  if (!chains.length) {
    return fail(`no arm joints matched JOINT_NAME_PATTERNS.arm among skin joints: [${jointNames.join(', ')}]`);
  }

  const pose = buildPoseSampler(doc, anim);
  if (!(pose.duration > 0)) return fail('"wave" clip has zero duration');
  const nSamples = Math.min(
    THRESHOLDS.WAVE_MAX_SAMPLES,
    Math.max(2, Math.ceil(pose.duration * THRESHOLDS.WAVE_SAMPLES_PER_SECOND) + 1),
  );
  const times = Array.from({ length: nSamples }, (_, i) => (i / (nSamples - 1)) * pose.duration);

  // Per-frame local rotations for every joint; rotation vectors relative to
  // rest pose AND relative to the clip's first frame, with quaternion
  // double-cover continuity (flip to keep dot >= 0 with previous sample).
  const jointList = [...cls.joints];
  const relRestVec = new Map(); // j -> Array<[x,y,z]> (radians)
  const relStartVec = new Map();
  {
    const prevRest = new Map(), prevStart = new Map();
    const startRot = new Map();
    for (const j of jointList) { relRestVec.set(j, []); relStartVec.set(j, []); }
    for (let f = 0; f < nSamples; f++) {
      for (const j of jointList) {
        const l = pose.localTRS(j, times[f]);
        const q = l.r;
        if (f === 0) startRot.set(j, q);
        let qr = quatMul(quatConj(pose.rest[j].r), q);
        let qs = quatMul(quatConj(startRot.get(j)), q);
        const pr = prevRest.get(j), ps = prevStart.get(j);
        if (pr && quatDot(qr, pr) < 0) qr = qr.map(v => -v);
        if (ps && quatDot(qs, ps) < 0) qs = qs.map(v => -v);
        prevRest.set(j, qr); prevStart.set(j, qs);
        relRestVec.get(j).push(quatToRotVec(qr));
        relStartVec.get(j).push(quatToRotVec(qs));
      }
    }
  }
  const angleDeg = v => Math.hypot(v[0], v[1], v[2]) * DEG;

  // (i) peak arm rotation from rest, per chain.
  let best = null; // { side, joint, name, peakRestDeg, peakStartDeg }
  const chainStats = [];
  for (const [side, set] of chains) {
    let chainBest = null;
    for (const j of set) {
      const peakRest = Math.max(...relRestVec.get(j).map(angleDeg));
      const peakStart = Math.max(...relStartVec.get(j).map(angleDeg));
      if (!chainBest || peakRest > chainBest.peakRestDeg) {
        chainBest = { side, joint: j, name: doc.nodeName(j), peakRestDeg: peakRest, peakStartDeg: peakStart };
      }
    }
    chainStats.push(chainBest);
    if (!best || chainBest.peakRestDeg > best.peakRestDeg) best = chainBest;
  }
  sub.armRotation = {
    pass: best.peakRestDeg >= THRESHOLDS.WAVE_MIN_ARM_PEAK_DEG,
    measured: `${best.side} arm, joint "${best.name}": peak ${best.peakRestDeg.toFixed(1)} deg from rest ` +
      `(${best.peakStartDeg.toFixed(1)} deg from clip start); threshold >= ${THRESHOLDS.WAVE_MIN_ARM_PEAK_DEG} deg`,
    peakRestDeg: best.peakRestDeg, peakStartDeg: best.peakStartDeg,
    arm: best.side, joint: best.name,
    perChain: chainStats.map(c => ({ side: c.side, joint: c.name, peakRestDeg: +c.peakRestDeg.toFixed(2) })),
  };

  // (ii) oscillations of the winning arm chain: best count over its joints
  // and over each rotation-vector component (x/y/z, degrees), measured
  // relative to the clip start (captures the wave-back-and-forth even when
  // the raise itself is a one-way move).
  const winningChain = cls.armChains[best.side];
  let osc = { oscillations: 0, halfSwings: 0, crossings: 0 }, oscJoint = best.name, oscAxis = 'x';
  for (const j of winningChain) {
    const vecs = relStartVec.get(j);
    for (let axis = 0; axis < 3; axis++) {
      const sig = vecs.map(v => v[axis] * DEG);
      const c = countOscillations(sig);
      if (c.oscillations > osc.oscillations) {
        osc = c; oscJoint = doc.nodeName(j); oscAxis = 'xyz'[axis];
      }
    }
  }
  sub.oscillations = {
    pass: osc.oscillations >= THRESHOLDS.WAVE_MIN_OSCILLATIONS,
    measured: `${osc.oscillations} oscillation(s) on joint "${oscJoint}" axis ${oscAxis} ` +
      `(${osc.halfSwings} half-swings, ${osc.crossings} mean-crossings); threshold >= ${THRESHOLDS.WAVE_MIN_OSCILLATIONS}`,
    oscillations: osc.oscillations, halfSwings: osc.halfSwings, crossings: osc.crossings,
    joint: oscJoint, axis: oscAxis,
  };

  // Per-frame global matrices (needed by iii and v).
  const globalsPerFrame = times.map(t => pose.globalsAt(t));

  // (iii) root/hips world-translation range (bbox diagonal of sampled root
  // positions) <= 5% of model height.
  {
    const j = cls.rootJoint;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const g of globalsPerFrame) {
      const m = g[j];
      const p = [m[12], m[13], m[14]];
      for (let c = 0; c < 3; c++) { if (p[c] < lo[c]) lo[c] = p[c]; if (p[c] > hi[c]) hi[c] = p[c]; }
    }
    const range = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const frac = range / height;
    sub.rootTranslation = {
      pass: frac <= THRESHOLDS.WAVE_MAX_ROOT_TRANSLATION_FRAC,
      measured: `root joint "${doc.nodeName(j)}" translation range ${range.toFixed(4)} units = ` +
        `${(frac * 100).toFixed(2)}% of model height ${height.toFixed(3)}; threshold <= ${THRESHOLDS.WAVE_MAX_ROOT_TRANSLATION_FRAC * 100}%`,
      rangeUnits: range, rangeFrac: frac, rootJoint: doc.nodeName(j), modelHeight: height,
    };
  }

  // (iv) torso joints' local rotation range (max angle vs the clip's first
  // frame) <= 15 deg.
  {
    let worst = { deg: -1, name: '(none)' };
    for (const j of cls.torso) {
      const peak = Math.max(...relStartVec.get(j).map(angleDeg));
      if (peak > worst.deg) worst = { deg: peak, name: doc.nodeName(j) };
    }
    worst.deg = Math.max(0, worst.deg);
    sub.torsoRotation = {
      pass: worst.deg <= THRESHOLDS.WAVE_MAX_TORSO_ROTATION_DEG,
      measured: cls.torso.size
        ? `max torso rotation ${worst.deg.toFixed(1)} deg on "${worst.name}" ` +
          `(over ${cls.torso.size} torso joint(s)); threshold <= ${THRESHOLDS.WAVE_MAX_TORSO_ROTATION_DEG} deg`
        : 'no torso-named joints found — treated as 0 deg (pattern list may need extending)',
      maxDeg: worst.deg, joint: worst.name, torsoJointCount: cls.torso.size,
    };
  }

  // (v) torso co-deformation via linear blend skinning.
  {
    // world-space v(t) = sum_i w_i * G_joint(t) * IBM_i * v  (glTF skinning)
    let torsoVertexCount = 0;
    const perVertexMax = [];
    let skinnedPrims = 0;
    const rendered = renderedNodeSet(doc);
    for (const n of rendered) {
      const node = doc.json.nodes[n];
      if (!node || node.mesh === undefined || node.skin === undefined) continue;
      const skin = doc.json.skins[node.skin];
      const jointsArr = skin.joints;
      const ibmAcc = skin.inverseBindMatrices !== undefined ? doc.accessor(skin.inverseBindMatrices) : null;
      const ibm = jointsArr.map((_, k) => {
        if (!ibmAcc) return mat4Identity();
        return Array.from(ibmAcc.data.slice(k * 16, k * 16 + 16));
      });
      // Precompute per frame: jointMat[k] = G[joint k] * IBM[k]
      const jointMats = globalsPerFrame.map(g => jointsArr.map((j, k) => mat4Mul(g[j], ibm[k])));

      const isTorsoDominant = k => {
        const j = jointsArr[k];
        return j !== undefined && !cls.armAll.has(j) && !cls.headTail.has(j);
      };

      for (const prim of doc.json.meshes[node.mesh].primitives || []) {
        const at = prim.attributes || {};
        if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) continue;
        skinnedPrims++;
        const pos = doc.accessor(at.POSITION);
        const jnt = doc.accessor(at.JOINTS_0);
        const wgt = doc.accessor(at.WEIGHTS_0);
        for (let v = 0; v < pos.count; v++) {
          // dominant joint
          let domK = -1, domW = -1;
          for (let c = 0; c < 4; c++) {
            const w = wgt.data[v * 4 + c];
            if (w > domW) { domW = w; domK = jnt.data[v * 4 + c]; }
          }
          if (domW <= 0 || !isTorsoDominant(domK)) continue;
          torsoVertexCount++;
          const p = [pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2]];
          let first = null, maxD = 0;
          for (let f = 0; f < nSamples; f++) {
            const jm = jointMats[f];
            let ox = 0, oy = 0, oz = 0;
            for (let c = 0; c < 4; c++) {
              const w = wgt.data[v * 4 + c];
              if (w === 0) continue;
              const k = jnt.data[v * 4 + c];
              const q = mat4TransformPoint(jm[k], p);
              ox += w * q[0]; oy += w * q[1]; oz += w * q[2];
            }
            if (f === 0) first = [ox, oy, oz];
            else {
              const d = Math.hypot(ox - first[0], oy - first[1], oz - first[2]);
              if (d > maxD) maxD = d;
            }
          }
          perVertexMax.push(maxD);
        }
      }
    }
    if (!skinnedPrims) {
      sub.torsoCoDeformation = { pass: false, measured: 'no skinned primitives (POSITION+JOINTS_0+WEIGHTS_0) found', p95Frac: null };
    } else if (!torsoVertexCount) {
      sub.torsoCoDeformation = { pass: false, measured: 'no vertices with a torso-dominant joint found — cannot verify torso stability', p95Frac: null };
    } else {
      perVertexMax.sort((a, b) => a - b);
      const p95 = percentile(perVertexMax, 0.95);
      const frac = p95 / height;
      sub.torsoCoDeformation = {
        pass: frac <= THRESHOLDS.WAVE_MAX_TORSO_P95_DISPLACEMENT_FRAC,
        measured: `p95 torso-vertex displacement ${p95.toFixed(4)} units = ${(frac * 100).toFixed(2)}% of model height ` +
          `(${torsoVertexCount} torso-dominant vertices, ${nSamples} frames); threshold <= ${THRESHOLDS.WAVE_MAX_TORSO_P95_DISPLACEMENT_FRAC * 100}%`,
        p95Units: p95, p95Frac: frac, torsoVertexCount, samples: nSamples,
      };
    }
  }

  const pass = Object.values(sub).every(s => s.pass);
  const summary = `arm=${best.side} ("${best.name}"), peak ${best.peakRestDeg.toFixed(1)} deg, ` +
    `${sub.oscillations.oscillations} oscillations, root ${(sub.rootTranslation.rangeFrac * 100).toFixed(2)}% h, ` +
    `torso rot ${sub.torsoRotation.maxDeg.toFixed(1)} deg, torso p95 ` +
    `${sub.torsoCoDeformation.p95Frac === null ? 'n/a' : (sub.torsoCoDeformation.p95Frac * 100).toFixed(2) + '% h'}`;
  return {
    id: 'waveArm', title: 'wave-arm check (rotation, oscillation, stability, co-deformation)',
    pass, measured: summary,
    details: { subChecks: sub, clipDuration: pose.duration, samples: nSamples, wavedArm: best.side, modelHeight: height },
  };
}

/* ========================================================================== *
 * Layout resolution + top-level validate().
 * ========================================================================== */

function loadDocFromDisk(file) {
  const bytes = new Uint8Array(fs.readFileSync(file));
  const { json, bin } = parseGLB(bytes);
  return new Doc(json, bin, path.dirname(file), path.basename(file));
}

function resolveLayout(p) {
  // p may be: a .glb file, an asset dir, or a repo root containing assets/kitty.
  if (fs.existsSync(p) && fs.statSync(p).isFile()) {
    return { layout: 'single', files: [p] };
  }
  const dirs = [];
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) {
    dirs.push(p, path.join(p, ASSET_DIR));
  } else {
    dirs.push(p); // report the missing path as-is
  }
  for (const dir of dirs) {
    const single = path.join(dir, SINGLE_FILE);
    if (fs.existsSync(single)) return { layout: 'single', files: [single] };
  }
  for (const dir of dirs) {
    const chosen = [];
    let all = true;
    for (const [clip, candidates] of Object.entries(MULTI_FILES)) {
      const hit = candidates.map(c => path.join(dir, c)).find(f => fs.existsSync(f));
      if (hit) chosen.push({ clip, file: hit });
      else all = false;
    }
    if (all) return { layout: 'multi', files: chosen.map(c => c.file) };
    if (chosen.length) {
      return {
        layout: 'missing',
        error: `partial multi-file layout in ${dir}: found [${chosen.map(c => path.basename(c.file)).join(', ')}] ` +
          `but not all four clips`,
      };
    }
  }
  return {
    layout: 'missing',
    error: `no kitty asset found: looked for ${SINGLE_FILE} or the four-file layout ` +
      `(${Object.values(MULTI_FILES).flat().join(', ')}) under: ${dirs.join(' , ')}`,
  };
}

/**
 * validate(options) -> report
 *   options.path  — .glb file, asset dir, or repo root (default `assets/kitty`).
 *   options.files — in-memory bytes instead of disk:
 *                   { single: Uint8Array } or
 *                   { idle, jump, walk|run, wave: Uint8Array }.
 * report = { ok, layout, files: [labels], checks: [{id,title,pass,measured,details}], error? }
 */
export function validate(options = {}) {
  let docs = [];
  let layout, fileLabels = [];

  try {
    if (options.files) {
      if (options.files.single) {
        layout = 'single';
        const { json, bin } = parseGLB(options.files.single);
        docs = [new Doc(json, bin, null, 'kitty.glb')];
      } else {
        layout = 'multi';
        const wanted = [['idle'], ['jump'], ['walk', 'run'], ['wave']];
        for (const group of wanted) {
          const key = group.find(k => options.files[k]);
          if (!key) {
            return {
              ok: false, layout: 'missing', files: [],
              error: `in-memory multi layout missing "${group.join('|')}" file`, checks: [],
            };
          }
          const { json, bin } = parseGLB(options.files[key]);
          docs.push(new Doc(json, bin, null, `kitty-${key}.glb`));
        }
      }
    } else {
      const p = path.resolve(options.path || ASSET_DIR);
      const res = resolveLayout(p);
      if (res.layout === 'missing') {
        return { ok: false, layout: 'missing', files: [], error: res.error, checks: [] };
      }
      layout = res.layout;
      docs = res.files.map(loadDocFromDisk);
    }
    fileLabels = docs.map(d => d.label);
  } catch (e) {
    return { ok: false, layout: 'error', files: fileLabels, error: `failed to parse asset: ${e.message}`, checks: [] };
  }

  const checks = [];
  const guard = fn => {
    try { return fn(); }
    catch (e) {
      return { id: 'internal', title: 'check crashed', pass: false, measured: String(e.stack || e), details: {} };
    }
  };
  checks.push(guard(() => checkSkins(docs)));
  checks.push(guard(() => checkClips(docs, layout)));
  checks.push(guard(() => checkTriangles(docs)));
  checks.push(guard(() => checkBaseColorTexture(docs)));
  checks.push(guard(() => checkWaveArm(docs)));

  return { ok: checks.every(c => c.pass), layout, files: fileLabels, checks };
}

/* ========================================================================== *
 * CLI
 * ========================================================================== */

function printReport(report) {
  const B = s => `\x1b[1m${s}\x1b[0m`;
  const G = s => `\x1b[32m${s}\x1b[0m`;
  const R = s => `\x1b[31m${s}\x1b[0m`;
  const color = process.stdout.isTTY;
  const pass = s => (color ? G(s) : s);
  const failc = s => (color ? R(s) : s);

  console.log(`kitty asset validation — layout: ${report.layout}` +
    (report.files.length ? ` [${report.files.join(', ')}]` : ''));
  if (report.error) {
    console.log(failc(`FAIL  ${report.error}`));
    return;
  }
  const wide = Math.max(...report.checks.map(c => c.title.length));
  for (const c of report.checks) {
    const badge = c.pass ? pass('PASS') : failc('FAIL');
    console.log(`${badge}  ${c.title.padEnd(wide)}  ${c.measured}`);
    if (c.id === 'waveArm' && c.details?.subChecks) {
      for (const [k, s] of Object.entries(c.details.subChecks)) {
        const b = s.pass ? pass(' ok ') : failc('FAIL');
        console.log(`      [${b}] ${k}: ${s.measured}`);
      }
    }
  }
  console.log(report.ok ? pass(color ? B('RESULT: PASS') : 'RESULT: PASS')
    : failc(color ? B('RESULT: FAIL') : 'RESULT: FAIL'));
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const target = process.argv[2];
  const report = validate({ path: target });
  printReport(report);
  process.exit(report.ok ? 0 : 1);
}
