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
 *   c. rendered triangles <= 20,000 (summed per node instance, all scenes,
 *      multiplied by EXT_mesh_gpu_instancing instance counts)
 *   d. a PBR baseColorTexture on a material used by a skinned mesh
 *      primitive, whose referenced image bytes are a REAL PNG/JPEG (magic
 *      bytes + parseable header with nonzero width/height)
 *   V1 wave arm: the waving side is chosen by IN-CLIP motion; >= 45 deg
 *      rotation vs the wave clip's FIRST frame (never the bind pose);
 *      the hand/wrist raised above the shoulder; >= 2 oscillations; the
 *      other arm stays still (<= 15 deg in-clip)
 *   V2 root/hips: EVERY root candidate (top joint, any root|hips|pelvis
 *      joint, topmost joint with an animated translation channel) each
 *      translates <= 5% of model height; torso rotation <= 15 deg
 *   V3' co-deformation (LBS): over ALL vertices OUTSIDE the waving arm's
 *      GEOMETRIC limb volume fixed at BIND (the limb side of the plane
 *      through the upper-arm joint perpendicular to the upper-arm bone,
 *      within 12% h of the arm's bind bone chain) — membership can NOT be
 *      changed by re-pointing weights; at most 0.5% may move > 3% h from
 *      the clip's first frame and none > 8% h
 *   V4 weight leakage: ZERO head/neck-dominated vertices carry > 0.2 total
 *      arm-chain weight (either arm)
 *   V4' reverse leakage: ZERO vertices DOMINATED by an arm-chain joint
 *      (clavicle included) inside the geometric head region (bind height
 *      above the topmost neck joint, within 45% h of the head bone star)
 *   V5 loop seams: `idle` and `walk`/`run` channels end where they start
 *      (<= 1% h translation / <= 2 deg rotation)
 *   V6 jump: at some frame every foot joint rises >= 8% h above its
 *      clip-start height
 *   V7 limb integrity: in EVERY clip (idle, jump, walk|run, wave), skinned
 *      geometry must ride the bones it wraps. For each vertex,
 *      d_bind = bind-pose distance to the nearest bone segment of the
 *      vertex's own weighted joints, and d_t = the same distance at each
 *      sampled frame with each candidate bone carried rigidly by its
 *      joint's motion; at most 0.2% of vertices may exceed
 *      min(1.5 * d_bind, d_bind + 3% h) + 3% h at any frame and none may
 *      exceed min(2 * d_bind, d_bind + 8% h) + 8% h (catches detached /
 *      doubled limb geometry riding along with a bone chain even when its
 *      dominant joint is exempt from V3')
 *   V7' mis-binding at BIND: V7 is invariant by construction for rigid
 *      single-joint binds (d_t = d_bind), so the BINDING itself is judged:
 *      <= 0.2% of vertices may be bound only to bones > 10% h away while
 *      another bone is < 10% h away and 2.5x closer; ZERO influences
 *      >= 0.35 weight bound > 20% h from their bones under the same
 *      nearness/dominance; every skin joint's scale (rest + animated) = 1
 *      within 0.05 per component
 *   V8 edge strain over ALL edges (no exemption for arm-touching
 *      triangles), in EVERY clip, vs BIND and vs the clip's first frame:
 *      <= 0.5% of edges past 1.5x (2x inside a limb volume), none past 3x
 *      (4x limb) and none opening > 3% h in absolute length
 *   V9 seam cracks: vertices co-located within 1e-5 at bind (UV-seam
 *      duplicates) stay within 0.5% h of each other in every clip
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
   *  mesh, once per node instance that references it, multiplied by the
   *  node's EXT_mesh_gpu_instancing instance count when present (spec-001:
   *  the budget is on RENDERED triangles). */
  MAX_TRIANGLES: 20_000,

  /** V1/V6. Animation sampling rate (samples per second over the full clip
   *  duration, correct per-sampler interpolation). */
  WAVE_SAMPLES_PER_SECOND: 60,
  /** Safety cap on total samples (caps pathological clip durations). */
  WAVE_MAX_SAMPLES: 3_601,

  /** V1. Minimum peak rotation (degrees) of at least one joint of the
   *  waving arm chain, measured RELATIVE TO THE WAVE CLIP'S FIRST FRAME —
   *  never the bind/rest pose (reviewer A1: on a rig whose bind pose is an
   *  arms-up "cheer", rest-relative angles score a limp arm at 100+ deg). */
  WAVE_MIN_ARM_PEAK_DEG: 45,

  /** V1. At the peak frame the waving arm's hand/wrist world height must be
   *  >= the same arm's shoulder (or upper-arm) joint world height plus this
   *  fraction of model height (a wave reads as a RAISED hand, not a wiggle
   *  at the hip — reviewer A1). */
  WAVE_HAND_ABOVE_SHOULDER_FRAC: 0.05,

  /** V1. Minimum oscillation count of the waving arm (an oscillation is a
   *  full back-and-forth: two direction reversals of the dominant rotation
   *  component, or two zero-crossings about its mean). */
  WAVE_MIN_OSCILLATIONS: 2,
  /** Hysteresis floor for oscillation detection: a monotone swing counts
   *  only if it spans at least this many degrees ... */
  WAVE_OSC_MIN_SWING_DEG: 10,
  /** ... or this fraction of the signal's full range, whichever is larger
   *  (rejects numeric jitter without missing small-but-real waves). */
  WAVE_OSC_MIN_SWING_FRAC: 0.2,

  /** V1. Maximum in-clip rotation (degrees vs the clip's first frame) of
   *  ANY joint of the NON-waving arm chain — one arm waves, the other
   *  hangs still. */
  WAVE_MAX_OTHER_ARM_DEG: 15,

  /** V2. Maximum world-translation range during `wave` of EVERY root
   *  candidate — the skeleton's top joint, every root|hips|pelvis-named
   *  joint, and the topmost joint with an animated translation channel —
   *  each as a fraction of model height (reviewer A6: checking one chosen
   *  "root" misses translation authored on `Hips` under a static `Root`). */
  WAVE_MAX_ROOT_TRANSLATION_FRAC: 0.05,

  /** V2. Maximum torso joint rotation during `wave` (degrees, largest angle
   *  between any sampled frame's local rotation and the clip's first
   *  frame). */
  WAVE_MAX_TORSO_ROTATION_DEG: 15,

  /** V3. Soft displacement cap: fraction of model height a non-waving-arm
   *  vertex may move (LBS, vs the clip's first frame) before it counts as
   *  an offender. */
  BODY_DISPLACEMENT_SOFT_FRAC: 0.03,
  /** V3. Maximum fraction of non-waving-arm vertices that may exceed the
   *  soft cap (reviewer A3: a p95 hides a 4% pocket of dragged vertices;
   *  0.5% tolerates only sub-visible noise). */
  BODY_DISPLACEMENT_MAX_OFFENDER_RATIO: 0.005,
  /** V3. Hard displacement cap: NO non-waving-arm vertex may move more than
   *  this fraction of model height. */
  BODY_DISPLACEMENT_HARD_FRAC: 0.08,

  /** V4. Maximum total arm-chain weight (either arm, summed over the 4
   *  influences) on a vertex whose DOMINANT joint is a head/neck joint.
   *  ZERO vertices may exceed it (reviewer A2: face vertices 40% weighted
   *  to a forearm deform visibly yet keep a head-dominant weight). */
  HEAD_ARM_WEIGHT_MAX: 0.2,

  /** V5. Loop seam: every translation channel of `idle` and `walk`/`run`
   *  must end within this fraction of model height of its start value. */
  LOOP_SEAM_TRANSLATION_FRAC: 0.01,
  /** V5. Loop seam: every rotation channel must end within this many
   *  degrees of its start value. */
  LOOP_SEAM_ROTATION_DEG: 2,
  /** V5 (documented extension): scale channels must end within this
   *  relative tolerance of their start value (the order pins translation
   *  and rotation; an unclosed scale loop pops just the same). */
  LOOP_SEAM_SCALE_REL: 0.01,

  /** V6. Minimum rise of the feet during `jump`: at some sampled frame,
   *  EVERY foot joint's world height must be at least this fraction of
   *  model height above that joint's clip-start height (all feet at once =
   *  actually airborne; a single swinging foot is not a jump). */
  JUMP_MIN_FOOT_RISE_FRAC: 0.08,

  /** V7. Limb integrity — geometry must ride the bones it wraps, in EVERY
   *  clip. For each vertex, d_bind = distance (bind pose) to the nearest
   *  bone segment of the vertex's OWN weighted joints; at each sampled
   *  frame, d_t = the same distance with each candidate bone carried
   *  rigidly by its joint's motion (equivalently: the skinned vertex is
   *  transported back to bind space through the joint's rigid delta before
   *  measuring). Geometry rigidly attached to any of its joints keeps
   *  d_t = d_bind exactly — muzzles, ears and tail tips hanging off a leaf
   *  joint are handled IN PRINCIPLE (a bone SEGMENT cannot represent the
   *  leaf's orientation, so plain nearest-segment distance false-fails a
   *  benign head turn by ~10% h; the joint's frame can). No joint is ever
   *  exempted by name. A vertex offends at a frame when
   *  d_t > min(d_bind * LIMB_SOFT_DBIND_SCALE,
   *            d_bind + LIMB_DRIFT_SOFT_FRAC * h) + LIMB_SOFT_BASE_FRAC * h.
   *  Maximum fraction of skinned vertices that may offend at ANY single
   *  sampled frame; 0.2% tolerates isolated numeric outliers on a dense
   *  mesh while a doubled limb (hundreds of vertices leaving their bones
   *  together) is orders of magnitude past it. */
  LIMB_OFFENDER_RATIO: 0.002,
  /** V7. Soft allowance: multiplier on d_bind. 1.5x lets skin slide/bulge
   *  around a bending joint (elbows, shoulders compress and stretch the
   *  wrap distance) proportionally to its wrap radius. */
  LIMB_SOFT_DBIND_SCALE: 1.5,
  /** V7. Soft allowance: the multiplicative slack is CAPPED at this
   *  fraction of model height — skin slide is bounded by tissue scale, not
   *  by how far the geometry already floats. Without the cap, a limb-shaped
   *  mass parked 25% h from every bone earns 12% h of free drift and a
   *  doubled forearm sails through the check. */
  LIMB_DRIFT_SOFT_FRAC: 0.03,
  /** V7. Soft allowance: absolute floor as a fraction of model height, so
   *  vertices that hug a bone (d_bind ~ 0) keep a realistic slack for
   *  volume-preserving deformation. Matches V3's 3% h soft cap. */
  LIMB_SOFT_BASE_FRAC: 0.03,
  /** V7. Hard cap: NO vertex may ever exceed
   *  min(d_bind * 2, d_bind + LIMB_DRIFT_HARD_FRAC * h) + 8% h. Anything
   *  past this is geometry visibly detached from its bones, whatever its
   *  weights. */
  LIMB_HARD_DBIND_SCALE: 2,
  /** V7. Hard cap on the multiplicative slack (fraction of model height). */
  LIMB_DRIFT_HARD_FRAC: 0.08,
  /** V7. Hard cap absolute part (fraction of model height). */
  LIMB_HARD_BASE_FRAC: 0.08,

  /* --- fix round 3: V3' geometric limb volume ---------------------------- */

  /** V3'/V8. Radius of the GEOMETRIC arm-limb volume, as a fraction of model
   *  height. A vertex belongs to an arm limb only if it (a) lies on the limb
   *  side of the plane through the upper-arm joint perpendicular to the
   *  upper-arm bone AND (b) is within this radius of the arm's bone chain
   *  (upper arm -> hand, bind pose). Set in principle: a chubby cartoon
   *  biped's arm (paw included) is at most ~12% of body height thick around
   *  its bones; anything farther out is torso/head/tail skin whatever its
   *  weights say. V3 previously classified "arm" by CURRENT dominant joint,
   *  so re-pointing torso vertices at the waving arm removed them from the
   *  co-deformation check (both round-3 probes proved it: a belly slab
   *  100% on RightArm PASSED). Geometry at bind cannot be re-pointed. */
  LIMB_RADIUS_FRAC: 0.12,

  /* --- fix round 3: V8 all-edge strain ------------------------------------ */

  /** V8. Soft edge-length ratio for edges NOT wholly inside a limb volume:
   *  an edge stretched past this (vs bind OR vs the clip's first frame)
   *  counts as an offender. Same 1.5x as the old V3 strain — but over ALL
   *  edges: the old check dropped every triangle touching a waving-arm
   *  vertex, which is exactly where the arm/torso boundary tears. */
  EDGE_SOFT_RATIO_BODY: 1.5,
  /** V8. Soft ratio for edges wholly INSIDE a limb volume. Looser because
   *  elbow/shoulder creases legitimately stretch compressed skin when the
   *  limb bends (crease edges are short and sit in the fold). */
  EDGE_SOFT_RATIO_LIMB: 2.0,
  /** V8. Maximum fraction of ALL edges (per clip) that may exceed their
   *  soft ratio. 0.5% tolerates isolated crease/noise edges; a torn
   *  boundary ring (dozens to hundreds of edges) is far past it. */
  EDGE_OFFENDER_RATIO: 0.005,
  /** V8. Hard ratio cap for non-limb edges: NO edge may stretch past this
   *  (vs bind or clip start). 3x skin stretch is visible shredding. */
  EDGE_HARD_RATIO_BODY: 3.0,
  /** V8. Hard ratio cap for edges wholly inside a limb volume (justified:
   *  a fully-flexed elbow can triple a fold edge; 4x cannot be a fold). */
  EDGE_HARD_RATIO_LIMB: 4.0,
  /** V8. Absolute opening cap: NO edge may grow LONGER than its baseline
   *  by more than this fraction of model height (vs bind or clip start) —
   *  a 3% h gap is a visible crack whatever the ratio. */
  EDGE_ABS_OPEN_FRAC: 0.03,
  /** V8. Noise floor: a ratio offense only counts when the edge also opens
   *  by at least this fraction of model height. A 0.05% h edge stretching
   *  2x moved by half a pixel — sub-visible micro-edges must not dominate
   *  a ratio metric. The absolute cap above is independent of the floor. */
  EDGE_MIN_OPEN_FRAC: 0.002,

  /* --- fix round 3: V9 seam cracks ---------------------------------------- */

  /** V9. Weld epsilon (MODEL UNITS, absolute per the work order): vertices
   *  whose bind positions are within this distance are one weld group
   *  (UV/normal seams duplicate vertices at identical positions). */
  SEAM_WELD_EPS: 1e-5,
  /** V9. Maximum intra-group separation (fraction of model height) at any
   *  sampled frame of any clip. Duplicates with different weights split
   *  when their joints move apart — a visible crack along the UV seam.
   *  0.5% h is sub-pixel at preview scale; the probe found seams opening
   *  up to 5% h. */
  SEAM_MAX_SEPARATION_FRAC: 0.005,

  /* --- fix round 3: V4' reverse leakage ----------------------------------- */

  /** V4'. Radius (fraction of model height) around the head joints' bind
   *  bone star that, combined with "bind height above the topmost neck
   *  joint", defines the GEOMETRIC head region. Generous by principle: this
   *  cartoon head (cheek tufts included) is roughly the top half of the
   *  model; the neck-height gate is what excludes shoulders. ZERO vertices
   *  in this region may be DOMINATED by an arm-chain joint (clavicle/
   *  shoulder included): V4 only tested head-DOMINATED vertices, so
   *  RightShoulder owning 393 cheek/head vertices passed. */
  HEAD_REGION_RADIUS_FRAC: 0.45,

  /* --- fix round 3: V7' bind-time mis-binding ----------------------------- */

  /** V7'. A vertex is mis-bound (soft offender) when the bone stars of ALL
   *  its weighted joints are farther than this fraction of model height ... */
  MISBIND_OWN_FAR_FRAC: 0.10,
  /** V7'. ... while some OTHER bone segment is within this fraction of
   *  model height of the vertex ... */
  MISBIND_OTHER_NEAR_FRAC: 0.10,
  /** V7'. ... and the nearest other bone explains the vertex at least this
   *  many times better (dOwn > factor * dNearest). The factor is what keeps
   *  legitimately-far skin safe: for a muzzle/ear/belly vertex the nearest
   *  bone IS its own bone, so dOwn = dNearest and the test cannot fire.
   *  V7 is invariant BY CONSTRUCTION for any vertex bound 100% to one
   *  joint (skinned = delta_k * v_bind, so d_t = d_bind at every frame) —
   *  both probes proved rigid rebinds of a paw to Hips/Spine2 PASS V7.
   *  Mis-binding must therefore be caught AT BIND, geometrically. */
  MISBIND_DOMINANCE: 2.5,
  /** V7'. Maximum fraction of skinned vertices that may be soft offenders. */
  MISBIND_OFFENDER_RATIO: 0.002,
  /** V7'. Hard offender: ANY single influence carrying at least this much
   *  weight ... */
  MISBIND_HARD_WEIGHT: 0.35,
  /** V7'. ... whose own bone star is farther than this fraction of model
   *  height while the vertex sits within MISBIND_OTHER_NEAR_FRAC of some
   *  other bone (with the same dominance factor). Catches small BLENDED
   *  fragments (e.g. 20 paw vertices 0.55 Hand / 0.45 Spine2) that stay
   *  under the soft ratio because their dominant bone is nearby. ZERO hard
   *  offenders allowed. */
  MISBIND_HARD_FAR_FRAC: 0.20,
  /** V7'. Limb-scale sanity: every skin joint's scale — rest pose and every
   *  animated scale sample in every clip — must be 1 within this tolerance
   *  per component. A joint animated (or posed) at scale 1.6 balloons its
   *  geometry while V7's rigid-delta transport stays invariant (proven:
   *  RightForeArm scaled 1.6x PASSED V7). */
  JOINT_SCALE_TOL: 0.05,
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
  /** Shoulder-girdle tokens. Joints matching these stay in the arm chain
   *  for the ROTATION checks (V1) and the weight-leakage sum (V4), but are
   *  EXCLUDED from the waving-arm set that V3 exempts from co-deformation:
   *  chest/flank skin is routinely dominated by a Shoulder/Clavicle joint,
   *  and exempting it hid real 7.9x strain (reviewer A3). */
  shoulder: ['shoulder', 'clavicle'],
  /** Hand-end tokens: used to pick the "hand/wrist" joint for the
   *  raised-above-the-shoulder test (V1). */
  handEnd: ['hand', 'wrist', 'paw'],
  /** Left / right side tokens (prefixes/suffixes like `L_`, `_l`, `.L`,
   *  `Left`, `mixamorig:LeftArm` all tokenize to these). */
  left: ['l', 'left'],
  right: ['r', 'right'],
  /** Root / hips joints — ALL matches become root candidates for V2
   *  (fallback candidate: the skeleton's top joint). */
  root: ['root', 'hips', 'pelvis'],
  /** Torso joints for the rotation-stability check (V2). */
  torso: ['spine', 'chest', 'torso', 'neck', 'hips', 'pelvis'],
  /** Head/neck joints (descendants included) for the weight-leakage check
   *  (V4). NOTE: head/tail/leg vertices are NOT exempt from the V3
   *  co-deformation set — only the waving arm chain is (reviewer A2). */
  head: ['head', 'skull', 'jaw', 'face', 'eye', 'ear'],
  neck: ['neck'],
  /** Tail joints (kept for documentation/extension; tail vertices are body
   *  vertices for V3 like everything else outside the waving arm). */
  tail: ['tail'],
  /** Foot joints for the jump check (V6), sided like arms. */
  foot: ['foot', 'toe', 'ankle'],
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
 *   headNeck: Set,              // head/neck matched joints + descendants (V4)
 *   torso: Set,                 // torso-name-matched joints (V2)
 *   feet: Set,                  // foot|toe|ankle matched joints (V6)
 *   rootNamed: Set,             // ALL joints named root|hips|pelvis (V2)
 *   topJoint: nodeIdx | null,   // a joint whose parent is not a joint
 *   depth: Map<nodeIdx, number> // node-tree depth of every joint
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
  const headNeck = new Set();
  const headSeeds = new Set(); // joints whose OWN name matches the head patterns
  const neckSeeds = new Set(); // joints whose OWN name matches the neck patterns
  const torso = new Set();
  const feet = new Set();
  const rootNamed = new Set();
  let topJoint = null;

  for (const j of joints) {
    const name = (doc.json.nodes[j] || {}).name || '';
    if (matchesCategory(name, JOINT_NAME_PATTERNS.arm)) {
      const side = sideOfName(name) || 'unsided';
      for (const d of descendantsWithin(j)) armChains[side].add(d);
    }
    if (matchesCategory(name, JOINT_NAME_PATTERNS.head)) headSeeds.add(j);
    if (matchesCategory(name, JOINT_NAME_PATTERNS.neck)) neckSeeds.add(j);
    if (matchesCategory(name, JOINT_NAME_PATTERNS.head) || matchesCategory(name, JOINT_NAME_PATTERNS.neck)) {
      for (const d of descendantsWithin(j)) headNeck.add(d);
    }
    if (matchesCategory(name, JOINT_NAME_PATTERNS.torso)) torso.add(j);
    if (matchesCategory(name, JOINT_NAME_PATTERNS.foot)) feet.add(j);
    if (matchesCategory(name, JOINT_NAME_PATTERNS.root)) rootNamed.add(j);
  }

  // The skeleton's top joint — a joint whose parent is not a joint.
  for (const j of joints) {
    const p = parents.get(j);
    if (p === undefined || !joints.has(p)) { topJoint = j; break; }
  }

  // Node-tree depth of every joint (for "topmost"/"deepest" selection).
  const depth = new Map();
  for (const j of joints) {
    let d = 0, n = j;
    while (parents.get(n) !== undefined) { n = parents.get(n); d++; }
    depth.set(j, d);
  }

  const armAll = new Set([...armChains.left, ...armChains.right, ...armChains.unsided]);
  return { joints, armChains, armAll, headNeck, headSeeds, neckSeeds, torso, feet, rootNamed, topJoint, depth };
}

/**
 * The joints V3' does NOT accept as "upper arm" when deriving the geometric
 * limb volume: shoulder/clavicle-named joints belong to the torso girdle
 * (see JOINT_NAME_PATTERNS.shoulder — shoulder-dominated chest/flank skin
 * must stay accountable to the body checks; reviewer A3).
 */

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

/**
 * EXT_mesh_gpu_instancing: a node carrying the extension renders its mesh
 * once per instance — the instance count is the count of any of the
 * extension's attribute accessors (they must agree; the max is taken
 * defensively). A node without the extension renders once. (Reviewer A4:
 * an instanced node doubles the RENDERED triangles while the plain sum
 * still reports the single-copy count.)
 */
function nodeInstanceCount(doc, node) {
  const ext = node?.extensions?.EXT_mesh_gpu_instancing;
  if (!ext || !ext.attributes) return 1;
  let count = 0;
  for (const acc of Object.values(ext.attributes)) {
    const a = (doc.json.accessors || [])[acc];
    if (a && a.count > count) count = a.count;
  }
  return count > 0 ? count : 1;
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
    const instances = nodeInstanceCount(doc, node);
    t *= instances;
    total += t;
    perMesh.push({ node: doc.nodeName(n), mesh: mesh.name || `mesh#${node.mesh}`, triangles: t, instances });
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
  if (doc._modelHeight !== undefined) return doc._modelHeight;
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
  if (min[0] === Infinity) return (doc._modelHeight = 0);
  const ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return (doc._modelHeight = ext[1] > 1e-9 ? ext[1] : Math.max(...ext));
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
  const instancedNote = rows.some(r => r.perMesh.some(m => m.instances > 1))
    ? '; EXT_mesh_gpu_instancing multiplied in: ' + rows.flatMap(r => r.perMesh.filter(m => m.instances > 1)
        .map(m => `${r.file}:${m.node} x${m.instances}`)).join(', ')
    : '';
  return {
    id: 'triangles',
    title: `rendered triangles <= ${THRESHOLDS.MAX_TRIANGLES}`,
    pass,
    measured: rows.map(r => `${r.file}: ${r.total}`).join('; ') +
      ` (limit ${THRESHOLDS.MAX_TRIANGLES}; per node instance, all scenes, x gpu-instancing count, morph targets add none)` +
      instancedNote,
    details: { rows, worst },
  };
}

/**
 * Sniff image bytes: valid PNG (8-byte magic + IHDR width/height) or JPEG
 * (SOI + a SOFn frame header) with nonzero dimensions. The declared
 * mimeType is deliberately ignored — only the actual bytes count
 * (reviewer A5: a zeroed payload with an intact JSON reference must FAIL).
 * Returns { format, width, height } or null.
 */
export function sniffImage(bytes) {
  if (!bytes || bytes.length < 4) return null;
  // PNG: 89 50 4E 47 0D 0A 1A 0A, then the IHDR chunk (must be first).
  const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length >= 33 && PNG.every((b, i) => bytes[i] === b)) {
    const type = String.fromCharCode(bytes[12], bytes[13], bytes[14], bytes[15]);
    if (type !== 'IHDR') return null;
    const be32 = o => (bytes[o] << 24 | bytes[o + 1] << 16 | bytes[o + 2] << 8 | bytes[o + 3]) >>> 0;
    const width = be32(16), height = be32(20);
    return width > 0 && height > 0 ? { format: 'PNG', width, height } : null;
  }
  // JPEG: FF D8, then scan markers for SOF0..15 (except DHT C4 / JPG C8 / DAC CC).
  if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    let i = 2;
    while (i + 9 < bytes.length) {
      if (bytes[i] !== 0xff) return null; // marker desync -> not parseable
      let m = bytes[i + 1];
      while (m === 0xff && i + 2 < bytes.length) { i++; m = bytes[i + 1]; } // fill bytes
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
        const height = (bytes[i + 5] << 8) | bytes[i + 6];
        const width = (bytes[i + 7] << 8) | bytes[i + 8];
        return width > 0 && height > 0 ? { format: 'JPEG', width, height } : null;
      }
      if (m === 0xd8 || (m >= 0xd0 && m <= 0xd7) || m === 0x01) { i += 2; continue; } // standalone
      if (m === 0xd9 || m === 0xda) return null; // EOI/SOS before any SOF
      const segLen = (bytes[i + 2] << 8) | bytes[i + 3];
      if (segLen < 2) return null;
      i += 2 + segLen;
    }
  }
  return null;
}

/** Load the actual bytes of image `imageIdx` (bufferView / data: URI / file). */
function loadImageBytes(doc, imageIdx) {
  const img = (doc.json.images || [])[imageIdx];
  if (!img) return { bytes: null, why: `image ${imageIdx} missing` };
  if (img.bufferView !== undefined) {
    try {
      const bv = doc.json.bufferViews[img.bufferView];
      const buf = doc.buffer(bv.buffer);
      const off = bv.byteOffset || 0;
      return {
        bytes: buf.subarray(off, off + (bv.byteLength || 0)),
        why: `embedded bufferView ${img.bufferView} (${bv.byteLength} bytes)`,
      };
    } catch (e) { return { bytes: null, why: String(e.message) }; }
  }
  if (img.uri !== undefined) {
    if (img.uri.startsWith('data:')) {
      const comma = img.uri.indexOf(',');
      const meta = img.uri.slice(5, comma);
      const data = img.uri.slice(comma + 1);
      const bytes = meta.endsWith(';base64')
        ? Uint8Array.from(Buffer.from(data, 'base64'))
        : new TextEncoder().encode(decodeURIComponent(data));
      return { bytes, why: `data: URI (${bytes.length} bytes)` };
    }
    if (!doc.baseDir) return { bytes: null, why: `external uri "${img.uri}" not resolvable from memory` };
    const p = path.resolve(doc.baseDir, decodeURIComponent(img.uri));
    if (!fs.existsSync(p)) return { bytes: null, why: `uri "${img.uri}" NOT FOUND` };
    return { bytes: new Uint8Array(fs.readFileSync(p)), why: `uri "${img.uri}"` };
  }
  return { bytes: null, why: 'image has neither bufferView nor uri' };
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
        const loaded = loadImageBytes(doc, source);
        if (!loaded.bytes) { detail = `image data missing: ${loaded.why}`; continue; }
        const sniffed = sniffImage(loaded.bytes);
        if (!sniffed) {
          detail = `image bytes are NOT a valid PNG/JPEG (magic/header check failed): ${loaded.why}`;
          continue;
        }
        ok = true;
        detail = `material "${mat.name || prim.material}" on skinned node "${doc.nodeName(n)}" -> ` +
          `${loaded.why} = valid ${sniffed.format} ${sniffed.width}x${sniffed.height}`;
        break;
      }
      if (ok) break;
    }
    rows.push({ file: doc.label, ok, detail });
    if (!ok) pass = false;
  }
  return {
    id: 'baseColorTexture',
    title: 'PBR baseColorTexture on a skinned mesh material, image bytes = valid PNG/JPEG',
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

/** Shared: locate the doc + clip for a canonical name (list of accepted names). */
function findClip(docs, names) {
  for (const d of docs) {
    for (const a of d.json.animations || []) {
      if (names.includes(a.name)) return { doc: d, anim: a, clipName: a.name };
    }
  }
  return null;
}

/** Sample times for a clip: WAVE_SAMPLES_PER_SECOND, >= 2, capped. */
function sampleTimes(duration) {
  const n = Math.min(
    THRESHOLDS.WAVE_MAX_SAMPLES,
    Math.max(2, Math.ceil(duration * THRESHOLDS.WAVE_SAMPLES_PER_SECOND) + 1),
  );
  return Array.from({ length: n }, (_, i) => (i / (n - 1)) * duration);
}

/**
 * The wave check — V1 (arm), V2 (root/torso stability), V3 (co-deformation
 * + edge strain). See THRESHOLDS for every limit. Sampling:
 * WAVE_SAMPLES_PER_SECOND over the clip duration (>= 2 samples, capped at
 * WAVE_MAX_SAMPLES), with per-sampler STEP/LINEAR/CUBICSPLINE interpolation.
 *
 * ALL rotation measurements are relative to the WAVE CLIP'S FIRST FRAME,
 * never the bind/rest pose (reviewer A1: this rig's bind pose is an arms-up
 * "cheer", so rest-relative angles score a limp arm at 121.7 deg and a
 * STATIC arm at 78 deg — and picking the waving side by rest-relative peak
 * scores the wrong arm).
 */
function checkWaveArm(docs) {
  const sub = {};
  const fail = (msg) => ({
    id: 'waveArm', title: 'wave check V1-V3 (arm, oscillation, stability, co-deformation, strain)',
    pass: false, measured: msg, details: { subChecks: sub, error: msg },
  });

  const hit = findClip(docs, ['wave']);
  if (!hit) return fail('no animation named "wave" found in any file');
  const { doc, anim } = hit;

  const cls = classifyJoints(doc);
  if (!cls.joints.size) return fail('no skin joints in the file carrying "wave"');
  const jointNames = [...cls.joints].map(j => doc.nodeName(j));
  const height = modelHeight(doc);
  if (!(height > 0)) return fail('model height is zero — cannot scale thresholds');

  const sidedChains = ['left', 'right', 'unsided']
    .map(side => [side, cls.armChains[side]])
    .filter(([, s]) => s.size > 0);
  if (!sidedChains.length) {
    return fail(`no arm joints matched JOINT_NAME_PATTERNS.arm among skin joints: [${jointNames.join(', ')}]`);
  }

  const pose = buildPoseSampler(doc, anim);
  if (!(pose.duration > 0)) return fail('"wave" clip has zero duration');
  const times = sampleTimes(pose.duration);
  const nSamples = times.length;

  // Per-frame local rotations for every joint, as rotation vectors relative
  // to the CLIP'S FIRST FRAME, with quaternion double-cover continuity.
  const jointList = [...cls.joints];
  const relStartVec = new Map(); // j -> Array<[x,y,z]> (radians)
  {
    const prevStart = new Map();
    const startRot = new Map();
    for (const j of jointList) relStartVec.set(j, []);
    for (let f = 0; f < nSamples; f++) {
      for (const j of jointList) {
        const q = pose.localTRS(j, times[f]).r;
        if (f === 0) startRot.set(j, q);
        let qs = quatMul(quatConj(startRot.get(j)), q);
        const ps = prevStart.get(j);
        if (ps && quatDot(qs, ps) < 0) qs = qs.map(v => -v);
        prevStart.set(j, qs);
        relStartVec.get(j).push(quatToRotVec(qs));
      }
    }
  }
  const angleDeg = v => Math.hypot(v[0], v[1], v[2]) * DEG;
  const chainPeak = set => {
    let best = null;
    for (const j of set) {
      const peak = Math.max(...relStartVec.get(j).map(angleDeg));
      if (!best || peak > best.peakStartDeg) best = { joint: j, name: doc.nodeName(j), peakStartDeg: peak };
    }
    return best;
  };

  // --- V1: pick the waving side by IN-CLIP motion (peak rotation vs the
  // clip's first frame, over each chain's joints).
  const chainStats = sidedChains.map(([side, set]) => ({ side, ...chainPeak(set) }));
  chainStats.sort((a, b) => b.peakStartDeg - a.peakStartDeg);
  const best = chainStats[0];
  const winningChain = cls.armChains[best.side];

  sub.armRotation = {
    pass: best.peakStartDeg >= THRESHOLDS.WAVE_MIN_ARM_PEAK_DEG,
    measured: `${best.side} arm (side chosen by in-clip motion), joint "${best.name}": ` +
      `peak ${best.peakStartDeg.toFixed(1)} deg from the clip's first frame; ` +
      `threshold >= ${THRESHOLDS.WAVE_MIN_ARM_PEAK_DEG} deg`,
    peakStartDeg: best.peakStartDeg, arm: best.side, joint: best.name,
    perChain: chainStats.map(c => ({ side: c.side, joint: c.name, peakStartDeg: +c.peakStartDeg.toFixed(2) })),
  };

  // Per-frame global matrices (V1 hand height, V2, V3).
  const globalsPerFrame = times.map(t => pose.globalsAt(t));
  const worldPos = (f, j) => {
    const m = globalsPerFrame[f][j];
    return [m[12], m[13], m[14]];
  };

  // --- V1: at the peak frame the hand/wrist must be raised above the
  // shoulder (or upper-arm) joint by >= 5% of model height. "Peak frame" =
  // the frame maximizing (handY - shoulderY).
  {
    const deepest = (set, filterCat) => {
      let bestJ = null, bestD = -1;
      for (const j of set) {
        if (filterCat && !matchesCategory(doc.nodeName(j), filterCat)) continue;
        const d = cls.depth.get(j) || 0;
        if (d > bestD) { bestD = d; bestJ = j; }
      }
      return bestJ;
    };
    const topmost = (set, filterCat) => {
      let bestJ = null, bestD = Infinity;
      for (const j of set) {
        if (filterCat && !matchesCategory(doc.nodeName(j), filterCat)) continue;
        const d = cls.depth.get(j) || 0;
        if (d < bestD) { bestD = d; bestJ = j; }
      }
      return bestJ;
    };
    const hand = deepest(winningChain, JOINT_NAME_PATTERNS.handEnd) ?? deepest(winningChain, null);
    const shoulder = topmost(winningChain, JOINT_NAME_PATTERNS.shoulder) ?? topmost(winningChain, null);
    let peak = { margin: -Infinity, f: 0, handY: 0, shoulderY: 0 };
    for (let f = 0; f < nSamples; f++) {
      const hy = worldPos(f, hand)[1];
      const sy = worldPos(f, shoulder)[1];
      if (hy - sy > peak.margin) peak = { margin: hy - sy, f, handY: hy, shoulderY: sy };
    }
    const need = THRESHOLDS.WAVE_HAND_ABOVE_SHOULDER_FRAC * height;
    sub.handAboveShoulder = {
      pass: peak.margin >= need,
      measured: `hand "${doc.nodeName(hand)}" vs shoulder "${doc.nodeName(shoulder)}": best margin ` +
        `${peak.margin.toFixed(4)} units (${(peak.margin / height * 100).toFixed(1)}% h) at t=${times[peak.f].toFixed(2)}s ` +
        `(hand y ${peak.handY.toFixed(3)}, shoulder y ${peak.shoulderY.toFixed(3)}); ` +
        `threshold >= +${THRESHOLDS.WAVE_HAND_ABOVE_SHOULDER_FRAC * 100}% h`,
      marginUnits: peak.margin, marginFrac: peak.margin / height,
      handJoint: doc.nodeName(hand), shoulderJoint: doc.nodeName(shoulder), peakTime: times[peak.f],
    };
  }

  // --- V1: oscillations of the winning arm chain: best count over its
  // joints and over each rotation-vector component, vs the clip start.
  {
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
  }

  // --- V1: the OTHER arm chain must stay still (<= 15 deg in-clip).
  {
    const others = chainStats.slice(1);
    if (!others.length) {
      sub.otherArmStill = {
        pass: true,
        measured: 'only one arm chain found — nothing to hold still (pattern list may need extending)',
        maxDeg: 0,
      };
    } else {
      const worst = others.reduce((a, b) => (b.peakStartDeg > a.peakStartDeg ? b : a));
      sub.otherArmStill = {
        pass: worst.peakStartDeg <= THRESHOLDS.WAVE_MAX_OTHER_ARM_DEG,
        measured: `non-waving arm (${others.map(o => o.side).join('/')}): max in-clip rotation ` +
          `${worst.peakStartDeg.toFixed(1)} deg on "${worst.name}"; threshold <= ${THRESHOLDS.WAVE_MAX_OTHER_ARM_DEG} deg`,
        maxDeg: worst.peakStartDeg, joint: worst.name, side: worst.side,
      };
    }
  }

  // --- V2: EVERY root candidate must stay put: the top joint, every
  // root|hips|pelvis-named joint, and the topmost joint that actually has an
  // animated translation channel in this clip (reviewer A6: translation
  // authored on `Hips` under a never-animated `Root` must not hide).
  {
    const candidates = new Map(); // j -> Set<label>
    const addCand = (j, label) => {
      if (j === undefined || j === null) return;
      if (!candidates.has(j)) candidates.set(j, new Set());
      candidates.get(j).add(label);
    };
    addCand(cls.topJoint, 'top joint');
    for (const j of cls.rootNamed) addCand(j, 'root-named');
    let topAnim = null;
    for (const [nodeIdx, tr] of pose.tracks) {
      if (!tr.translation || !cls.joints.has(nodeIdx)) continue;
      const d = cls.depth.get(nodeIdx) || 0;
      if (!topAnim || d < topAnim.d) topAnim = { j: nodeIdx, d };
    }
    if (topAnim) addCand(topAnim.j, 'topmost animated translation');

    const rows = [];
    let worst = null;
    for (const [j, labels] of candidates) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let f = 0; f < nSamples; f++) {
        const p = worldPos(f, j);
        for (let c = 0; c < 3; c++) { if (p[c] < lo[c]) lo[c] = p[c]; if (p[c] > hi[c]) hi[c] = p[c]; }
      }
      const range = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      const frac = range / height;
      const row = { joint: doc.nodeName(j), roles: [...labels].join('+'), rangeUnits: range, rangeFrac: frac };
      rows.push(row);
      if (!worst || frac > worst.rangeFrac) worst = row;
    }
    sub.rootTranslation = {
      pass: rows.every(r => r.rangeFrac <= THRESHOLDS.WAVE_MAX_ROOT_TRANSLATION_FRAC),
      measured: rows.map(r => `"${r.joint}" (${r.roles}): ${(r.rangeFrac * 100).toFixed(2)}% h`).join('; ') +
        `; threshold each <= ${THRESHOLDS.WAVE_MAX_ROOT_TRANSLATION_FRAC * 100}% of model height ${height.toFixed(3)}`,
      candidates: rows, rangeFrac: worst ? worst.rangeFrac : 0, modelHeight: height,
    };
  }

  // --- V2: torso joints' local rotation (max angle vs the clip's first
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

  // --- V3': co-deformation via linear blend skinning, over ALL vertices
  // OUTSIDE the waving arm's GEOMETRIC limb volume, fixed at BIND (head,
  // tail, legs, the other arm, and — crucially — any torso vertex whose
  // WEIGHTS were re-pointed at the arm: round-3 probes proved that
  // classifying by current dominant joint let a belly slab bound 100% to
  // RightArm escape the check entirely; geometry cannot be re-pointed).
  // A vertex is limb only if it lies on the limb side of the plane through
  // the upper-arm joint perpendicular to the upper-arm bone AND within
  // LIMB_RADIUS_FRAC*h of the arm's bind bone chain (see limbVolumes).
  // Edge strain moved to V8 (all edges, every clip, vs bind AND clip
  // start); the wave clip's V8 verdict is surfaced here as a sub-check.
  {
    const vol = limbVolumes(doc)[best.side] || null;
    let bodyVertexCount = 0;
    let limbVertexCount = 0;
    let over3Count = 0;
    let maxDisp = { units: 0, frac: 0, node: null, vert: -1 };
    let over3Example = null;
    let skinnedPrims = 0;
    const soft = THRESHOLDS.BODY_DISPLACEMENT_SOFT_FRAC * height;
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
      // Per frame: jointMat[k] = G[joint k] * IBM[k]
      const jointMats = globalsPerFrame.map(g => jointsArr.map((j, k) => mat4Mul(g[j], ibm[k])));

      for (const prim of doc.json.meshes[node.mesh].primitives || []) {
        const at = prim.attributes || {};
        if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) continue;
        skinnedPrims++;
        const pos = doc.accessor(at.POSITION);
        const jnt = doc.accessor(at.JOINTS_0);
        const wgt = doc.accessor(at.WEIGHTS_0);
        const nv = pos.count;

        // Body vertices: skinned AND outside the waving limb's volume.
        const isBody = new Uint8Array(nv);
        for (let v = 0; v < nv; v++) {
          let tw = 0;
          for (let c = 0; c < 4; c++) tw += wgt.data[v * 4 + c];
          if (!(tw > 0)) continue;
          if (vol && vol.member(pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2])) {
            limbVertexCount++;
            continue;
          }
          isBody[v] = 1;
          bodyVertexCount++;
        }

        // Frame loop: skinned positions of body vertices; track per-vertex
        // max displacement from frame 0.
        const base = new Float64Array(nv * 3);
        const maxD = new Float64Array(nv);
        for (let f = 0; f < nSamples; f++) {
          const jm = jointMats[f];
          for (let v = 0; v < nv; v++) {
            if (!isBody[v]) continue;
            const p = [pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2]];
            let ox = 0, oy = 0, oz = 0;
            for (let c = 0; c < 4; c++) {
              const w = wgt.data[v * 4 + c];
              if (w === 0) continue;
              const q = mat4TransformPoint(jm[jnt.data[v * 4 + c]], p);
              ox += w * q[0]; oy += w * q[1]; oz += w * q[2];
            }
            if (f === 0) {
              base[v * 3] = ox; base[v * 3 + 1] = oy; base[v * 3 + 2] = oz;
            } else {
              const d = Math.hypot(ox - base[v * 3], oy - base[v * 3 + 1], oz - base[v * 3 + 2]);
              if (d > maxD[v]) maxD[v] = d;
              if (d > maxDisp.units) maxDisp = { units: d, frac: d / height, node: doc.nodeName(n), vert: v };
            }
          }
        }
        for (let v = 0; v < nv; v++) {
          if (isBody[v] && maxD[v] > soft) {
            over3Count++;
            if (!over3Example || maxD[v] > over3Example.units) {
              over3Example = { units: maxD[v], frac: maxD[v] / height, node: doc.nodeName(n), vert: v };
            }
          }
        }
      }
    }

    if (!skinnedPrims) {
      sub.bodyCoDeformation = { pass: false, measured: 'no skinned primitives (POSITION+JOINTS_0+WEIGHTS_0) found', over3Frac: null };
    } else if (!bodyVertexCount) {
      sub.bodyCoDeformation = { pass: false, measured: 'no skinned vertices outside the waving limb volume — cannot verify body stability', over3Frac: null };
    } else {
      const over3Frac = over3Count / bodyVertexCount;
      const okFrac = over3Frac <= THRESHOLDS.BODY_DISPLACEMENT_MAX_OFFENDER_RATIO;
      const okHard = maxDisp.frac <= THRESHOLDS.BODY_DISPLACEMENT_HARD_FRAC;
      sub.bodyCoDeformation = {
        pass: okFrac && okHard,
        measured: `${over3Count} of ${bodyVertexCount} vertices outside the ${best.side} limb volume ` +
          `(geometric at bind: plane at "${vol ? vol.upperArmJoint : 'n/a'}" + ` +
          `${THRESHOLDS.LIMB_RADIUS_FRAC * 100}% h chain radius; ${limbVertexCount} limb vertices exempt) ` +
          `(${(over3Frac * 100).toFixed(2)}%) moved > ${THRESHOLDS.BODY_DISPLACEMENT_SOFT_FRAC * 100}% h ` +
          `(allowed <= ${THRESHOLDS.BODY_DISPLACEMENT_MAX_OFFENDER_RATIO * 100}%)` +
          (over3Example ? `, worst offender ${(over3Example.frac * 100).toFixed(2)}% h` : '') +
          `; max displacement ${(maxDisp.frac * 100).toFixed(2)}% h (hard cap ${THRESHOLDS.BODY_DISPLACEMENT_HARD_FRAC * 100}% h)` +
          `; ${nSamples} frames`,
        bodyVertexCount, limbVertexCount, over3Count, over3Frac, maxFrac: maxDisp.frac, maxUnits: maxDisp.units,
        samples: nSamples,
      };
    }

    // Edge strain (V8) on the wave clip, surfaced as a wave sub-check.
    const scan = deformScanClip(doc, anim, 'wave');
    sub.edgeStrain = {
      pass: !!scan.strain.ok,
      measured: `V8 over ALL edges (vs bind AND clip start): ${scan.strain.detail}`,
      maxRatio: scan.strain.maxRatio ?? null,
      softCount: scan.strain.softCount, hardCount: scan.strain.hardCount,
      totalEdges: scan.strain.totalEdges,
    };
  }

  const pass = Object.values(sub).every(s => s.pass);
  const summary = `arm=${best.side} ("${best.name}"), peak ${best.peakStartDeg.toFixed(1)} deg from clip start, ` +
    `hand-above-shoulder ${sub.handAboveShoulder.pass ? 'yes' : 'NO'}, ` +
    `${sub.oscillations.oscillations} oscillations, other arm ${sub.otherArmStill.maxDeg.toFixed(1)} deg, ` +
    `root worst ${(sub.rootTranslation.rangeFrac * 100).toFixed(2)}% h, ` +
    `torso rot ${sub.torsoRotation.maxDeg.toFixed(1)} deg, ` +
    `body over-3%h ${sub.bodyCoDeformation.over3Frac === null ? 'n/a' : (sub.bodyCoDeformation.over3Frac * 100).toFixed(2) + '%'}` +
    `, max ${sub.bodyCoDeformation.maxFrac === undefined ? 'n/a' : (sub.bodyCoDeformation.maxFrac * 100).toFixed(2) + '% h'}` +
    `, strain ${sub.edgeStrain.maxRatio === null ? 'n/a' : sub.edgeStrain.maxRatio.toFixed(2) + 'x'}`;
  return {
    id: 'waveArm', title: 'wave check V1-V3 (arm, oscillation, stability, co-deformation, strain)',
    pass, measured: summary,
    details: { subChecks: sub, clipDuration: pose.duration, samples: nSamples, wavedArm: best.side, modelHeight: height },
  };
}

/* ---------------------------- V4 weight leakage --------------------------- */

/**
 * V4: ZERO vertices whose dominant joint is a head/neck joint may carry
 * more than HEAD_ARM_WEIGHT_MAX total arm-chain weight (either arm, full
 * chain including shoulder, summed over the 4 influences). Static check —
 * no animation needed (reviewer A2).
 */
function checkWeightLeakage(docs) {
  const rows = [];
  let pass = true;
  for (const doc of docs) {
    const cls = classifyJoints(doc);
    if (!cls.joints.size) { rows.push({ file: doc.label, ok: false, detail: 'no skin joints' }); pass = false; continue; }
    if (!cls.headNeck.size) {
      rows.push({ file: doc.label, ok: true, detail: 'no head/neck-named joints — nothing to leak onto (pattern list may need extending)' });
      continue;
    }
    let headVerts = 0, violations = 0, worst = { w: 0, node: null, vert: -1 };
    const rendered = renderedNodeSet(doc);
    for (const n of rendered) {
      const node = doc.json.nodes[n];
      if (!node || node.mesh === undefined || node.skin === undefined) continue;
      const jointsArr = doc.json.skins[node.skin].joints;
      for (const prim of doc.json.meshes[node.mesh].primitives || []) {
        const at = prim.attributes || {};
        if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) continue;
        const jnt = doc.accessor(at.JOINTS_0);
        const wgt = doc.accessor(at.WEIGHTS_0);
        const nv = doc.accessor(at.POSITION).count;
        for (let v = 0; v < nv; v++) {
          let domK = -1, domW = -1, armW = 0;
          for (let c = 0; c < 4; c++) {
            const w = wgt.data[v * 4 + c];
            const j = jointsArr[jnt.data[v * 4 + c]];
            if (w > domW) { domW = w; domK = j; }
            if (w > 0 && j !== undefined && cls.armAll.has(j)) armW += w;
          }
          if (domW <= 0 || domK === undefined || !cls.headNeck.has(domK)) continue;
          headVerts++;
          if (armW > THRESHOLDS.HEAD_ARM_WEIGHT_MAX) {
            violations++;
            if (armW > worst.w) worst = { w: armW, node: doc.nodeName(n), vert: v };
          }
        }
      }
    }
    const ok = violations === 0;
    rows.push({
      file: doc.label, ok, headVerts, violations,
      detail: `${headVerts} head/neck-dominated vertices; ${violations} carry > ` +
        `${THRESHOLDS.HEAD_ARM_WEIGHT_MAX} total arm-chain weight` +
        (violations ? ` (worst ${worst.w.toFixed(2)} on node "${worst.node}" vertex ${worst.vert})` : ''),
    });
    if (!ok) pass = false;
  }
  return {
    id: 'weightLeakage',
    title: `V4 weight leakage: no head/neck-dominated vertex carries > ${THRESHOLDS.HEAD_ARM_WEIGHT_MAX} arm weight`,
    pass,
    measured: rows.map(r => `${r.file}: ${r.ok ? 'ok' : 'FAIL'} — ${r.detail}`).join('; '),
    details: { rows },
  };
}

/* ------------------------------ V5 loop seams ----------------------------- */

/**
 * V5: in `idle` and `walk`/`run`, every animated channel's value at the
 * clip's end must equal its value at the clip's start within
 * LOOP_SEAM_TRANSLATION_FRAC of model height (translation) /
 * LOOP_SEAM_ROTATION_DEG degrees (rotation) / LOOP_SEAM_SCALE_REL relative
 * (scale) — otherwise the loop visibly pops (and a clip whose root drifts,
 * like a mislabeled walk, cannot close its seam).
 */
function checkLoopSeams(docs) {
  const clipsWanted = [['idle'], ['walk', 'run']];
  const rows = [];
  let pass = true;
  for (const names of clipsWanted) {
    const hit = findClip(docs, names);
    if (!hit) { rows.push({ clip: names.join('|'), ok: false, detail: 'clip not found' }); pass = false; continue; }
    const { doc, anim, clipName } = hit;
    const height = modelHeight(doc) || 1;
    // Clip duration: the max end time over all samplers in the clip.
    let duration = 0;
    const chans = [];
    for (const ch of anim.channels || []) {
      const target = ch.target || {};
      if (target.node === undefined) continue;
      if (!['translation', 'rotation', 'scale'].includes(target.path)) continue;
      const s = buildAnimSampler(doc, anim.samplers[ch.sampler], target.path);
      duration = Math.max(duration, s.duration);
      chans.push({ node: target.node, path: target.path, s });
    }
    const violations = [];
    for (const { node, path: p, s } of chans) {
      const v0 = s.sample(0), v1 = s.sample(duration);
      if (!v0 || !v1) continue;
      if (p === 'rotation') {
        const dq = quatMul(quatConj(quatNormalize(v0)), quatNormalize(v1));
        const deg = quatAngle(dq) * DEG;
        if (deg > THRESHOLDS.LOOP_SEAM_ROTATION_DEG) {
          violations.push({ node: doc.nodeName(node), path: p, delta: `${deg.toFixed(2)} deg` });
        }
      } else if (p === 'translation') {
        const d = Math.hypot(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]);
        if (d / height > THRESHOLDS.LOOP_SEAM_TRANSLATION_FRAC) {
          violations.push({ node: doc.nodeName(node), path: p, delta: `${(d / height * 100).toFixed(2)}% h` });
        }
      } else { // scale
        let rel = 0;
        for (let c = 0; c < 3; c++) rel = Math.max(rel, Math.abs(v1[c] - v0[c]) / (Math.abs(v0[c]) || 1));
        if (rel > THRESHOLDS.LOOP_SEAM_SCALE_REL) {
          violations.push({ node: doc.nodeName(node), path: p, delta: `${(rel * 100).toFixed(2)}% rel` });
        }
      }
    }
    const ok = violations.length === 0;
    rows.push({
      clip: clipName, ok, channels: chans.length, violations,
      detail: ok
        ? `${chans.length} channels close their loop (<= ${THRESHOLDS.LOOP_SEAM_TRANSLATION_FRAC * 100}% h / ` +
          `${THRESHOLDS.LOOP_SEAM_ROTATION_DEG} deg)`
        : `${violations.length} channel(s) do not close: ` +
          violations.slice(0, 4).map(v => `${v.node}.${v.path} Δ${v.delta}`).join(', ') +
          (violations.length > 4 ? ', …' : ''),
    });
    if (!ok) pass = false;
  }
  return {
    id: 'loopSeam',
    title: `V5 loop seams: idle & walk|run end where they start (<= ${THRESHOLDS.LOOP_SEAM_TRANSLATION_FRAC * 100}% h / ${THRESHOLDS.LOOP_SEAM_ROTATION_DEG} deg)`,
    pass,
    measured: rows.map(r => `${r.clip}: ${r.ok ? 'ok' : 'FAIL'} — ${r.detail}`).join('; '),
    details: { rows },
  };
}

/* -------------------------------- V6 jump --------------------------------- */

/**
 * V6: in `jump`, at some sampled frame EVERY foot joint (name tokens
 * foot|toe|ankle) must be at least JUMP_MIN_FOOT_RISE_FRAC of model height
 * above that joint's own clip-start world height — the character actually
 * leaves the ground.
 */
function checkJump(docs) {
  const hit = findClip(docs, ['jump']);
  const failRes = msg => ({
    id: 'jump', title: `V6 jump: feet rise >= ${THRESHOLDS.JUMP_MIN_FOOT_RISE_FRAC * 100}% h`,
    pass: false, measured: msg, details: { error: msg },
  });
  if (!hit) return failRes('no animation named "jump" found in any file');
  const { doc, anim } = hit;
  const cls = classifyJoints(doc);
  const height = modelHeight(doc);
  if (!(height > 0)) return failRes('model height is zero — cannot scale thresholds');
  if (!cls.feet.size) {
    return failRes(`no foot joints matched JOINT_NAME_PATTERNS.foot (${JOINT_NAME_PATTERNS.foot.join('|')}) ` +
      `among skin joints: [${[...cls.joints].map(j => doc.nodeName(j)).join(', ')}]`);
  }
  const pose = buildPoseSampler(doc, anim);
  if (!(pose.duration > 0)) return failRes('"jump" clip has zero duration');
  const times = sampleTimes(pose.duration);
  const feet = [...cls.feet];
  const startY = new Map();
  let peak = { rise: -Infinity, t: 0 };
  for (let f = 0; f < times.length; f++) {
    const g = pose.globalsAt(times[f]);
    let minRise = Infinity;
    for (const j of feet) {
      const y = g[j][13];
      if (f === 0) startY.set(j, y);
      const rise = y - startY.get(j);
      if (rise < minRise) minRise = rise;
    }
    if (f > 0 && minRise > peak.rise) peak = { rise: minRise, t: times[f] };
  }
  const frac = peak.rise / height;
  const need = THRESHOLDS.JUMP_MIN_FOOT_RISE_FRAC;
  return {
    id: 'jump',
    title: `V6 jump: feet rise >= ${need * 100}% h`,
    pass: frac >= need,
    measured: `best simultaneous rise of ALL ${feet.length} foot joint(s) ` +
      `(${feet.map(j => doc.nodeName(j)).join(', ')}): ${peak.rise.toFixed(4)} units = ` +
      `${(frac * 100).toFixed(1)}% h at t=${peak.t.toFixed(2)}s; threshold >= ${need * 100}% h`,
    details: { riseUnits: peak.rise, riseFrac: frac, peakTime: peak.t, feet: feet.map(j => doc.nodeName(j)) },
  };
}

/* ---------------------------- V7 limb integrity --------------------------- */

/**
 * Origin of the inverse of an affine column-major mat4 — i.e. the point p
 * with M * p = origin. For a joint's inverseBindMatrix this is the joint's
 * BIND-POSE position in mesh space. Returns null when the linear part is
 * singular.
 */
function mat4AffineInvOrigin(m) {
  // column-major: A[r][c] = m[c*4+r], t = (m[12], m[13], m[14])
  const a = m[0], b = m[4], c = m[8];
  const d = m[1], e = m[5], f = m[9];
  const g = m[2], h = m[6], i = m[10];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(Math.abs(det) > 1e-12)) return null;
  const tx = m[12], ty = m[13], tz = m[14];
  return [
    -((e * i - f * h) * tx + (c * h - b * i) * ty + (b * f - c * e) * tz) / det,
    -((f * g - d * i) * tx + (a * i - c * g) * ty + (c * d - a * f) * tz) / det,
    -((d * h - e * g) * tx + (b * g - a * h) * ty + (a * e - b * d) * tz) / det,
  ];
}

/** Full affine inverse of a column-major mat4 (null when singular). */
function mat4AffineInverse(m) {
  const a = m[0], b = m[4], c = m[8];
  const d = m[1], e = m[5], f = m[9];
  const g = m[2], h = m[6], i = m[10];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(Math.abs(det) > 1e-12)) return null;
  const i00 = (e * i - f * h) / det, i01 = (c * h - b * i) / det, i02 = (b * f - c * e) / det;
  const i10 = (f * g - d * i) / det, i11 = (a * i - c * g) / det, i12 = (c * d - a * f) / det;
  const i20 = (d * h - e * g) / det, i21 = (b * g - a * h) / det, i22 = (a * e - b * d) / det;
  const tx = m[12], ty = m[13], tz = m[14];
  return [
    i00, i10, i20, 0,
    i01, i11, i21, 0,
    i02, i12, i22, 0,
    -(i00 * tx + i01 * ty + i02 * tz),
    -(i10 * tx + i11 * ty + i12 * tz),
    -(i20 * tx + i21 * ty + i22 * tz), 1,
  ];
}

/** Rest-pose world position of a node (fallback when an IBM is singular). */
function restWorldPos(doc, nodeIdx) {
  const parents = doc.parents();
  let m = null;
  for (let n = nodeIdx; n !== undefined; n = parents.get(n)) {
    const r = nodeRest(doc.json.nodes[n] || {});
    const local = r.matrix || mat4FromTRS(r.t, r.r, r.s);
    m = m ? mat4Mul(local, m) : local;
  }
  return m ? [m[12], m[13], m[14]] : [0, 0, 0];
}

/** Distance from point (px,py,pz) to segment ends[o..o+5] of a packed
 *  [ax,ay,az,bx,by,bz]* array. */
function distPointSeg(px, py, pz, ends, o) {
  const ax = ends[o], ay = ends[o + 1], az = ends[o + 2];
  let dx = ends[o + 3] - ax, dy = ends[o + 4] - ay, dz = ends[o + 5] - az;
  const len2 = dx * dx + dy * dy + dz * dz;
  let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  dx = px - (ax + t * dx); dy = py - (ay + t * dy); dz = pz - (az + t * dz);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Per-skin V7 data: bone segments, per-joint bone "stars" and bind-pose
 * joint positions.
 *
 * Segments are parent-joint -> child-joint pairs over the skin's joints,
 * DROPPING (a) segments whose parent is a top joint (a joint whose node-tree
 * parent is not itself one of the skin's joints — a Root->Hips "segment"
 * spans the whole body and is not a limb bone) and (b) segments of
 * ~zero bind length (twist/helper joints stacked on their parent). Bind
 * positions come from the inverse of each joint's inverseBindMatrix — the
 * bind pose is exactly the pose in which every joint delta G*IBM is the
 * identity, i.e. skinned vertices sit at their raw POSITION.
 *
 * star[k] = the segments incident to joint k (to its parent and to each
 * child), as a packed [ax,ay,az,bx,by,bz]* Float64Array of BIND endpoints —
 * the bones a vertex weighted to joint k can legitimately claim to wrap.
 * A joint left with no segments (e.g. the top joint) falls back to its own
 * bind position as a degenerate point-segment.
 */
function limbSkinData(doc, skinIdx, height) {
  if (!doc._limbSkin) doc._limbSkin = new Map();
  if (doc._limbSkin.has(skinIdx)) return doc._limbSkin.get(skinIdx);
  const skin = doc.json.skins[skinIdx];
  const jointsArr = skin.joints || [];
  const parents = doc.parents();
  const ibmAcc = skin.inverseBindMatrices !== undefined ? doc.accessor(skin.inverseBindMatrices) : null;
  const ibm = jointsArr.map((_, k) =>
    ibmAcc ? Array.from(ibmAcc.data.slice(k * 16, k * 16 + 16)) : mat4Identity());
  const bindPos = jointsArr.map((j, k) => mat4AffineInvOrigin(ibm[k]) || restWorldPos(doc, j));

  const idxOf = new Map();
  jointsArr.forEach((j, k) => { if (!idxOf.has(j)) idxOf.set(j, k); });
  const isTop = j => {
    const p = parents.get(j);
    return p === undefined || !idxOf.has(p);
  };
  const eps = Math.max(1e-9, 1e-6 * height);
  const segs = []; // [parentK, childK] indices into jointsArr
  const starSegs = jointsArr.map(() => []);
  for (let k = 0; k < jointsArr.length; k++) {
    const j = jointsArr[k];
    const p = parents.get(j);
    if (p === undefined || !idxOf.has(p)) continue; // j is a top joint
    if (isTop(p)) continue;                         // segment from the top joint
    const pk = idxOf.get(p);
    const a = bindPos[pk], bp = bindPos[k];
    if (Math.hypot(bp[0] - a[0], bp[1] - a[1], bp[2] - a[2]) < eps) continue; // zero-length
    segs.push([pk, k]);
    starSegs[k].push([a, bp]);
    starSegs[pk].push([a, bp]);
  }
  const star = starSegs.map((list, k) => {
    if (!list.length) list = [[bindPos[k], bindPos[k]]]; // point fallback
    const packed = new Float64Array(list.length * 6);
    list.forEach(([a, b], s) => { packed.set(a, s * 6); packed.set(b, s * 6 + 3); });
    return packed;
  });
  const data = { jointsArr, ibm, bindPos, segs, star };
  doc._limbSkin.set(skinIdx, data);
  return data;
}

/**
 * V7 for one clip: sample the clip like the wave check; at every frame,
 * measure each skinned vertex against the bone segments of its OWN weighted
 * joints, each bone carried rigidly by its joint's motion:
 *
 *   d_bind(v) = min over weighted joints k of dist(POSITION(v), star(k))
 *   d_t(v)    = min over weighted joints k of
 *               dist(delta_k(t)^-1 * skinned(v, t), star(k))
 *
 * where delta_k(t) = G_k(t) * IBM_k is joint k's rigid delta from the bind
 * pose and star(k) its incident BIND bone segments. dist(delta^-1 x, star)
 * equals the distance from the skinned vertex to the bone carried by the
 * joint's motion — so geometry rigidly attached to any of its joints keeps
 * d_t = d_bind EXACTLY, whatever the pose (this is what makes a muzzle or
 * ear hanging off a leaf joint benign in principle: the joint's FRAME
 * rotates with it, while a bare bone segment cannot represent that
 * orientation). Detached or doubled geometry whose skinned motion no bone
 * explains — a second forearm lagging behind the real one — grows d_t past
 * its allowance on hundreds of vertices at once.
 *
 * Restricting candidates to the vertex's own weighted joints is what stops
 * detached geometry from "re-anchoring": a paw parked beside a leg it
 * carries no weight for gets no credit for that proximity.
 *
 * See THRESHOLDS.LIMB_* for the pass criteria and their rationale.
 * Vertices with zero total skin weight are not skinned (they never move)
 * and are excluded from the counts.
 */
function limbIntegrityClip(doc, anim, clipName) {
  const T = THRESHOLDS;
  const fail = detail => ({ clip: clipName, ok: false, detail });
  const height = modelHeight(doc);
  if (!(height > 0)) return fail('model height is zero — cannot scale thresholds');
  const pose = buildPoseSampler(doc, anim);
  if (!(pose.duration > 0)) return fail('clip has zero duration');
  const times = sampleTimes(pose.duration);
  const rendered = renderedNodeSet(doc);

  const softBase = T.LIMB_SOFT_BASE_FRAC * height;
  const softDriftCap = T.LIMB_DRIFT_SOFT_FRAC * height;
  const hardBase = T.LIMB_HARD_BASE_FRAC * height;
  const hardDriftCap = T.LIMB_DRIFT_HARD_FRAC * height;

  const distToStar = (star, px, py, pz) => {
    let best = Infinity;
    for (let s = 0; s < star.length; s += 6) {
      const d = distPointSeg(px, py, pz, star, s);
      if (d < best) best = d;
    }
    return best;
  };

  // Collect skinned primitives with their per-skin bone-star data and
  // per-vertex bind distances (cached on the doc across clips).
  if (!doc._limbPrim) doc._limbPrim = new Map();
  const prims = [];
  for (const n of rendered) {
    const node = doc.json.nodes[n];
    if (!node || node.mesh === undefined || node.skin === undefined) continue;
    const sd = limbSkinData(doc, node.skin, height);
    if (!sd.segs.length) {
      return fail(`skin of node "${doc.nodeName(n)}" has no measurable bone segments ` +
        '(only top-joint or zero-length segments) — limb integrity cannot be verified');
    }
    (doc.json.meshes[node.mesh].primitives || []).forEach((prim, pi) => {
      const at = prim.attributes || {};
      if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) return;
      const key = `${n}/${pi}`;
      let cached = doc._limbPrim.get(key);
      if (!cached) {
        const pos = doc.accessor(at.POSITION);
        const jnt = doc.accessor(at.JOINTS_0);
        const wgt = doc.accessor(at.WEIGHTS_0);
        const nv = pos.count;
        const dBind = new Float64Array(nv);
        const domK = new Int32Array(nv);
        const skinned = new Uint8Array(nv);
        let skinnedCount = 0;
        for (let v = 0; v < nv; v++) {
          let total = 0, dk = -1, dw = -1;
          for (let c = 0; c < 4; c++) {
            const w = wgt.data[v * 4 + c];
            total += w;
            if (w > dw) { dw = w; dk = jnt.data[v * 4 + c]; }
          }
          if (!(total > 0)) continue; // unskinned: never moves
          skinned[v] = 1; skinnedCount++;
          domK[v] = dk;
          const px = pos.data[v * 3], py = pos.data[v * 3 + 1], pz = pos.data[v * 3 + 2];
          let best = Infinity;
          for (let c = 0; c < 4; c++) {
            if (wgt.data[v * 4 + c] === 0) continue;
            const d = distToStar(sd.star[jnt.data[v * 4 + c]], px, py, pz);
            if (d < best) best = d;
          }
          dBind[v] = best;
        }
        cached = { pos, jnt, wgt, nv, dBind, domK, skinned, skinnedCount };
        doc._limbPrim.set(key, cached);
      }
      prims.push({ node: n, sd, ...cached });
    });
  }
  if (!prims.length) return fail('no skinned primitives (POSITION+JOINTS_0+WEIGHTS_0) found');
  const totalVerts = prims.reduce((a, p) => a + p.skinnedCount, 0);
  if (!totalVerts) return fail('no vertices carry any skin weight');

  let worstFrame = { ratio: -1, count: 0, t: 0, tally: null };
  let maxExcess = { units: 0, frac: 0, t: 0, joint: null };
  let hard = { count: 0, worstUnits: 0, joint: null, t: 0 };

  for (let f = 0; f < times.length; f++) {
    const g = pose.globalsAt(times[f]);
    let frameOff = 0;
    const tally = new Map(); // dominant joint name -> offender count
    for (const pr of prims) {
      const { sd } = pr;
      const nJ = sd.jointsArr.length;
      const jm = new Array(nJ);   // joint delta G * IBM
      const jmInv = new Array(nJ);
      for (let k = 0; k < nJ; k++) {
        jm[k] = mat4Mul(g[sd.jointsArr[k]], sd.ibm[k]);
        jmInv[k] = mat4AffineInverse(jm[k]); // null when degenerate (scale 0)
      }
      const { pos, jnt, wgt, nv, dBind, domK, skinned } = pr;
      for (let v = 0; v < nv; v++) {
        if (!skinned[v]) continue;
        const p = [pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2]];
        let ox = 0, oy = 0, oz = 0;
        for (let c = 0; c < 4; c++) {
          const w = wgt.data[v * 4 + c];
          if (w === 0) continue;
          const q = mat4TransformPoint(jm[jnt.data[v * 4 + c]], p);
          ox += w * q[0]; oy += w * q[1]; oz += w * q[2];
        }
        const db = dBind[v];
        const soft = Math.min(db * T.LIMB_SOFT_DBIND_SCALE, db + softDriftCap) + softBase;
        // d_t: best explanation of the skinned position by any weighted
        // joint's rigidly-carried bone star (early exit once within soft).
        let d = Infinity;
        for (let c = 0; c < 4; c++) {
          if (wgt.data[v * 4 + c] === 0) continue;
          const k = jnt.data[v * 4 + c];
          const inv = jmInv[k];
          if (!inv) continue;
          const q = mat4TransformPoint(inv, [ox, oy, oz]);
          const dk = distToStar(sd.star[k], q[0], q[1], q[2]);
          if (dk < d) { d = dk; if (d <= soft) break; }
        }
        if (d <= soft) continue;
        frameOff++;
        const name = doc.nodeName(sd.jointsArr[domK[v]]);
        tally.set(name, (tally.get(name) || 0) + 1);
        const excess = d - soft;
        if (excess > maxExcess.units) {
          maxExcess = { units: excess, frac: excess / height, t: times[f], joint: name };
        }
        const hardCap = Math.min(db * T.LIMB_HARD_DBIND_SCALE, db + hardDriftCap) + hardBase;
        if (d > hardCap) {
          hard.count++;
          if (d - hardCap > hard.worstUnits) hard = { ...hard, worstUnits: d - hardCap, joint: name, t: times[f] };
        }
      }
    }
    const ratio = frameOff / totalVerts;
    if (ratio > worstFrame.ratio) worstFrame = { ratio, count: frameOff, t: times[f], tally };
  }

  const domEntry = worstFrame.tally && worstFrame.tally.size
    ? [...worstFrame.tally.entries()].sort((a, b) => b[1] - a[1])[0]
    : null;
  const okSoft = worstFrame.ratio <= T.LIMB_OFFENDER_RATIO;
  const okHard = hard.count === 0;
  return {
    clip: clipName, ok: okSoft && okHard,
    worstRatio: worstFrame.ratio, worstCount: worstFrame.count, worstTime: worstFrame.t,
    totalVerts, samples: times.length,
    maxExcessFrac: maxExcess.frac, hardCount: hard.count,
    dominantJoint: domEntry ? domEntry[0] : null,
    detail: `worst frame t=${worstFrame.t.toFixed(2)}s: ${worstFrame.count} of ${totalVerts} ` +
      `skinned vertices (${(worstFrame.ratio * 100).toFixed(2)}%) beyond ` +
      `min(${T.LIMB_SOFT_DBIND_SCALE}x d_bind, d_bind + ${T.LIMB_DRIFT_SOFT_FRAC * 100}% h) + ` +
      `${T.LIMB_SOFT_BASE_FRAC * 100}% h ` +
      `(allowed <= ${T.LIMB_OFFENDER_RATIO * 100}%)` +
      (worstFrame.count
        ? `, max excess ${(maxExcess.frac * 100).toFixed(2)}% h` +
          (domEntry ? `, dominant offender joint "${domEntry[0]}" (${domEntry[1]} of ${worstFrame.count})` : '')
        : '') +
      `; ${hard.count} vertex-frame(s) past the hard cap ` +
      `min(${T.LIMB_HARD_DBIND_SCALE}x d_bind, d_bind + ${T.LIMB_DRIFT_HARD_FRAC * 100}% h) + ` +
      `${T.LIMB_HARD_BASE_FRAC * 100}% h` +
      (hard.count ? ` (worst +${(hard.worstUnits / height * 100).toFixed(2)}% h on "${hard.joint}" at t=${hard.t.toFixed(2)}s)` : '') +
      `; ${times.length} frames`,
  };
}

/**
 * V7: limb integrity over EVERY required clip. Geometry that leaves the
 * bones it wraps — a doubled forearm lobe that stays behind while the real
 * arm rises, a detached paw parked beside the body — FAILS here even when
 * its dominant joint is exempt from V3 (the waving arm chain) or its clip
 * passes every joint-motion check. No joint is exempted by name; regions
 * legitimately far from every bone are handled in principle: each vertex's
 * allowance scales with its own bind-pose wrap distance d_bind (capped in
 * absolute terms — see THRESHOLDS.LIMB_DRIFT_SOFT_FRAC), and each vertex is
 * measured against its own weighted joints' bones carried by those joints'
 * rigid motion, so anything that actually rides its bones scores
 * d_t = d_bind exactly.
 */
function checkLimbIntegrity(docs) {
  const rows = [];
  let pass = true;
  for (const group of REQUIRED_CLIPS) {
    const hit = findClip(docs, group);
    if (!hit) {
      rows.push({ clip: group.join('|'), ok: false, detail: 'clip not found' });
      pass = false;
      continue;
    }
    const r = limbIntegrityClip(hit.doc, hit.anim, hit.clipName);
    rows.push(r);
    if (!r.ok) pass = false;
  }
  let worst = null;
  for (const r of rows) {
    if (r.worstRatio === undefined) continue;
    if (!worst || r.worstRatio > worst.worstRatio) worst = r;
  }
  return {
    id: 'limbIntegrity',
    title: `V7 limb integrity: skinned geometry rides its own bones in every clip ` +
      `(<= ${THRESHOLDS.LIMB_OFFENDER_RATIO * 100}% of vertices past min(1.5x d_bind, d_bind + 3% h) + 3% h, ` +
      `none past min(2x d_bind, d_bind + 8% h) + 8% h)`,
    pass,
    measured: (worst
      ? `worst clip "${worst.clip}": ${worst.detail}. `
      : '') + `per clip — ` + rows.map(r => `${r.clip}: ${r.ok ? 'ok' : 'FAIL'}`).join('; '),
    details: { rows, worstClip: worst ? worst.clip : null },
  };
}

/* ====================================================================== *
 * Fix round 3 — geometric limb volumes (V3'/V8), all-edge strain (V8),
 * seam cracks (V9), reverse leakage (V4'), bind-time mis-binding (V7').
 * ====================================================================== */

/** Bind-pose position (mesh space) of joint node j: inverse of its IBM in
 *  the first skin that lists it; rest-pose world position as fallback. */
function jointBindMap(doc) {
  if (doc._jointBind) return doc._jointBind;
  const map = new Map();
  const h = modelHeight(doc) || 1;
  (doc.json.skins || []).forEach((skin, si) => {
    const sd = limbSkinData(doc, si, h);
    (skin.joints || []).forEach((j, k) => { if (!map.has(j)) map.set(j, sd.bindPos[k]); });
  });
  doc._jointBind = map;
  return map;
}

/**
 * GEOMETRIC arm-limb volumes, fixed at BIND — V3' body membership and V8
 * edge classes key off geometry, never off weights (a re-pointed torso
 * vertex stays a torso vertex). Per sided arm chain:
 *
 *   upper-arm joint U = the topmost non-shoulder/clavicle joint of the
 *   chain (the shoulder girdle belongs to the torso);
 *   upper-arm bone   = U -> its chain child on the path to the hand end;
 *   limb volume      = the half-space on the limb side of the plane
 *   through U perpendicular to the upper-arm bone, INTERSECTED with the
 *   set of points within LIMB_RADIUS_FRAC * h of the chain's bind bone
 *   segments at/below U.
 *
 * `union(px,py,pz)` tests membership in ANY side's volume.
 */
function limbVolumes(doc) {
  if (doc._limbVolumes) return doc._limbVolumes;
  const cls = classifyJoints(doc);
  const h = modelHeight(doc) || 1;
  const R = THRESHOLDS.LIMB_RADIUS_FRAC * h;
  const bind = jointBindMap(doc);
  const parents = doc.parents();
  const pos = j => bind.get(j) || restWorldPos(doc, j);
  const sides = {};

  for (const side of ['left', 'right', 'unsided']) {
    const chain = cls.armChains[side];
    if (!chain || !chain.size) continue;
    // upper-arm joint U: topmost non-shoulder chain joint (fallback: topmost).
    let U = null, bestD = Infinity;
    for (const j of chain) {
      if (matchesCategory(doc.nodeName(j), JOINT_NAME_PATTERNS.shoulder)) continue;
      const d = cls.depth.get(j) || 0;
      if (d < bestD) { bestD = d; U = j; }
    }
    if (U === null) {
      for (const j of chain) {
        const d = cls.depth.get(j) || 0;
        if (d < bestD) { bestD = d; U = j; }
      }
    }
    // hand-end joint: deepest handEnd-named chain joint (fallback: deepest).
    let handJ = null, hd = -1;
    for (const j of chain) {
      if (!matchesCategory(doc.nodeName(j), JOINT_NAME_PATTERNS.handEnd)) continue;
      const d = cls.depth.get(j) || 0;
      if (d > hd) { hd = d; handJ = j; }
    }
    if (handJ === null) {
      for (const j of chain) {
        const d = cls.depth.get(j) || 0;
        if (d > hd) { hd = d; handJ = j; }
      }
    }
    const Upos = pos(U);
    // plane normal: along the upper-arm bone (U -> its child on the path to
    // the hand end; fallbacks: any chain child of U, then parent(U) -> U).
    let dirTarget = null;
    {
      let cur = handJ, prev = null, found = false;
      while (cur !== undefined) {
        if (cur === U) { found = true; break; }
        prev = cur;
        cur = parents.get(cur);
      }
      if (found && prev !== null) dirTarget = prev;
    }
    if (dirTarget === null) {
      for (const c of doc.json.nodes[U]?.children || []) if (chain.has(c)) { dirTarget = c; break; }
    }
    let n;
    if (dirTarget !== null) {
      const c = pos(dirTarget);
      n = [c[0] - Upos[0], c[1] - Upos[1], c[2] - Upos[2]];
    } else {
      const p = parents.get(U);
      const pp = p !== undefined ? pos(p) : [0, 0, 0];
      n = [Upos[0] - pp[0], Upos[1] - pp[1], Upos[2] - pp[2]];
    }
    const nl = Math.hypot(n[0], n[1], n[2]) || 1;
    n = [n[0] / nl, n[1] / nl, n[2] / nl];
    // bind bone segments of the chain at/below U.
    const under = new Set([U]);
    const stack = [U];
    while (stack.length) {
      const x = stack.pop();
      for (const c of doc.json.nodes[x]?.children || []) {
        if (chain.has(c) && !under.has(c)) { under.add(c); stack.push(c); }
      }
    }
    const segList = [];
    for (const j of under) {
      if (j === U) continue;
      const p = parents.get(j);
      if (p === undefined || !under.has(p)) continue;
      segList.push([pos(p), pos(j)]);
    }
    if (!segList.length) segList.push([Upos, pos(handJ)]);
    const packed = new Float64Array(segList.length * 6);
    segList.forEach(([a, b], s) => { packed.set(a, s * 6); packed.set(b, s * 6 + 3); });
    sides[side] = {
      side,
      upperArmJoint: doc.nodeName(U),
      member(px, py, pz) {
        if ((px - Upos[0]) * n[0] + (py - Upos[1]) * n[1] + (pz - Upos[2]) * n[2] < 0) return false;
        for (let s = 0; s < packed.length; s += 6) {
          if (distPointSeg(px, py, pz, packed, s) <= R) return true;
        }
        return false;
      },
    };
  }
  const list = Object.values(sides);
  const vols = { ...sides, union: (px, py, pz) => list.some(v => v.member(px, py, pz)) };
  doc._limbVolumes = vols;
  return vols;
}

/**
 * One deformation scan of a clip, shared by V8 (edge strain) and V9 (seam
 * cracks) — LBS over every skinned vertex at WAVE_SAMPLES_PER_SECOND.
 *
 * V8: EVERY edge of every skinned primitive (NO exemption for triangles
 * touching waving-arm vertices — the old V3 dropped exactly the boundary
 * where the arm tears off the torso), measured against BOTH baselines: the
 * BIND length (raw POSITION — catches assets whose very first frame is
 * already torn open vs the authored surface) and the clip's first frame.
 * Edges wholly inside a geometric limb volume get the looser elbow/shoulder
 * fold caps. See THRESHOLDS.EDGE_*.
 *
 * V9: weld groups = vertices co-located within SEAM_WELD_EPS at bind (UV /
 * normal seams duplicate vertices); at every sampled frame the maximum
 * intra-group separation must stay <= SEAM_MAX_SEPARATION_FRAC * h, else
 * the texture seam visibly cracks apart. See THRESHOLDS.SEAM_*.
 */
function deformScanClip(doc, anim, clipName) {
  if (!doc._deformScan) doc._deformScan = new Map();
  const key = (doc.json.animations || []).indexOf(anim);
  if (doc._deformScan.has(key)) return doc._deformScan.get(key);
  const T = THRESHOLDS;
  const done = r => { doc._deformScan.set(key, r); return r; };
  const failBoth = detail => done({
    clip: clipName,
    strain: { ok: false, detail, maxRatio: null },
    seams: { ok: false, detail, maxSepFrac: null },
  });

  const height = modelHeight(doc);
  if (!(height > 0)) return failBoth('model height is zero — cannot scale thresholds');
  const pose = buildPoseSampler(doc, anim);
  if (!(pose.duration > 0)) return failBoth('clip has zero duration');
  const times = sampleTimes(pose.duration);
  const rendered = renderedNodeSet(doc);
  const vols = limbVolumes(doc);

  // ---- collect skinned primitives + per-vertex bind data.
  const prims = [];
  for (const nIdx of rendered) {
    const node = doc.json.nodes[nIdx];
    if (!node || node.mesh === undefined || node.skin === undefined) continue;
    for (const prim of doc.json.meshes[node.mesh].primitives || []) {
      const at = prim.attributes || {};
      if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) continue;
      const pos = doc.accessor(at.POSITION);
      const jnt = doc.accessor(at.JOINTS_0);
      const wgt = doc.accessor(at.WEIGHTS_0);
      const nv = pos.count;
      const skin = doc.json.skins[node.skin];
      const jointsArr = skin.joints;
      const ibmAcc = skin.inverseBindMatrices !== undefined ? doc.accessor(skin.inverseBindMatrices) : null;
      const ibm = jointsArr.map((_, k) => (ibmAcc ? Array.from(ibmAcc.data.slice(k * 16, k * 16 + 16)) : mat4Identity()));
      const inLimb = new Uint8Array(nv);
      for (let v = 0; v < nv; v++) {
        if (vols.union(pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2])) inLimb[v] = 1;
      }
      // unique edges from the primitive's triangles.
      const edgeKeys = new Map(); // a*nv+b (a<b) -> edge list index
      const ea = [], eb = [];
      {
        const idx = prim.indices !== undefined ? doc.accessor(prim.indices).data : null;
        const count = idx ? idx.length : nv;
        const mode = prim.mode === undefined ? 4 : prim.mode;
        const vtx = i => (idx ? idx[i] : i);
        const pushEdge = (p, q) => {
          const kk = p < q ? p * nv + q : q * nv + p;
          if (!edgeKeys.has(kk)) { edgeKeys.set(kk, ea.length); ea.push(Math.min(p, q)); eb.push(Math.max(p, q)); }
        };
        const pushTri = (a, b, c) => { pushEdge(a, b); pushEdge(b, c); pushEdge(a, c); };
        if (mode === 4) for (let i = 0; i + 2 < count; i += 3) pushTri(vtx(i), vtx(i + 1), vtx(i + 2));
        else if (mode === 5) for (let i = 0; i + 2 < count; i++) pushTri(vtx(i), vtx(i + 1), vtx(i + 2));
        else if (mode === 6) for (let i = 1; i + 1 < count; i++) pushTri(vtx(0), vtx(i), vtx(i + 1));
      }
      const ne = ea.length;
      const eLimb = new Uint8Array(ne);     // both endpoints inside a limb volume
      const bindLen = new Float64Array(ne);
      for (let e = 0; e < ne; e++) {
        const a = ea[e], b = eb[e];
        eLimb[e] = inLimb[a] && inLimb[b] ? 1 : 0;
        bindLen[e] = Math.hypot(
          pos.data[a * 3] - pos.data[b * 3],
          pos.data[a * 3 + 1] - pos.data[b * 3 + 1],
          pos.data[a * 3 + 2] - pos.data[b * 3 + 2]);
      }
      prims.push({
        nIdx, node, pos, jnt, wgt, nv, jointsArr, ibm, inLimb,
        ea: Uint32Array.from(ea), eb: Uint32Array.from(eb), ne, eLimb, bindLen,
        startLen: new Float64Array(ne),
        softFlag: new Uint8Array(ne), hardFlag: new Uint8Array(ne),
        cur: new Float64Array(nv * 3), skinned: null,
      });
    }
  }
  if (!prims.length) return failBoth('no skinned primitives (POSITION+JOINTS_0+WEIGHTS_0) found');

  // ---- weld groups across all primitives (bind positions, SEAM_WELD_EPS).
  const eps = T.SEAM_WELD_EPS;
  const groups = []; // arrays of { p: primIndex, v }
  {
    const cell = new Map(); // "x:y:z" -> array of { p, v, gi }
    for (let pi = 0; pi < prims.length; pi++) {
      const { pos, nv } = prims[pi];
      for (let v = 0; v < nv; v++) {
        const x = pos.data[v * 3], y = pos.data[v * 3 + 1], z = pos.data[v * 3 + 2];
        const cx = Math.round(x / eps), cy = Math.round(y / eps), cz = Math.round(z / eps);
        let gi = -1;
        outer:
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
          const list = cell.get(`${cx + dx}:${cy + dy}:${cz + dz}`);
          if (!list) continue;
          for (const e of list) {
            const q = prims[e.p].pos.data;
            if (Math.abs(q[e.v * 3] - x) <= eps && Math.abs(q[e.v * 3 + 1] - y) <= eps && Math.abs(q[e.v * 3 + 2] - z) <= eps) {
              gi = e.gi; break outer;
            }
          }
        }
        if (gi === -1) { gi = groups.length; groups.push([]); }
        groups[gi].push({ p: pi, v });
        const ck = `${cx}:${cy}:${cz}`;
        if (!cell.has(ck)) cell.set(ck, []);
        cell.get(ck).push({ p: pi, v, gi });
      }
    }
  }
  const multiGroups = groups.filter(g => g.length >= 2);
  const groupMaxSep = new Float64Array(multiGroups.length);

  // ---- frame loop: skin everything; edges vs bind + vs clip start; seams.
  const strain = {
    maxRatio: 1, maxRatioWhere: null, maxOpenFrac: 0, maxOpenWhere: null,
    totalEdges: prims.reduce((a, p) => a + p.ne, 0),
  };
  const minOpen = T.EDGE_MIN_OPEN_FRAC * height;
  const absOpen = T.EDGE_ABS_OPEN_FRAC * height;

  for (let f = 0; f < times.length; f++) {
    const g = pose.globalsAt(times[f]);
    for (const pr of prims) {
      const jm = pr.jointsArr.map((j, k) => mat4Mul(g[j], pr.ibm[k]));
      const { pos, jnt, wgt, nv, cur } = pr;
      for (let v = 0; v < nv; v++) {
        const p = [pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2]];
        let ox = 0, oy = 0, oz = 0, tw = 0;
        for (let c = 0; c < 4; c++) {
          const w = wgt.data[v * 4 + c];
          if (w === 0) continue;
          tw += w;
          const q = mat4TransformPoint(jm[jnt.data[v * 4 + c]], p);
          ox += w * q[0]; oy += w * q[1]; oz += w * q[2];
        }
        if (tw === 0) { ox = p[0]; oy = p[1]; oz = p[2]; } // unskinned: static
        cur[v * 3] = ox; cur[v * 3 + 1] = oy; cur[v * 3 + 2] = oz;
      }
      const { ea, eb, ne, eLimb, bindLen, startLen, softFlag, hardFlag } = pr;
      for (let e = 0; e < ne; e++) {
        const a = ea[e], b = eb[e];
        const len = Math.hypot(cur[a * 3] - cur[b * 3], cur[a * 3 + 1] - cur[b * 3 + 1], cur[a * 3 + 2] - cur[b * 3 + 2]);
        if (f === 0) startLen[e] = len;
        const softR = eLimb[e] ? T.EDGE_SOFT_RATIO_LIMB : T.EDGE_SOFT_RATIO_BODY;
        const hardR = eLimb[e] ? T.EDGE_HARD_RATIO_LIMB : T.EDGE_HARD_RATIO_BODY;
        for (const base of f === 0 ? [bindLen[e]] : [bindLen[e], startLen[e]]) {
          const open = len - base;
          if (open <= minOpen) continue; // shrinking / sub-visible: never an offense
          const ratio = base > 1e-12 ? len / base : Infinity;
          if (ratio > strain.maxRatio) {
            strain.maxRatio = ratio;
            strain.maxRatioWhere = { clipTime: times[f], limb: !!eLimb[e], openFrac: open / height };
          }
          if (open / height > strain.maxOpenFrac) {
            strain.maxOpenFrac = open / height;
            strain.maxOpenWhere = { clipTime: times[f], limb: !!eLimb[e], ratio };
          }
          if (ratio > softR) softFlag[e] = 1;
          if (ratio > hardR || open > absOpen) hardFlag[e] = 1;
        }
      }
    }
    for (let gi = 0; gi < multiGroups.length; gi++) {
      const g2 = multiGroups[gi];
      let worst = 0;
      for (let i = 0; i < g2.length; i++) {
        const ci = prims[g2[i].p].cur, vi = g2[i].v;
        for (let j2 = i + 1; j2 < g2.length; j2++) {
          const cj = prims[g2[j2].p].cur, vj = g2[j2].v;
          const d = Math.hypot(ci[vi * 3] - cj[vj * 3], ci[vi * 3 + 1] - cj[vj * 3 + 1], ci[vi * 3 + 2] - cj[vj * 3 + 2]);
          if (d > worst) worst = d;
        }
      }
      if (worst > groupMaxSep[gi]) groupMaxSep[gi] = worst;
    }
  }

  // ---- verdicts.
  let softCount = 0, hardCount = 0, softLimb = 0;
  for (const pr of prims) {
    for (let e = 0; e < pr.ne; e++) {
      if (pr.softFlag[e]) { softCount++; if (pr.eLimb[e]) softLimb++; }
      if (pr.hardFlag[e]) hardCount++;
    }
  }
  const softFrac = softCount / strain.totalEdges;
  const strainOk = softFrac <= T.EDGE_OFFENDER_RATIO && hardCount === 0;
  const strainRow = {
    ok: strainOk,
    maxRatio: strain.maxRatio, maxOpenFrac: strain.maxOpenFrac,
    softCount, softFrac, hardCount, totalEdges: strain.totalEdges, samples: times.length,
    detail: `${softCount} of ${strain.totalEdges} edges (${(softFrac * 100).toFixed(2)}%) stretched past ` +
      `${T.EDGE_SOFT_RATIO_BODY}x (limb-interior ${T.EDGE_SOFT_RATIO_LIMB}x; ${softLimb} of them limb) vs bind or clip start ` +
      `(allowed <= ${T.EDGE_OFFENDER_RATIO * 100}%); ${hardCount} edge(s) past the hard cap ` +
      `${T.EDGE_HARD_RATIO_BODY}x (limb ${T.EDGE_HARD_RATIO_LIMB}x) or opening > ${T.EDGE_ABS_OPEN_FRAC * 100}% h (allowed 0); ` +
      `max ratio ${Number.isFinite(strain.maxRatio) ? strain.maxRatio.toFixed(2) + 'x' : 'inf'}` +
      (strain.maxRatioWhere ? ` (${strain.maxRatioWhere.limb ? 'limb' : 'body'} edge, t=${strain.maxRatioWhere.clipTime.toFixed(2)}s)` : '') +
      `, max opening ${(strain.maxOpenFrac * 100).toFixed(2)}% h; ratio offenses require opening > ` +
      `${T.EDGE_MIN_OPEN_FRAC * 100}% h; ${times.length} frames`,
  };

  const sepCap = T.SEAM_MAX_SEPARATION_FRAC * height;
  let badGroups = 0, worstSep = 0, diffWeightGroups = 0;
  for (let gi = 0; gi < multiGroups.length; gi++) {
    if (groupMaxSep[gi] > sepCap) badGroups++;
    if (groupMaxSep[gi] > worstSep) worstSep = groupMaxSep[gi];
  }
  for (const g2 of multiGroups) {
    const sig = m => {
      const { jnt, wgt } = prims[m.p];
      const pairs = [];
      for (let c = 0; c < 4; c++) pairs.push([jnt.data[m.v * 4 + c], +wgt.data[m.v * 4 + c].toFixed(6)]);
      return JSON.stringify(pairs.sort((x, y) => x[0] - y[0]));
    };
    const first = sig(g2[0]);
    if (g2.some(m => sig(m) !== first)) diffWeightGroups++;
  }
  const seamsOk = badGroups === 0;
  const seamRow = {
    ok: seamsOk,
    groups: multiGroups.length, diffWeightGroups, badGroups,
    maxSepFrac: worstSep / height, samples: times.length,
    detail: `${multiGroups.length} weld group(s) of vertices co-located within ${T.SEAM_WELD_EPS} at bind ` +
      `(${diffWeightGroups} with differing weights); ${badGroups} group(s) split past ` +
      `${T.SEAM_MAX_SEPARATION_FRAC * 100}% h (allowed 0); worst separation ` +
      `${((worstSep / height) * 100).toFixed(2)}% h; ${times.length} frames`,
  };

  return done({ clip: clipName, strain: strainRow, seams: seamRow });
}

/** V8: all-edge strain, every clip, vs bind AND vs clip start. */
function checkEdgeStrain(docs) {
  const rows = [];
  let pass = true;
  for (const group of REQUIRED_CLIPS) {
    const hit = findClip(docs, group);
    if (!hit) { rows.push({ clip: group.join('|'), ok: false, detail: 'clip not found' }); pass = false; continue; }
    const scan = deformScanClip(hit.doc, hit.anim, hit.clipName);
    const r = { clip: hit.clipName, ...scan.strain };
    rows.push(r);
    if (!r.ok) pass = false;
  }
  const T = THRESHOLDS;
  return {
    id: 'edgeStrain',
    title: `V8 edge strain: ALL edges, every clip, vs bind AND clip start ` +
      `(<= ${T.EDGE_OFFENDER_RATIO * 100}% of edges past ${T.EDGE_SOFT_RATIO_BODY}x/` +
      `${T.EDGE_SOFT_RATIO_LIMB}x-limb; none past ${T.EDGE_HARD_RATIO_BODY}x/` +
      `${T.EDGE_HARD_RATIO_LIMB}x-limb or opening > ${T.EDGE_ABS_OPEN_FRAC * 100}% h)`,
    pass,
    measured: rows.map(r => `${r.clip}: ${r.ok ? 'ok' : 'FAIL'}`).join('; '),
    details: { rows },
  };
}

/** V9: UV-seam cracks — weld groups must not split in any clip. */
function checkSeamCracks(docs) {
  const rows = [];
  let pass = true;
  for (const group of REQUIRED_CLIPS) {
    const hit = findClip(docs, group);
    if (!hit) { rows.push({ clip: group.join('|'), ok: false, detail: 'clip not found' }); pass = false; continue; }
    const scan = deformScanClip(hit.doc, hit.anim, hit.clipName);
    const r = { clip: hit.clipName, ...scan.seams };
    rows.push(r);
    if (!r.ok) pass = false;
  }
  return {
    id: 'seamCracks',
    title: `V9 seam cracks: vertices co-located within ${THRESHOLDS.SEAM_WELD_EPS} at bind stay within ` +
      `${THRESHOLDS.SEAM_MAX_SEPARATION_FRAC * 100}% h of each other in every clip`,
    pass,
    measured: rows.map(r => `${r.clip}: ${r.ok ? 'ok' : 'FAIL'}`).join('; '),
    details: { rows },
  };
}

/**
 * V4' reverse leakage: ZERO vertices DOMINATED by an arm-chain joint
 * (clavicle/shoulder included) inside the GEOMETRIC head region — bind
 * height above the topmost neck joint AND within HEAD_REGION_RADIUS_FRAC*h
 * of the head joints' bind bone star. V4 tests head-DOMINATED vertices
 * only, so a cheek tuft whose dominant joint is RightShoulder never
 * entered it (proven: RightShoulder owned 393 vertices above the neck).
 * Static check — bind pose only.
 */
function checkReverseLeakage(docs) {
  const T = THRESHOLDS;
  const rows = [];
  let pass = true;
  for (const doc of docs) {
    const height = modelHeight(doc);
    const cls = classifyJoints(doc);
    if (!cls.joints.size || !(height > 0)) {
      rows.push({ file: doc.label, ok: false, detail: 'no skin joints / zero model height' });
      pass = false;
      continue;
    }
    if (!cls.headSeeds.size) {
      rows.push({ file: doc.label, ok: true, detail: 'no head-named joints — no head region to protect (pattern list may need extending)' });
      continue;
    }
    const bind = jointBindMap(doc);
    const parents = doc.parents();
    const pos = j => bind.get(j) || restWorldPos(doc, j);
    // neck height: topmost (max bind Y) neck-named joint; fallback: midpoint
    // between the topmost head joint and its parent joint.
    let neckY = -Infinity, neckName = null;
    for (const j of cls.neckSeeds) {
      const y = pos(j)[1];
      if (y > neckY) { neckY = y; neckName = doc.nodeName(j); }
    }
    if (neckY === -Infinity) {
      let H = null, hd = Infinity;
      for (const j of cls.headSeeds) {
        const d = cls.depth.get(j) || 0;
        if (d < hd) { hd = d; H = j; }
      }
      const p = parents.get(H);
      neckY = p !== undefined ? (pos(H)[1] + pos(p)[1]) / 2 : pos(H)[1] - 0.05 * height;
      neckName = `midpoint(${doc.nodeName(H)}, parent) [no neck joint]`;
    }
    // head bone star: bind segments incident to head-named joints.
    const headSegs = [];
    (doc.json.skins || []).forEach((skin, si) => {
      const sd = limbSkinData(doc, si, height);
      (skin.joints || []).forEach((j, k) => {
        if (!cls.headSeeds.has(j)) return;
        const star = sd.star[k];
        for (let s = 0; s < star.length; s += 6) headSegs.push(Array.from(star.slice(s, s + 6)));
      });
    });
    const headPacked = new Float64Array(headSegs.length * 6);
    headSegs.forEach((s, i) => headPacked.set(s, i * 6));
    const radius = T.HEAD_REGION_RADIUS_FRAC * height;
    const inHeadRegion = (x, y, z) => {
      if (y <= neckY) return false;
      for (let s = 0; s < headPacked.length; s += 6) {
        if (distPointSeg(x, y, z, headPacked, s) <= radius) return true;
      }
      return false;
    };

    let regionVerts = 0, violations = 0;
    let worst = { y: -Infinity, joint: null, vert: -1, node: null };
    const tally = new Map();
    const rendered = renderedNodeSet(doc);
    for (const nIdx of rendered) {
      const node = doc.json.nodes[nIdx];
      if (!node || node.mesh === undefined || node.skin === undefined) continue;
      const jointsArr = doc.json.skins[node.skin].joints;
      for (const prim of doc.json.meshes[node.mesh].primitives || []) {
        const at = prim.attributes || {};
        if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) continue;
        const posAcc = doc.accessor(at.POSITION);
        const jnt = doc.accessor(at.JOINTS_0);
        const wgt = doc.accessor(at.WEIGHTS_0);
        for (let v = 0; v < posAcc.count; v++) {
          const x = posAcc.data[v * 3], y = posAcc.data[v * 3 + 1], z = posAcc.data[v * 3 + 2];
          if (!inHeadRegion(x, y, z)) continue;
          regionVerts++;
          let domK = -1, domW = -1;
          for (let c = 0; c < 4; c++) {
            const w = wgt.data[v * 4 + c];
            if (w > domW) { domW = w; domK = jnt.data[v * 4 + c]; }
          }
          if (domW <= 0) continue;
          const j = jointsArr[domK];
          if (j === undefined || !cls.armAll.has(j)) continue;
          violations++;
          const name = doc.nodeName(j);
          tally.set(name, (tally.get(name) || 0) + 1);
          if (y > worst.y) worst = { y, joint: name, vert: v, node: doc.nodeName(nIdx) };
        }
      }
    }
    const ok = violations === 0;
    const tallyStr = [...tally.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4)
      .map(([n, c]) => `${n}:${c}`).join(', ');
    rows.push({
      file: doc.label, ok, regionVerts, violations,
      detail: `head region = bind y > ${neckY.toFixed(3)} ("${neckName}") and <= ` +
        `${T.HEAD_REGION_RADIUS_FRAC * 100}% h from the head bone star; ${regionVerts} vertices in region, ` +
        `${violations} DOMINATED by an arm-chain joint (allowed 0)` +
        (violations ? ` — by joint: ${tallyStr}; highest at y ${worst.y.toFixed(3)} (${worst.joint})` : ''),
    });
    if (!ok) pass = false;
  }
  return {
    id: 'reverseLeakage',
    title: `V4' reverse leakage: zero arm-chain-DOMINATED vertices (clavicle included) in the geometric head region`,
    pass,
    measured: rows.map(r => `${r.file}: ${r.ok ? 'ok' : 'FAIL'} — ${r.detail}`).join('; '),
    details: { rows },
  };
}

/**
 * V7' bind-time mis-binding + limb-scale sanity. V7 measures motion
 * relative to the bind wrap distance, so it is invariant BY CONSTRUCTION
 * for rigid (single-joint) binds and blind to a wrong-but-rigid rig; V7'
 * judges the BINDING itself, at bind:
 *
 *   dOwn  = min distance from the vertex to the bone stars of its weighted
 *           joints (any weight > 0);
 *   dNear = min distance to ANY bone segment of the skin.
 *
 *   soft offender: dOwn > MISBIND_OWN_FAR_FRAC*h  AND
 *                  dNear < MISBIND_OTHER_NEAR_FRAC*h AND
 *                  dOwn > MISBIND_DOMINANCE * dNear
 *     (someone ELSE's bone explains the vertex several times better than
 *      every bone it is weighted to — a paw chunk bound to the hips).
 *     Allowed fraction: MISBIND_OFFENDER_RATIO.
 *
 *   hard offender: ANY single influence with weight >= MISBIND_HARD_WEIGHT
 *     whose own star is > MISBIND_HARD_FAR_FRAC*h away while
 *     dNear < MISBIND_OTHER_NEAR_FRAC*h and that influence's distance
 *     > MISBIND_DOMINANCE * dNear. Catches small blended fragments
 *     (0.55 Hand / 0.45 Spine2) whose DOMINANT bone is nearby. Allowed: 0.
 *
 *   scale sanity: every skin joint's scale (rest pose and every animated
 *     scale value in every clip) must be 1 +- JOINT_SCALE_TOL per
 *     component (a joint scaled 1.6x balloons its skin while V7's rigid
 *     transport stays exactly invariant).
 *
 * Legitimately-far skin (muzzle, ears, belly, tail tip) is safe in
 * principle: its OWN bone is also the NEAREST bone, so dOwn = dNear and
 * the dominance factor can never fire.
 */
function checkMisBinding(docs) {
  const T = THRESHOLDS;
  const rows = [];
  let pass = true;
  for (const doc of docs) {
    const height = modelHeight(doc);
    const cls = classifyJoints(doc);
    if (!cls.joints.size || !(height > 0)) {
      rows.push({ file: doc.label, ok: false, detail: 'no skin joints / zero model height' });
      pass = false;
      continue;
    }
    // all bind bone segments of every skin.
    const skinsData = (doc.json.skins || []).map((_, si) => limbSkinData(doc, si, height));
    let segCount = 0;
    for (const sd of skinsData) segCount += sd.segs.length;
    const allSegs = new Float64Array(segCount * 6);
    {
      let o = 0;
      for (const sd of skinsData) {
        for (const [pk, k] of sd.segs) {
          allSegs.set(sd.bindPos[pk], o);
          allSegs.set(sd.bindPos[k], o + 3);
          o += 6;
        }
      }
    }
    const distAll = (x, y, z) => {
      let best = Infinity;
      for (let s = 0; s < allSegs.length; s += 6) {
        const d = distPointSeg(x, y, z, allSegs, s);
        if (d < best) best = d;
      }
      return best;
    };
    const distToStar = (star, x, y, z) => {
      let best = Infinity;
      for (let s = 0; s < star.length; s += 6) {
        const d = distPointSeg(x, y, z, star, s);
        if (d < best) best = d;
      }
      return best;
    };
    const FAR = T.MISBIND_OWN_FAR_FRAC * height;
    const NEAR = T.MISBIND_OTHER_NEAR_FRAC * height;
    const HARD_FAR = T.MISBIND_HARD_FAR_FRAC * height;

    let total = 0, soft = 0, hard = 0;
    let softWorst = null, hardWorst = null;
    const rendered = renderedNodeSet(doc);
    for (const nIdx of rendered) {
      const node = doc.json.nodes[nIdx];
      if (!node || node.mesh === undefined || node.skin === undefined) continue;
      const sd = skinsData[node.skin];
      for (const prim of doc.json.meshes[node.mesh].primitives || []) {
        const at = prim.attributes || {};
        if (at.POSITION === undefined || at.JOINTS_0 === undefined || at.WEIGHTS_0 === undefined) continue;
        const posAcc = doc.accessor(at.POSITION);
        const jnt = doc.accessor(at.JOINTS_0);
        const wgt = doc.accessor(at.WEIGHTS_0);
        for (let v = 0; v < posAcc.count; v++) {
          let tw = 0;
          for (let c = 0; c < 4; c++) tw += wgt.data[v * 4 + c];
          if (!(tw > 0)) continue;
          total++;
          const x = posAcc.data[v * 3], y = posAcc.data[v * 3 + 1], z = posAcc.data[v * 3 + 2];
          let dOwn = Infinity;
          const dInf = [Infinity, Infinity, Infinity, Infinity];
          for (let c = 0; c < 4; c++) {
            if (wgt.data[v * 4 + c] === 0) continue;
            dInf[c] = distToStar(sd.star[jnt.data[v * 4 + c]], x, y, z);
            if (dInf[c] < dOwn) dOwn = dInf[c];
          }
          const dNear = (dOwn > FAR || dInf.some((d, c2) => wgt.data[v * 4 + c2] >= T.MISBIND_HARD_WEIGHT && d > HARD_FAR))
            ? distAll(x, y, z) : Infinity;
          if (dOwn > FAR && dNear < NEAR && dOwn > T.MISBIND_DOMINANCE * dNear) {
            soft++;
            if (!softWorst || dOwn - dNear > softWorst.gap) {
              softWorst = {
                gap: dOwn - dNear, vert: v, node: doc.nodeName(nIdx),
                dOwnFrac: dOwn / height, dNearFrac: dNear / height,
              };
            }
          }
          for (let c = 0; c < 4; c++) {
            const w = wgt.data[v * 4 + c];
            if (w < T.MISBIND_HARD_WEIGHT) continue;
            if (dInf[c] > HARD_FAR && dNear < NEAR && dInf[c] > T.MISBIND_DOMINANCE * dNear) {
              hard++;
              if (!hardWorst || dInf[c] > hardWorst.dInf) {
                hardWorst = {
                  dInf: dInf[c], dInfFrac: dInf[c] / height, weight: w, vert: v,
                  joint: doc.nodeName(sd.jointsArr[jnt.data[v * 4 + c]]),
                  dNearFrac: dNear / height, node: doc.nodeName(nIdx),
                };
              }
              break;
            }
          }
        }
      }
    }

    // limb-scale sanity: rest scales + every animated scale value.
    let scaleWorst = { dev: 0, joint: null, clip: null };
    for (const j of cls.joints) {
      const s = (doc.json.nodes[j] || {}).scale;
      if (!s) continue;
      for (const comp of s) {
        const dev = Math.abs(comp - 1);
        if (dev > scaleWorst.dev) scaleWorst = { dev, joint: doc.nodeName(j), clip: '(rest pose)' };
      }
    }
    for (const anim of doc.json.animations || []) {
      for (const ch of anim.channels || []) {
        if (!ch.target || ch.target.path !== 'scale') continue;
        if (!cls.joints.has(ch.target.node)) continue;
        const sampler = anim.samplers[ch.sampler];
        const out = doc.accessor(sampler.output);
        const cubic = (sampler.interpolation || 'LINEAR') === 'CUBICSPLINE';
        const keys = cubic ? out.count / 3 : out.count;
        for (let k = 0; k < keys; k++) {
          const base = (cubic ? k * 3 + 1 : k) * 3;
          for (let c = 0; c < 3; c++) {
            const dev = Math.abs(out.data[base + c] - 1);
            if (dev > scaleWorst.dev) {
              scaleWorst = { dev, joint: doc.nodeName(ch.target.node), clip: anim.name || '(unnamed clip)' };
            }
          }
        }
      }
    }
    const scaleOk = scaleWorst.dev <= T.JOINT_SCALE_TOL;

    if (!total) {
      rows.push({ file: doc.label, ok: false, detail: 'no vertices carry any skin weight' });
      pass = false;
      continue;
    }
    const ratio = soft / total;
    const ok = ratio <= T.MISBIND_OFFENDER_RATIO && hard === 0 && scaleOk;
    rows.push({
      file: doc.label, ok, total, soft, softRatio: ratio, hard,
      scaleWorstDev: scaleWorst.dev,
      detail: `${soft} of ${total} skinned vertices (${(ratio * 100).toFixed(2)}%) mis-bound ` +
        `(all own bones > ${T.MISBIND_OWN_FAR_FRAC * 100}% h away, another bone < ` +
        `${T.MISBIND_OTHER_NEAR_FRAC * 100}% h and ${T.MISBIND_DOMINANCE}x closer; allowed <= ` +
        `${T.MISBIND_OFFENDER_RATIO * 100}%)` +
        (softWorst ? ` — worst: own ${(softWorst.dOwnFrac * 100).toFixed(1)}% h vs other ${(softWorst.dNearFrac * 100).toFixed(1)}% h` : '') +
        `; ${hard} hard offender(s) (an influence >= ${T.MISBIND_HARD_WEIGHT} bound > ` +
        `${T.MISBIND_HARD_FAR_FRAC * 100}% h from its bones while another bone is < ` +
        `${T.MISBIND_OTHER_NEAR_FRAC * 100}% h; allowed 0)` +
        (hardWorst ? ` — worst: ${(hardWorst.weight).toFixed(2)} on "${hardWorst.joint}" at ${(hardWorst.dInfFrac * 100).toFixed(1)}% h (nearest other ${(hardWorst.dNearFrac * 100).toFixed(1)}% h)` : '') +
        `; joint scale ${scaleOk ? 'ok' : 'FAIL'} (max |scale-1| = ${scaleWorst.dev.toExponential(2)}` +
        (scaleWorst.joint ? ` on "${scaleWorst.joint}" in ${scaleWorst.clip}` : '') +
        `, tolerance ${T.JOINT_SCALE_TOL})`,
    });
    if (!ok) pass = false;
  }
  return {
    id: 'misBinding',
    title: `V7' mis-binding at bind: <= ${THRESHOLDS.MISBIND_OFFENDER_RATIO * 100}% of vertices bound only to far bones ` +
      `while another bone is near; zero hard offenders; joint scales = 1 +- ${THRESHOLDS.JOINT_SCALE_TOL}`,
    pass,
    measured: rows.map(r => `${r.file}: ${r.ok ? 'ok' : 'FAIL'} — ${r.detail}`).join('; '),
    details: { rows },
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
  checks.push(guard(() => checkWeightLeakage(docs)));
  checks.push(guard(() => checkLoopSeams(docs)));
  checks.push(guard(() => checkJump(docs)));
  checks.push(guard(() => checkLimbIntegrity(docs)));
  checks.push(guard(() => checkEdgeStrain(docs)));
  checks.push(guard(() => checkSeamCracks(docs)));
  checks.push(guard(() => checkReverseLeakage(docs)));
  checks.push(guard(() => checkMisBinding(docs)));

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
    if (c.id === 'limbIntegrity' && c.details?.rows) {
      for (const r of c.details.rows) {
        const b = r.ok ? pass(' ok ') : failc('FAIL');
        console.log(`      [${b}] ${r.clip}: ${r.detail}`);
      }
    }
    if (['edgeStrain', 'seamCracks'].includes(c.id) && c.details?.rows) {
      for (const r of c.details.rows) {
        const b = r.ok ? pass(' ok ') : failc('FAIL');
        console.log(`      [${b}] ${r.clip}: ${r.detail}`);
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
