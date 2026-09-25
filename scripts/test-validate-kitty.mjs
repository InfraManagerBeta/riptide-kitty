/**
 * test-validate-kitty.mjs — self-test for scripts/validate-kitty.mjs.
 *
 * Builds SYNTHETIC skinned GLBs entirely in memory (a torso+head+arms mesh
 * with a 10-joint skeleton — Root/Hips/Spine/Head/feet/arms — and an
 * embedded 1x1 PNG base-color texture) and asserts that the validator
 * passes a correct fixture and fails each broken fixture on its SPECIFIC
 * check — including one fixture per adversarial mutation the round-1
 * reviewer used to defeat the old validator (A1–A6).
 *
 * Run:  node --test scripts/test-validate-kitty.mjs
 *   or: node scripts/test-validate-kitty.mjs
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { validate, THRESHOLDS, parseGLB, tokenizeName, sniffImage } from './validate-kitty.mjs';

/* ========================================================================== *
 * Minimal GLB writer.
 * ========================================================================== */

class Bin {
  constructor() { this.parts = []; this.len = 0; }
  align(n = 4) {
    const pad = (n - (this.len % n)) % n;
    if (pad) { this.parts.push(new Uint8Array(pad)); this.len += pad; }
  }
  push(typed) {
    this.align(4);
    const off = this.len;
    const u8 = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    this.parts.push(u8); this.len += u8.length;
    return { byteOffset: off, byteLength: u8.length };
  }
  bytes() {
    this.align(4);
    const out = new Uint8Array(this.len);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    return out;
  }
}

function buildGLB(json, bin) {
  const enc = new TextEncoder();
  let jsonBytes = enc.encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  if (jsonPad) {
    const padded = new Uint8Array(jsonBytes.length + jsonPad);
    padded.set(jsonBytes); padded.fill(0x20, jsonBytes.length);
    jsonBytes = padded;
  }
  const total = 12 + 8 + jsonBytes.length + (bin.length ? 8 + bin.length : 0);
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true); // glTF
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonBytes.length, true);
  dv.setUint32(16, 0x4e4f534a, true); // JSON
  out.set(jsonBytes, 20);
  if (bin.length) {
    const o = 20 + jsonBytes.length;
    dv.setUint32(o, bin.length, true);
    dv.setUint32(o + 4, 0x004e4942, true); // BIN
    out.set(bin, o + 8);
  }
  return out;
}

/** 1x1 white PNG. */
const PNG_1x1 = Uint8Array.from(Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
));

/* ========================================================================== *
 * Synthetic kitty fixture.
 *
 * Skeleton (Y-up, joint world positions), deliberately with an un-animated
 * `Root` ABOVE `Hips` (the reviewer's A6 mutation hid a hips lurch under a
 * static Root):
 *
 *   Root (0,0,0)
 *    └ Hips (0,1,0)
 *       ├ L_Foot (0.1,0.1,0)   R_Foot (-0.1,0.1,0)
 *       └ Spine (0,1.4,0)
 *          ├ Head (0,1.9,0)
 *          ├ L_Arm (0.35,1.55,0) → L_Hand (0.65,1.55,0)
 *          └ R_Arm (-0.35,1.55,0) → R_Hand (-0.65,1.55,0)
 *
 * Mesh: four boxes — torso (y 0.2..1.8), head (y 1.75..2.05 ⇒ model height
 * 1.85), left arm, right arm. Torso verts weighted to Hips/Spine; head
 * verts to Head; arm verts to L_Arm / R_Arm. Names deliberately exercise
 * `L_` / `R_` prefix tokens.
 * ========================================================================== */

export const MODEL_HEIGHT = 1.85;

function rotZ(deg) {
  const a = (deg * Math.PI / 180) / 2;
  return [0, 0, Math.sin(a), Math.cos(a)];
}
function rotX(deg) {
  const a = (deg * Math.PI / 180) / 2;
  return [Math.sin(a), 0, 0, Math.cos(a)];
}
function invTranslation(p) { // inverse bind matrix for a pure-translation joint
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -p[0], -p[1], -p[2], 1];
}

function boxVerts(cx, cy, cz, hx, hy, hz) {
  const v = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    v.push([cx + sx * hx, cy + sy * hy, cz + sz * hz]);
  }
  return v;
}
const BOX_TRIS = [ // 12 triangles over the 8-corner ordering above
  [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
];

/**
 * Build one synthetic kitty GLB.
 * opts:
 *   clips: array of clip names to embed (default all four)
 *   walkName: 'walk' | 'run'
 *   skin: include skin + JOINTS_0/WEIGHTS_0 (default true)
 *   texture: true | false | 'zeroed'  (zeroed = image bytes all 0x00 — A5)
 *   hugeTriangles: exceed the triangle budget with plain geometry
 *   instancing: 0 | N — put EXT_mesh_gpu_instancing xN on the mesh node (A4)
 *   jumpGrounded: jump clip never lifts the feet (V6 fail)
 *   walkDrift: walk clip's hips translate away and never return (V5 fail)
 *   wave:
 *     'good'          clean LEFT-arm wave (default; must PASS everything)
 *     'goodRight'     clean RIGHT-arm wave (side selection must follow)
 *     'leftCheerRight' clean LEFT wave; the static RIGHT arm's BIND pose is
 *                     an arms-up "cheer" (A1's side-pick trap — must PASS)
 *     'limp'          A1: bind pose arms-up, in-clip the arm hangs at the
 *                     hip and the forearm wiggles ±9°
 *     'small'         arm barely moves (10°)
 *     'oneOsc'        a single raise-lower
 *     'lurch'         A6: hips translate under the static Root
 *     'spineRotate'   the spine rotates with the wave
 *     'codeform'      A3: ONE torso vertex co-weighted to the arm (small
 *                     fraction — a p95 would hide it)
 *     'headLeak'      A2: head vertices 40% weighted to the arm
 *     'bothArms'      the other arm waves too (V1 fail)
 *     'strain'        a tiny all-body triangle stretches > 1.5x while every
 *                     displacement stays under 3% h (V3 strain fail)
 *     'doubledLimb'   a detached second "paw" box under the left arm,
 *                     DOMINATED by the upper-arm joint (V3-exempt) but
 *                     blended with static joints, so it lags behind and
 *                     sticks out when the arm raises (V7 fail; every other
 *                     check passes — the defect the older checks miss)
 *   articulated: walk swings both arms (loop-closed) and jump tucks them —
 *                correctly skinned limb articulation in walk/jump (V7 pass)
 *   --- fix round 3 (V3'/V4'/V7'/V8/V9 blind-spot fixtures) ---
 *   rebindPawHips: the outer paw chunk of the left arm (4 verts) rebound
 *                100% to Hips — rides rigidly, V7 invariant BY CONSTRUCTION
 *                (d_t = d_bind), V7' must flag the binding at bind time
 *   rebindTorsoArm: 4 lower-torso verts rebound 100% to the waving upper
 *                arm — the spec's own named disqualifier ("an arm welded to
 *                the torso"); the old V3 exempted them by dominant joint,
 *                V3' geometric body membership must fail them
 *   blendMisbind: the paw chunk blended 0.55 L_Hand / 0.45 Spine — under
 *                the 0.2% soft ratio (dominant bone is nearby) but a HARD
 *                V7' offender (a 0.45 influence bound to far bones)
 *   armScale:    animated scale on the upper-arm joint (e.g. 1.6) — V7's
 *                rigid transport is exactly invariant to it; V7' scale
 *                sanity must fail
 *   seamPair:    two co-located verts on the arm, one bound to the arm and
 *                one to the spine — a UV-seam duplicate pair that cracks
 *                open in wave (V9)
 *   cheekShoulder: 4 head-top verts DOMINATED by the right arm — V4 (head-
 *                dominated only) is blind to it; V4' reverse leakage fails
 *   boundaryTear: a tiny triangle at the arm/torso boundary whose short
 *                edge tears ~10x when the arm raises (V8; the old V3 strain
 *                dropped every triangle touching a waving-arm vertex)
 */
export function makeKitty(opts = {}) {
  const {
    clips = ['idle', 'jump', 'walk', 'wave'],
    walkName = 'walk',
    skin = true,
    texture = true,
    hugeTriangles = false,
    instancing = 0,
    jumpGrounded = false,
    walkDrift = false,
    wave = 'good',
    articulated = false,
    rebindPawHips = false,
    rebindTorsoArm = false,
    blendMisbind = false,
    armScale = 0,
    seamPair = false,
    cheekShoulder = false,
    boundaryTear = false,
  } = opts;

  const bin = new Bin();
  const bufferViews = [];
  const accessors = [];
  const addBV = (typed, extra = {}) => {
    const { byteOffset, byteLength } = bin.push(typed);
    bufferViews.push({ buffer: 0, byteOffset, byteLength, ...extra });
    return bufferViews.length - 1;
  };
  const addAccessor = (typed, componentType, type, extra = {}) => {
    const nc = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[type];
    const bv = addBV(typed);
    accessors.push({ bufferView: bv, componentType, count: typed.length / nc, type, ...extra });
    return accessors.length - 1;
  };

  // --- geometry ---------------------------------------------------------
  const torso = boxVerts(0, 1.0, 0, 0.25, 0.8, 0.2);        // verts 0-7,  y 0.2 .. 1.8
  const head = boxVerts(0, 1.9, 0, 0.12, 0.15, 0.12);       // verts 8-15, y 1.75 .. 2.05
  const armL = boxVerts(0.575, 1.55, 0, 0.225, 0.05, 0.05); // verts 16-23
  const armR = boxVerts(-0.575, 1.55, 0, 0.225, 0.05, 0.05);// verts 24-31
  const verts = [...torso, ...head, ...armL, ...armR];

  const indicesArr = [];
  for (let b = 0; b < 4; b++) for (const t of BOX_TRIS) indicesArr.push(...t.map(i => i + b * 8));

  // 'strain' adds a tiny torso-surface triangle (verts 32,33,34) whose apex
  // is lightly co-weighted to the arm: it stretches its 0.01-long edges by
  // > 1.5x while moving < 3% of model height.
  if (wave === 'strain') {
    verts.push([0, 0.5, 0.2], [0.01, 0.5, 0.2], [0, 0.51, 0.2]);
    indicesArr.push(32, 33, 34);
  }
  // 'doubledLimb' adds a detached second "paw" box (verts 32-39) hugging the
  // underside of the left arm's hand end. Its DOMINANT joint is L_Arm (the
  // upper-arm joint — so V3 exempts it as waving-arm geometry), but the
  // static co-weights make it lag ~2/3 behind when the arm raises: a doubled
  // limb that sticks out horizontally while the real arm points up.
  if (wave === 'doubledLimb') {
    for (const v of boxVerts(0.62, 1.47, 0, 0.07, 0.05, 0.05)) verts.push(v);
    for (const t of BOX_TRIS) indicesArr.push(...t.map(i => i + 32));
  }
  // V9 fixture: a UV-seam duplicate pair — two vertices at the SAME bind
  // position on the upper arm, one bound to the arm, one to the spine.
  let seamPairBase = -1;
  if (seamPair) {
    seamPairBase = verts.length;
    verts.push([0.6, 1.5, 0.05], [0.6, 1.5, 0.05]);
  }
  // V8 fixture: a tiny triangle at the arm/torso boundary; its 0.005-long
  // edge (arm-bound vert to spine-bound vert) tears ~10x when the arm
  // raises 60 degrees.
  let tearBase = -1;
  if (boundaryTear) {
    tearBase = verts.length;
    verts.push([0.40, 1.55, 0], [0.40, 1.545, 0], [0.40, 1.54, 0.01]);
    indicesArr.push(tearBase, tearBase + 1, tearBase + 2);
  }
  if (hugeTriangles) {
    // Degenerate but structurally valid extra triangles to blow the budget.
    const extra = THRESHOLDS.MAX_TRIANGLES; // 48 real + 20,000 extra > 20,000
    for (let i = 0; i < extra; i++) indicesArr.push(0, 1, 2);
  }
  const positions = new Float32Array(verts.flat());
  const indices = new Uint16Array(indicesArr);
  const texcoord = new Float32Array(verts.length * 2); // all zeros

  // Skin weights. skin.joints order:
  //   0 Root, 1 Hips, 2 Spine, 3 L_Foot, 4 R_Foot, 5 Head,
  //   6 L_Arm, 7 L_Hand, 8 R_Arm, 9 R_Hand.
  const joints0 = new Uint8Array(verts.length * 4);
  const weights0 = new Float32Array(verts.length * 4);
  const setW = (v, pairs) => {
    pairs.forEach(([j, w], c) => { joints0[v * 4 + c] = j; weights0[v * 4 + c] = w; });
  };
  let codeformDone = false;
  for (let v = 0; v < 8; v++) { // torso
    const y = verts[v][1];
    if (wave === 'codeform' && y > 1.0 && !codeformDone) {
      // A3: exactly ONE torso vertex co-weighted to the LEFT ARM — dominant
      // joint stays the torso (Spine 0.55), so the vertex is IN the body
      // set, and it is a SMALL fraction of body vertices (a p95 hides it).
      setW(v, [[2, 0.55], [6, 0.45]]);
      codeformDone = true;
    } else {
      setW(v, [[y > 1.0 ? 2 : 1, 1]]);
    }
  }
  for (let v = 8; v < 16; v++) { // head
    if (wave === 'headLeak') setW(v, [[5, 0.6], [6, 0.4]]); // A2: 40% on the arm
    else setW(v, [[5, 1]]);
  }
  for (let v = 16; v < 24; v++) setW(v, [[6, 1]]);  // left arm -> L_Arm
  for (let v = 24; v < 32; v++) setW(v, [[8, 1]]);  // right arm -> R_Arm
  if (wave === 'strain') {
    setW(32, [[1, 1]]);
    setW(33, [[1, 1]]);
    setW(34, [[1, 0.97], [7, 0.03]]); // apex: 3% on L_Hand — moves ~1.8% h
  }
  if (wave === 'doubledLimb') {
    // dominant = L_Arm (6), statics Hips (1) + Spine (2) sum to 0.66:
    for (let v = 32; v < 40; v++) setW(v, [[6, 0.34], [1, 0.33], [2, 0.33]]);
  }
  // --- fix round 3 fixture mutations (see the doc comment above) ---------
  if (rebindPawHips) for (const v of [20, 21, 22, 23]) setW(v, [[1, 1]]);   // paw chunk -> Hips
  if (rebindTorsoArm) for (const v of [0, 1, 4, 5]) setW(v, [[6, 1]]);      // lower torso -> L_Arm
  if (blendMisbind) for (const v of [20, 21, 22, 23]) setW(v, [[7, 0.55], [2, 0.45]]); // paw: 0.55 L_Hand / 0.45 Spine
  if (cheekShoulder) for (const v of [10, 11, 14, 15]) setW(v, [[8, 0.6], [5, 0.4]]);  // head top -> R_Arm-dominated
  if (seamPair) { setW(seamPairBase, [[6, 1]]); setW(seamPairBase + 1, [[2, 1]]); }
  if (boundaryTear) { setW(tearBase, [[6, 1]]); setW(tearBase + 1, [[2, 1]]); setW(tearBase + 2, [[2, 1]]); }

  const posMin = [Infinity, Infinity, Infinity], posMax = [-Infinity, -Infinity, -Infinity];
  for (const [x, y, z] of verts) {
    posMin[0] = Math.min(posMin[0], x); posMin[1] = Math.min(posMin[1], y); posMin[2] = Math.min(posMin[2], z);
    posMax[0] = Math.max(posMax[0], x); posMax[1] = Math.max(posMax[1], y); posMax[2] = Math.max(posMax[2], z);
  }

  const accPos = addAccessor(positions, 5126, 'VEC3', { min: posMin, max: posMax });
  const accUV = addAccessor(texcoord, 5126, 'VEC2');
  const accIdx = addAccessor(indices, 5123, 'SCALAR');
  let accJoints = null, accWeights = null;
  if (skin) {
    accJoints = addAccessor(joints0, 5121, 'VEC4');
    accWeights = addAccessor(weights0, 5126, 'VEC4');
  }

  // --- skeleton ---------------------------------------------------------
  // node 0: mesh node; nodes 1..10: joints.
  const jointWorld = {
    Root: [0, 0, 0], Hips: [0, 1, 0], Spine: [0, 1.4, 0],
    L_Foot: [0.1, 0.1, 0], R_Foot: [-0.1, 0.1, 0], Head: [0, 1.9, 0],
    L_Arm: [0.35, 1.55, 0], L_Hand: [0.65, 1.55, 0],
    R_Arm: [-0.35, 1.55, 0], R_Hand: [-0.65, 1.55, 0],
  };
  const nodes = [
    { name: 'KittyMesh', mesh: 0, ...(skin ? { skin: 0 } : {}) },
    { name: 'Root', translation: [0, 0, 0], children: [2] },
    { name: 'Hips', translation: [0, 1, 0], children: [3, 4, 5] },
    { name: 'Spine', translation: [0, 0.4, 0], children: [6, 7, 9] },
    { name: 'L_Foot', translation: [0.1, -0.9, 0] },
    { name: 'R_Foot', translation: [-0.1, -0.9, 0] },
    { name: 'Head', translation: [0, 0.5, 0] },
    { name: 'L_Arm', translation: [0.35, 0.15, 0], children: [8] },
    { name: 'L_Hand', translation: [0.3, 0, 0] },
    { name: 'R_Arm', translation: [-0.35, 0.15, 0], children: [10] },
    { name: 'R_Hand', translation: [-0.3, 0, 0] },
  ];
  // A1 rigs bind the arms UP (a "cheer") while the clip starts arms-down.
  if (wave === 'limp') nodes[7].rotation = rotZ(110);        // L_Arm bind: up
  if (wave === 'leftCheerRight') nodes[9].rotation = rotZ(-110); // R_Arm bind: up (static in-clip)

  const jointNodeOrder = ['Root', 'Hips', 'Spine', 'L_Foot', 'R_Foot', 'Head', 'L_Arm', 'L_Hand', 'R_Arm', 'R_Hand'];
  const skinDef = skin ? (() => {
    const ibm = new Float32Array(jointNodeOrder.flatMap(n => invTranslation(jointWorld[n])));
    const accIBM = addAccessor(ibm, 5126, 'MAT4');
    return [{ name: 'KittyRig', joints: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], skeleton: 1, inverseBindMatrices: accIBM }];
  })() : undefined;

  // --- material / texture ------------------------------------------------
  const images = [], textures = [], samplers = [];
  const material = { name: 'KittyFur', pbrMetallicRoughness: { baseColorFactor: [1, 0.7, 0.2, 1] } };
  if (texture) {
    const png = texture === 'zeroed' ? new Uint8Array(PNG_1x1.length) : PNG_1x1; // A5
    const bvPng = addBV(png);
    images.push({ mimeType: 'image/png', bufferView: bvPng });
    samplers.push({ magFilter: 9729, minFilter: 9729 });
    textures.push({ source: 0, sampler: 0 });
    material.pbrMetallicRoughness.baseColorTexture = { index: 0 };
  }

  // --- animations --------------------------------------------------------
  const animations = [];
  const addClip = (name, channels) => {
    // channels: [{ node, path, times: number[], values: number[][] }]
    const anim = { name, samplers: [], channels: [] };
    for (const ch of channels) {
      const accIn = addAccessor(new Float32Array(ch.times), 5126, 'SCALAR',
        { min: [ch.times[0]], max: [ch.times[ch.times.length - 1]] });
      const accOut = addAccessor(new Float32Array(ch.values.flat()), 5126,
        ch.path === 'rotation' ? 'VEC4' : 'VEC3');
      anim.samplers.push({ input: accIn, output: accOut, interpolation: 'LINEAR' });
      anim.channels.push({
        sampler: anim.samplers.length - 1,
        target: { node: ch.node, path: ch.path },
      });
    }
    animations.push(anim);
  };

  const NODE = { Root: 1, Hips: 2, Spine: 3, L_Foot: 4, R_Foot: 5, Head: 6, L_Arm: 7, L_Hand: 8, R_Arm: 9, R_Hand: 10 };
  const trivialClip = name => addClip(name, [{
    node: NODE.Spine, path: 'rotation',
    times: [0, 0.5, 1], values: [rotX(0), rotX(2), rotX(0)],
  }]);

  const armProfile = (node, degs, dur = 1.5) => ({
    node, path: 'rotation',
    times: degs.map((_, i) => (i / (degs.length - 1)) * dur),
    values: degs.map(rotZ),
  });

  const waveChannels = () => {
    const GOOD = [0, 60, 15, 60, 15, 60, 0];
    const ch = [];
    switch (wave) {
      case 'good':
      case 'leftCheerRight': // clip-side identical; only the R bind differs
      case 'codeform':       // geometry-side failure; the clip is a good wave
      case 'headLeak':       // ditto
      case 'strain':         // ditto
      case 'doubledLimb':    // ditto — the phantom paw is pure geometry
        ch.push(armProfile(NODE.L_Arm, GOOD));
        break;
      case 'goodRight':
        ch.push(armProfile(NODE.R_Arm, GOOD.map(d => -d)));
        break;
      case 'limp':
        // A1: in-clip the arm sits at the hip (0° = straight out from the
        // "down" start) and only wiggles ±9° — enormous vs the cheer BIND,
        // tiny vs the clip's first frame.
        ch.push(armProfile(NODE.L_Arm, [0, 9, 0, -9, 0, 9, 0, -9, 0]));
        break;
      case 'small':
        ch.push(armProfile(NODE.L_Arm, [0, 10, 2, 10, 2, 10, 0]));
        break;
      case 'oneOsc':
        ch.push(armProfile(NODE.L_Arm, [0, 60, 0]));
        break;
      case 'bothArms':
        ch.push(armProfile(NODE.L_Arm, GOOD));
        ch.push(armProfile(NODE.R_Arm, [0, -50, -12, -50, -12, -50, 0]));
        break;
      case 'lurch':
        ch.push(armProfile(NODE.L_Arm, GOOD));
        ch.push({
          node: NODE.Hips, path: 'translation',
          times: [0, 0.75, 1.5],
          values: [[0, 1, 0], [0.3, 1, 0], [0, 1, 0]], // 0.3 units ~ 16% of height
        });
        break;
      case 'spineRotate':
        ch.push(armProfile(NODE.L_Arm, GOOD));
        ch.push({
          node: NODE.Spine, path: 'rotation',
          times: [0, 0.75, 1.5], values: [rotX(0), rotX(40), rotX(0)],
        });
        break;
      default:
        throw new Error(`unknown wave variant ${wave}`);
    }
    return ch;
  };

  for (const clip of clips) {
    const name = clip === 'walk' ? walkName : clip;
    if (clip === 'wave') {
      const wch = waveChannels();
      if (armScale) {
        // V7' scale sanity: the upper-arm joint animated at a non-unit scale.
        wch.push({
          node: NODE.L_Arm, path: 'scale',
          times: [0, 1.5], values: [[armScale, armScale, armScale], [armScale, armScale, armScale]],
        });
      }
      addClip('wave', wch);
    } else if (clip === 'jump' && !jumpGrounded) {
      // A real jump: the hips (and with them the feet) leave the ground.
      const jumpCh = [{
        node: NODE.Hips, path: 'translation',
        times: [0, 0.4, 0.8],
        values: [[0, 1, 0], [0, 1.35, 0], [0, 1, 0]], // +0.35 ~ 19% of height
      }];
      if (articulated) {
        // ... and the correctly skinned arms tuck in and back out.
        jumpCh.push(armProfile(NODE.L_Arm, [0, 40, 0], 0.8));
        jumpCh.push(armProfile(NODE.R_Arm, [0, -40, 0], 0.8));
      }
      addClip('jump', jumpCh);
    } else if (clip === 'walk' && walkDrift) {
      // V5: the hips translate away and the clip ends off its start value.
      addClip(name, [{
        node: NODE.Hips, path: 'translation',
        times: [0, 1], values: [[0, 1, 0], [0.35, 1, 0]],
      }]);
    } else if (clip === 'walk' && articulated) {
      // Correctly skinned limb articulation: both arms swing opposite ways
      // and the loop closes where it starts (V5).
      addClip(name, [
        armProfile(NODE.L_Arm, [0, 25, 0, -25, 0], 1),
        armProfile(NODE.R_Arm, [0, -25, 0, 25, 0], 1),
      ]);
    } else {
      trivialClip(name); // includes jumpGrounded: feet never rise
    }
  }

  // --- assemble ----------------------------------------------------------
  const attributes = { POSITION: accPos, TEXCOORD_0: accUV };
  if (skin) { attributes.JOINTS_0 = accJoints; attributes.WEIGHTS_0 = accWeights; }
  let extensionsUsed;
  if (instancing > 0) {
    // A4: the node renders its mesh `instancing` times.
    const accInst = addAccessor(new Float32Array(instancing * 3), 5126, 'VEC3');
    nodes[0].extensions = { EXT_mesh_gpu_instancing: { attributes: { TRANSLATION: accInst } } };
    extensionsUsed = ['EXT_mesh_gpu_instancing'];
  }
  const binBytes = bin.bytes();
  const json = {
    asset: { version: '2.0', generator: 'test-validate-kitty synthetic fixture' },
    scene: 0,
    scenes: [{ nodes: [0, 1] }],
    nodes,
    meshes: [{ name: 'KittyBody', primitives: [{ attributes, indices: accIdx, material: 0, mode: 4 }] }],
    materials: [material],
    ...(skinDef ? { skins: skinDef } : {}),
    ...(animations.length ? { animations } : {}),
    ...(images.length ? { images, textures, samplers } : {}),
    ...(extensionsUsed ? { extensionsUsed } : {}),
    accessors,
    bufferViews,
    buffers: [{ byteLength: binBytes.length }],
  };
  return buildGLB(json, binBytes);
}

/* ========================================================================== *
 * Helpers for assertions.
 * ========================================================================== */

const check = (report, id) => report.checks.find(c => c.id === id);
const subChecks = report => check(report, 'waveArm')?.details?.subChecks || {};

/* ========================================================================== *
 * Tests — the original coverage.
 * ========================================================================== */

test('correct single-GLB fixture PASSES all checks', () => {
  const report = validate({ files: { single: makeKitty() } });
  for (const c of report.checks) assert.equal(c.pass, true, `${c.id} should pass: ${c.measured}`);
  assert.equal(report.ok, true);
  assert.equal(report.layout, 'single');
  const sc = subChecks(report);
  assert.equal(check(report, 'waveArm').details.wavedArm, 'left');
  assert.ok(sc.armRotation.peakStartDeg >= 45 && sc.armRotation.peakStartDeg <= 90,
    `peak ${sc.armRotation.peakStartDeg}`);
  assert.ok(sc.handAboveShoulder.marginFrac >= 0.05, `hand margin ${sc.handAboveShoulder.marginFrac}`);
  assert.ok(sc.oscillations.oscillations >= 2, `oscillations ${sc.oscillations.oscillations}`);
  assert.ok(sc.otherArmStill.maxDeg <= 1, `other arm ${sc.otherArmStill.maxDeg}`);
  assert.ok(sc.rootTranslation.rangeFrac <= 0.001, `root moved ${sc.rootTranslation.rangeFrac}`);
  assert.equal(sc.bodyCoDeformation.over3Count, 0, sc.bodyCoDeformation.measured);
  assert.ok(sc.edgeStrain.maxRatio <= 1.05, sc.edgeStrain.measured);
});

test('`run` is accepted in place of `walk`', () => {
  const report = validate({ files: { single: makeKitty({ walkName: 'run' }) } });
  assert.equal(check(report, 'clips').pass, true, check(report, 'clips').measured);
  assert.equal(check(report, 'loopSeam').pass, true, check(report, 'loopSeam').measured);
  assert.equal(report.ok, true);
});

test('clip names are matched EXACTLY (a `Wave` clip does not satisfy `wave`)', () => {
  const glb = makeKitty({ clips: ['idle', 'jump', 'walk'] });
  const { json } = parseGLB(glb);
  assert.ok(!json.animations.some(a => a.name === 'wave'));
  const report = validate({ files: { single: glb } });
  assert.equal(check(report, 'clips').pass, false);
  assert.match(check(report, 'clips').measured, /wave/);
});

test('FAILS: no skin', () => {
  const report = validate({ files: { single: makeKitty({ skin: false }) } });
  assert.equal(check(report, 'skin').pass, false, check(report, 'skin').measured);
  assert.equal(report.ok, false);
});

test('FAILS: a missing clip (no `jump`)', () => {
  const report = validate({ files: { single: makeKitty({ clips: ['idle', 'walk', 'wave'] }) } });
  const c = check(report, 'clips');
  assert.equal(c.pass, false);
  assert.match(c.measured, /jump/);
  assert.equal(check(report, 'jump').pass, false, 'V6 also reports the missing clip');
  assert.equal(report.ok, false);
});

test(`FAILS: more than ${THRESHOLDS.MAX_TRIANGLES} rendered triangles`, () => {
  const report = validate({ files: { single: makeKitty({ hugeTriangles: true }) } });
  const c = check(report, 'triangles');
  assert.equal(c.pass, false);
  assert.ok(c.details.worst > THRESHOLDS.MAX_TRIANGLES, c.measured);
  assert.equal(report.ok, false);
});

test('FAILS: no base-color texture', () => {
  const report = validate({ files: { single: makeKitty({ texture: false }) } });
  assert.equal(check(report, 'baseColorTexture').pass, false);
  assert.equal(report.ok, false);
});

test('FAILS wave V1: the arm barely moves (10 deg < 45 deg)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'small' }) } });
  const sc = subChecks(report);
  assert.equal(sc.armRotation.pass, false, sc.armRotation.measured);
  assert.ok(sc.armRotation.peakStartDeg < 15, `peak ${sc.armRotation.peakStartDeg}`);
  assert.equal(check(report, 'waveArm').pass, false);
  assert.equal(report.ok, false);
});

test('FAILS wave V1: only one oscillation', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'oneOsc' }) } });
  const sc = subChecks(report);
  assert.equal(sc.armRotation.pass, true, 'the raise itself is big enough');
  assert.equal(sc.handAboveShoulder.pass, true, 'the hand does get up there');
  assert.equal(sc.oscillations.pass, false, sc.oscillations.measured);
  assert.ok(sc.oscillations.oscillations < 2);
  assert.equal(report.ok, false);
});

test('FAILS wave V2: hips translate (whole-body lurch)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'lurch' }) } });
  const sc = subChecks(report);
  assert.equal(sc.rootTranslation.pass, false, sc.rootTranslation.measured);
  assert.ok(sc.rootTranslation.rangeFrac > 0.05, `frac ${sc.rootTranslation.rangeFrac}`);
  assert.equal(report.ok, false);
});

test('FAILS wave V2: the spine rotates with the wave', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'spineRotate' }) } });
  const sc = subChecks(report);
  assert.equal(sc.torsoRotation.pass, false, sc.torsoRotation.measured);
  assert.ok(sc.torsoRotation.maxDeg > 15, `deg ${sc.torsoRotation.maxDeg}`);
  assert.equal(report.ok, false);
});

test('FAILS wave V3: torso vertices co-weighted to the arm (co-deformation)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'codeform' }) } });
  const sc = subChecks(report);
  // the clip itself is a clean wave — only the skinning is fused:
  assert.equal(sc.armRotation.pass, true, sc.armRotation.measured);
  assert.equal(sc.oscillations.pass, true, sc.oscillations.measured);
  assert.equal(sc.rootTranslation.pass, true, sc.rootTranslation.measured);
  assert.equal(sc.torsoRotation.pass, true, sc.torsoRotation.measured);
  assert.equal(sc.bodyCoDeformation.pass, false, sc.bodyCoDeformation.measured);
  assert.ok(sc.bodyCoDeformation.over3Frac > THRESHOLDS.BODY_DISPLACEMENT_MAX_OFFENDER_RATIO,
    `over-3%h fraction ${sc.bodyCoDeformation.over3Frac}`);
  assert.equal(report.ok, false);
});

test('four-file layout PASSES (kitty-idle/jump/walk/wave.glb)', () => {
  const files = {
    idle: makeKitty({ clips: ['idle'] }),
    jump: makeKitty({ clips: ['jump'] }),
    walk: makeKitty({ clips: ['walk'] }),
    wave: makeKitty({ clips: ['wave'] }),
  };
  const report = validate({ files });
  assert.equal(report.layout, 'multi');
  for (const c of report.checks) assert.equal(c.pass, true, `${c.id}: ${c.measured}`);
  assert.equal(report.ok, true);
});

test('four-file layout accepts kitty-run.glb for the walk slot', () => {
  const files = {
    idle: makeKitty({ clips: ['idle'] }),
    jump: makeKitty({ clips: ['jump'] }),
    run: makeKitty({ clips: ['walk'], walkName: 'run' }),
    wave: makeKitty({ clips: ['wave'] }),
  };
  const report = validate({ files });
  assert.equal(report.ok, true, JSON.stringify(report.checks.map(c => [c.id, c.pass, c.measured])));
});

test('missing asset: clear error, non-ok report', () => {
  const report = validate({ path: '/nonexistent/kitty-dir' });
  assert.equal(report.ok, false);
  assert.equal(report.layout, 'missing');
  assert.match(report.error, /no kitty asset found/);
});

test('joint-name tokenizer handles the documented rig-name shapes', () => {
  assert.deepEqual(tokenizeName('mixamorig:LeftArm'), ['mixamorig', 'left', 'arm']);
  assert.deepEqual(tokenizeName('L_hand'), ['l', 'hand']);
  assert.deepEqual(tokenizeName('hand.R.001'), ['hand', 'r', '001']);
  assert.deepEqual(tokenizeName('UpperArm_L'), ['upper', 'arm', 'l']);
  assert.deepEqual(tokenizeName('Spine02'), ['spine', '02']);
  // "Armature" must NOT tokenize to an `arm` match:
  assert.ok(!tokenizeName('Armature').includes('arm'));
});

/* ========================================================================== *
 * Tests — the round-1 adversarial mutations (A1–A6) and the new checks.
 * ========================================================================== */

test('A1 FAILS: limp wave — arms-up bind pose, arm wiggles ±9° at the hip', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'limp' }) } });
  const sc = subChecks(report);
  // Measured vs the CLIP START (not the cheer bind): tiny.
  assert.equal(sc.armRotation.pass, false, sc.armRotation.measured);
  assert.ok(sc.armRotation.peakStartDeg < 20, `peak from clip start ${sc.armRotation.peakStartDeg}`);
  // And the hand never rises above the shoulder.
  assert.equal(sc.handAboveShoulder.pass, false, sc.handAboveShoulder.measured);
  assert.equal(check(report, 'waveArm').pass, false);
  assert.equal(report.ok, false);
});

test('A1 counterpart PASSES: left wave while the static right arm has a "cheer" bind pose (side selection by in-clip motion)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'leftCheerRight' }) } });
  assert.equal(report.ok, true, JSON.stringify(report.checks.filter(c => !c.pass).map(c => [c.id, c.measured])));
  assert.equal(check(report, 'waveArm').details.wavedArm, 'left',
    'the waving side must be chosen by in-clip motion, not by distance from the bind pose');
  const sc = subChecks(report);
  assert.equal(sc.otherArmStill.pass, true, sc.otherArmStill.measured);
});

test('right-arm wave PASSES and the side is reported as right', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'goodRight' }) } });
  assert.equal(report.ok, true, JSON.stringify(report.checks.filter(c => !c.pass).map(c => [c.id, c.measured])));
  assert.equal(check(report, 'waveArm').details.wavedArm, 'right');
});

test('A2 FAILS: head vertices 40% weighted to the arm (V4 weight leakage + V3 body set includes the head)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'headLeak' }) } });
  const leak = check(report, 'weightLeakage');
  assert.equal(leak.pass, false, leak.measured);
  assert.ok(leak.details.rows[0].violations >= 8, `violations ${leak.details.rows[0].violations}`);
  // The dragged face is ALSO visible to V3 now (head-dominated vertices are
  // body vertices, not exempt):
  const sc = subChecks(report);
  assert.equal(sc.bodyCoDeformation.pass, false, sc.bodyCoDeformation.measured);
  assert.equal(report.ok, false);
});

test('A3 FAILS: a SMALL fraction of torso vertices co-weighted to the arm (a p95 would hide it)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'codeform' }) } });
  const sc = subChecks(report);
  assert.equal(sc.bodyCoDeformation.pass, false, sc.bodyCoDeformation.measured);
  // The offender pocket is under 5% of body vertices — exactly the pocket a
  // 95th percentile ignores — but over the 0.5% budget:
  assert.ok(sc.bodyCoDeformation.over3Frac < 0.05, `offenders ${sc.bodyCoDeformation.over3Frac}`);
  assert.ok(sc.bodyCoDeformation.over3Frac > THRESHOLDS.BODY_DISPLACEMENT_MAX_OFFENDER_RATIO);
  assert.equal(report.ok, false);
});

test('A4 FAILS: EXT_mesh_gpu_instancing multiplies rendered triangles past the budget', () => {
  const report = validate({ files: { single: makeKitty({ instancing: 500 }) } });
  const c = check(report, 'triangles');
  assert.equal(c.pass, false, c.measured);
  assert.equal(c.details.worst, 48 * 500, `worst ${c.details.worst}`);
  assert.equal(report.ok, false);
  // and a small instance count within budget still passes:
  const ok = validate({ files: { single: makeKitty({ instancing: 3 }) } });
  assert.equal(check(ok, 'triangles').pass, true, check(ok, 'triangles').measured);
  assert.equal(check(ok, 'triangles').details.worst, 48 * 3);
});

test('A5 FAILS: base-color image bytes zeroed (magic/header check)', () => {
  const report = validate({ files: { single: makeKitty({ texture: 'zeroed' }) } });
  const c = check(report, 'baseColorTexture');
  assert.equal(c.pass, false, c.measured);
  assert.match(c.measured, /NOT a valid PNG\/JPEG/);
  assert.equal(report.ok, false);
  // sniffImage itself: real PNG parses, zeroed bytes do not.
  assert.deepEqual(sniffImage(PNG_1x1), { format: 'PNG', width: 1, height: 1 });
  assert.equal(sniffImage(new Uint8Array(PNG_1x1.length)), null);
});

test('A6 FAILS: hips-translation lurch hides under a never-animated Root parent', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'lurch' }) } });
  const sc = subChecks(report);
  assert.equal(sc.rootTranslation.pass, false, sc.rootTranslation.measured);
  const byJoint = Object.fromEntries(sc.rootTranslation.candidates.map(c => [c.joint, c]));
  // The old validator checked only `Root` (0.00%) and passed; every root
  // candidate must be checked:
  assert.ok(byJoint.Root, 'Root is a candidate (top joint + root-named)');
  assert.ok(byJoint.Root.rangeFrac < 0.001, `Root moved ${byJoint.Root.rangeFrac}`);
  assert.ok(byJoint.Hips, 'Hips is a candidate (root-named + topmost animated translation)');
  assert.ok(byJoint.Hips.rangeFrac > 0.05, `Hips moved ${byJoint.Hips.rangeFrac}`);
  assert.equal(report.ok, false);
});

test('FAILS wave V1: the other arm moves too', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'bothArms' }) } });
  const sc = subChecks(report);
  assert.equal(check(report, 'waveArm').details.wavedArm, 'left', 'the bigger mover is the waving arm');
  assert.equal(sc.otherArmStill.pass, false, sc.otherArmStill.measured);
  assert.ok(sc.otherArmStill.maxDeg > THRESHOLDS.WAVE_MAX_OTHER_ARM_DEG, `other arm ${sc.otherArmStill.maxDeg}`);
  assert.equal(report.ok, false);
});

test('FAILS V5: walk with root drift does not close its loop seam', () => {
  const report = validate({ files: { single: makeKitty({ walkDrift: true }) } });
  const c = check(report, 'loopSeam');
  assert.equal(c.pass, false, c.measured);
  const walkRow = c.details.rows.find(r => r.clip === 'walk');
  assert.ok(walkRow && !walkRow.ok, JSON.stringify(c.details.rows));
  assert.ok(walkRow.violations.some(v => v.path === 'translation'), JSON.stringify(walkRow.violations));
  const idleRow = c.details.rows.find(r => r.clip === 'idle');
  assert.ok(idleRow && idleRow.ok, 'idle still closes its loop');
  assert.equal(report.ok, false);
});

test('FAILS V6: a jump that never leaves the ground', () => {
  const report = validate({ files: { single: makeKitty({ jumpGrounded: true }) } });
  const c = check(report, 'jump');
  assert.equal(c.pass, false, c.measured);
  assert.ok(c.details.riseFrac < THRESHOLDS.JUMP_MIN_FOOT_RISE_FRAC, `rise ${c.details.riseFrac}`);
  assert.equal(report.ok, false);
});

test('FAILS wave V3: body-triangle stretch (edge strain) even when every displacement stays under 3% h', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'strain' }) } });
  const sc = subChecks(report);
  assert.equal(sc.bodyCoDeformation.pass, true, sc.bodyCoDeformation.measured);
  assert.equal(sc.bodyCoDeformation.over3Count, 0, sc.bodyCoDeformation.measured);
  assert.equal(sc.edgeStrain.pass, false, sc.edgeStrain.measured);
  assert.ok(sc.edgeStrain.maxRatio > THRESHOLDS.EDGE_SOFT_RATIO_BODY, `ratio ${sc.edgeStrain.maxRatio}`);
  assert.equal(report.ok, false);
});

/* ========================================================================== *
 * Tests — V7 limb integrity (fix round 2).
 * ========================================================================== */

test('V7 FAILS: doubled limb — a detached second paw dominated by the upper-arm joint sticks out when the arm raises, and every OTHER check passes', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'doubledLimb' }) } });
  // The defect the older checks miss: everything else is green...
  for (const c of report.checks) {
    if (c.id === 'limbIntegrity') continue;
    assert.equal(c.pass, true, `${c.id} should pass: ${c.measured}`);
  }
  const sc = subChecks(report);
  assert.equal(sc.bodyCoDeformation.pass, true,
    'the phantom is dominated by the waving arm chain, so V3 exempts it: ' + sc.bodyCoDeformation.measured);
  assert.equal(sc.edgeStrain.pass, true, sc.edgeStrain.measured);
  // ...but V7 fails it, on the wave clip, blaming the arm joint that
  // dominates the phantom.
  const c = check(report, 'limbIntegrity');
  assert.equal(c.pass, false, c.measured);
  const rows = Object.fromEntries(c.details.rows.map(r => [r.clip, r]));
  assert.equal(rows.wave.ok, false, rows.wave.detail);
  assert.ok(rows.wave.worstRatio > THRESHOLDS.LIMB_OFFENDER_RATIO * 10,
    `offender ratio ${rows.wave.worstRatio} should dwarf the ${THRESHOLDS.LIMB_OFFENDER_RATIO} budget`);
  assert.equal(rows.wave.dominantJoint, 'L_Arm', rows.wave.detail);
  assert.ok(rows.idle.ok && rows.jump.ok && rows.walk.ok,
    'the phantom moves rigidly with the skeleton in the other clips: ' + c.measured);
  assert.equal(report.ok, false);
});

test('V7 PASSES: the correct fixture keeps every clip clean', () => {
  const report = validate({ files: { single: makeKitty() } });
  const c = check(report, 'limbIntegrity');
  assert.equal(c.pass, true, c.measured);
  assert.equal(c.details.rows.length, 4);
  for (const r of c.details.rows) {
    assert.equal(r.ok, true, `${r.clip}: ${r.detail}`);
    assert.equal(r.worstCount, 0, `${r.clip} should have zero offenders: ${r.detail}`);
  }
  assert.equal(report.ok, true);
});

test('V7 PASSES: correctly skinned limbs articulating in walk and jump', () => {
  const report = validate({ files: { single: makeKitty({ articulated: true }) } });
  const c = check(report, 'limbIntegrity');
  assert.equal(c.pass, true, c.measured);
  const rows = Object.fromEntries(c.details.rows.map(r => [r.clip, r]));
  assert.equal(rows.walk.worstCount, 0, `swinging arms ride their bones: ${rows.walk.detail}`);
  assert.equal(rows.jump.worstCount, 0, `tucking arms ride their bones: ${rows.jump.detail}`);
  // the whole fixture stays green (the articulation breaks no other check):
  assert.equal(report.ok, true,
    JSON.stringify(report.checks.filter(x => !x.pass).map(x => [x.id, x.measured])));
});

/* ========================================================================== *
 * Tests — fix round 3: V3' / V4' / V7' / V8 / V9 blind-spot fixtures.
 * Both round-3 probes proved V7 is invariant BY CONSTRUCTION for any vertex
 * bound 100% to one joint (skinned = delta_k * v_bind, so d_t = d_bind at
 * every frame) and that V3's dominant-joint body membership can be escaped
 * by re-pointing weights. Each fixture below reproduces one proven pass.
 * ========================================================================== */

test("V7' FAILS: a paw chunk rebound 100% to Hips rides rigidly — V7 stays invariant, V7' flags the binding at bind", () => {
  const report = validate({ files: { single: makeKitty({ rebindPawHips: true }) } });
  // The proven V7 blind spot: the rigid rebind is exactly invariant.
  assert.equal(check(report, 'limbIntegrity').pass, true,
    'V7 must NOT see a rigid rebind: ' + check(report, 'limbIntegrity').measured);
  const mb = check(report, 'misBinding');
  assert.equal(mb.pass, false, mb.measured);
  const row = mb.details.rows[0];
  assert.ok(row.soft >= 4, `soft offenders ${row.soft}`);
  assert.ok(row.softRatio > THRESHOLDS.MISBIND_OFFENDER_RATIO, `ratio ${row.softRatio}`);
  assert.equal(report.ok, false);
});

test("V3' FAILS: torso vertices rebound 100% to the waving upper arm — the spec's named disqualifier, caught by GEOMETRIC body membership", () => {
  const report = validate({ files: { single: makeKitty({ rebindTorsoArm: true }) } });
  // V7 is invariant (rigid rebind), and the old V3 exempted these vertices
  // because their CURRENT dominant joint was the waving arm:
  assert.equal(check(report, 'limbIntegrity').pass, true,
    'V7 must NOT see a rigid rebind: ' + check(report, 'limbIntegrity').measured);
  const sc = subChecks(report);
  assert.equal(sc.bodyCoDeformation.pass, false, sc.bodyCoDeformation.measured);
  assert.ok(sc.bodyCoDeformation.over3Frac > THRESHOLDS.BODY_DISPLACEMENT_MAX_OFFENDER_RATIO,
    `over-3%h fraction ${sc.bodyCoDeformation.over3Frac}`);
  assert.ok(sc.bodyCoDeformation.maxFrac > THRESHOLDS.BODY_DISPLACEMENT_HARD_FRAC,
    `hard cap must also trip: ${sc.bodyCoDeformation.maxFrac}`);
  assert.equal(report.ok, false);
});

test("V7' FAILS hard: a small blended fragment (0.55 hand / 0.45 spine) stays under the 0.2% soft ratio but is a HARD offender", () => {
  const report = validate({ files: { single: makeKitty({ blendMisbind: true }) } });
  const mb = check(report, 'misBinding');
  assert.equal(mb.pass, false, mb.measured);
  const row = mb.details.rows[0];
  assert.ok(row.softRatio <= THRESHOLDS.MISBIND_OFFENDER_RATIO,
    `the fragment's dominant bone is nearby, so the soft ratio must NOT trip: ${row.softRatio}`);
  assert.ok(row.hard >= 1, `hard offenders ${row.hard}: ${row.detail}`);
  assert.equal(report.ok, false);
});

test("V7' FAILS: the upper-arm joint animated at scale 1.6x — V7's rigid transport is exactly invariant to it", () => {
  const report = validate({ files: { single: makeKitty({ armScale: 1.6 }) } });
  assert.equal(check(report, 'limbIntegrity').pass, true,
    'V7 must NOT see the scale (proven blind spot): ' + check(report, 'limbIntegrity').measured);
  const mb = check(report, 'misBinding');
  assert.equal(mb.pass, false, mb.measured);
  assert.ok(mb.details.rows[0].scaleWorstDev > THRESHOLDS.JOINT_SCALE_TOL,
    `scale deviation ${mb.details.rows[0].scaleWorstDev}`);
  // V8 also sees the ballooned skin: arm edges stretch 1.6x vs BIND.
  assert.equal(check(report, 'edgeStrain').pass, false, check(report, 'edgeStrain').measured);
  assert.equal(report.ok, false);
});

test('V9 FAILS: UV-seam duplicate vertices with different weights crack apart in wave', () => {
  const report = validate({ files: { single: makeKitty({ seamPair: true }) } });
  const sm = check(report, 'seamCracks');
  assert.equal(sm.pass, false, sm.measured);
  const waveRow = sm.details.rows.find(r => r.clip === 'wave');
  assert.ok(waveRow && !waveRow.ok, JSON.stringify(sm.details.rows));
  assert.ok(waveRow.badGroups >= 1, waveRow.detail);
  assert.ok(waveRow.maxSepFrac > THRESHOLDS.SEAM_MAX_SEPARATION_FRAC, `separation ${waveRow.maxSepFrac}`);
  assert.equal(report.ok, false);
});

test("V4' FAILS: an arm-DOMINATED cheek region above the neck — V4 (head-dominated only) stays blind", () => {
  const report = validate({ files: { single: makeKitty({ cheekShoulder: true }) } });
  // V4 checks vertices whose DOMINANT joint is head/neck — these are
  // arm-dominated, so V4 must NOT see them (the proven blind spot):
  assert.equal(check(report, 'weightLeakage').pass, true, check(report, 'weightLeakage').measured);
  const rl = check(report, 'reverseLeakage');
  assert.equal(rl.pass, false, rl.measured);
  assert.ok(rl.details.rows[0].violations >= 4, rl.details.rows[0].detail);
  assert.equal(report.ok, false);
});

test('V8 FAILS: a boundary edge at the arm/torso seam tears ~10x — the old strain dropped every arm-touching triangle', () => {
  const report = validate({ files: { single: makeKitty({ boundaryTear: true }) } });
  const es = check(report, 'edgeStrain');
  assert.equal(es.pass, false, es.measured);
  const waveRow = es.details.rows.find(r => r.clip === 'wave');
  assert.ok(waveRow && !waveRow.ok, JSON.stringify(es.details.rows));
  assert.ok(waveRow.hardCount >= 1, waveRow.detail);
  assert.ok(waveRow.maxRatio >= 5, `tear ratio ${waveRow.maxRatio}`);
  // the wave sub-check mirrors the V8 wave verdict:
  assert.equal(subChecks(report).edgeStrain.pass, false);
  assert.equal(report.ok, false);
});

test("correct fixture PASSES every fix-round-3 check (V3'/V4'/V7'/V8/V9) in both layouts", () => {
  for (const files of [
    { single: makeKitty() },
    {
      idle: makeKitty({ clips: ['idle'] }),
      jump: makeKitty({ clips: ['jump'] }),
      walk: makeKitty({ clips: ['walk'] }),
      wave: makeKitty({ clips: ['wave'] }),
    },
  ]) {
    const report = validate({ files });
    for (const id of ['edgeStrain', 'seamCracks', 'reverseLeakage', 'misBinding']) {
      assert.equal(check(report, id).pass, true, `${id}: ${check(report, id).measured}`);
    }
    assert.equal(subChecks(report).bodyCoDeformation.pass, true);
    assert.equal(report.ok, true,
      JSON.stringify(report.checks.filter(c => !c.pass).map(c => [c.id, c.measured])));
  }
});
