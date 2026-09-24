/**
 * test-validate-kitty.mjs — self-test for scripts/validate-kitty.mjs.
 *
 * Builds SYNTHETIC skinned GLBs entirely in memory (a tiny torso+arms mesh
 * with a 6-joint skeleton and an embedded 1x1 PNG base-color texture) and
 * asserts that the validator passes a correct fixture and fails each broken
 * fixture on its SPECIFIC check.
 *
 * Run:  node --test scripts/test-validate-kitty.mjs
 *   or: node scripts/test-validate-kitty.mjs
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { validate, THRESHOLDS, parseGLB, tokenizeName } from './validate-kitty.mjs';

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
 * Skeleton (Y-up, joint world positions):
 *   Hips (0,1,0) -> Spine (0,1.4,0) -> { L_Arm (0.35,1.55,0) -> L_Hand (0.65,1.55,0),
 *                                        R_Arm (-0.35,1.55,0) -> R_Hand (-0.65,1.55,0) }
 * Mesh: three boxes — torso (y 0.2..1.8, POSITION Y-extent => model height 1.6),
 * left arm, right arm. Torso verts weighted to Hips/Spine; arm verts to
 * L_Arm / R_Arm. Names deliberately exercise `L_` / `R_` prefix tokens.
 * ========================================================================== */

const MODEL_HEIGHT = 1.6;

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
 *   texture: include baseColorTexture (default true)
 *   hugeTriangles: exceed the triangle budget (default false)
 *   wave: 'good' | 'small' | 'oneOsc' | 'lurch' | 'spineRotate' | 'codeform'
 */
export function makeKitty(opts = {}) {
  const {
    clips = ['idle', 'jump', 'walk', 'wave'],
    walkName = 'walk',
    skin = true,
    texture = true,
    hugeTriangles = false,
    wave = 'good',
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
  const torso = boxVerts(0, 1.0, 0, 0.25, 0.8, 0.2);      // y 0.2 .. 1.8
  const armL = boxVerts(0.575, 1.55, 0, 0.225, 0.05, 0.05); // x 0.35 .. 0.8
  const armR = boxVerts(-0.575, 1.55, 0, 0.225, 0.05, 0.05);
  const verts = [...torso, ...armL, ...armR];
  const positions = new Float32Array(verts.flat());

  const indicesArr = [];
  for (let b = 0; b < 3; b++) for (const t of BOX_TRIS) indicesArr.push(...t.map(i => i + b * 8));
  if (hugeTriangles) {
    // Degenerate but structurally valid extra triangles to blow the budget.
    const extra = THRESHOLDS.MAX_TRIANGLES; // 36 real + 20,000 extra > 20,000
    for (let i = 0; i < extra; i++) indicesArr.push(0, 1, 2);
  }
  const indices = new Uint16Array(indicesArr);
  const texcoord = new Float32Array(verts.length * 2); // all zeros

  // Skin weights. Joint order in skin.joints: 0 Hips, 1 Spine, 2 L_Arm,
  // 3 L_Hand, 4 R_Arm, 5 R_Hand.
  const joints0 = new Uint8Array(verts.length * 4);
  const weights0 = new Float32Array(verts.length * 4);
  const setW = (v, pairs) => {
    pairs.forEach(([j, w], c) => { joints0[v * 4 + c] = j; weights0[v * 4 + c] = w; });
  };
  for (let v = 0; v < 8; v++) {
    const y = verts[v][1];
    if (wave === 'codeform' && y > 1.0) {
      // torso verts co-weighted to the LEFT ARM: dominant joint stays the
      // torso (Spine, 0.55) so these verts are IN the torso set — and they
      // get dragged by the waving arm. This is the fused-arm failure mode.
      setW(v, [[1, 0.55], [2, 0.45]]);
    } else {
      setW(v, [[y > 1.0 ? 1 : 0, 1]]);
    }
  }
  for (let v = 8; v < 16; v++) setW(v, [[2, 1]]);   // left arm -> L_Arm
  for (let v = 16; v < 24; v++) setW(v, [[4, 1]]);  // right arm -> R_Arm

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
  // node 0: mesh node; nodes 1..6: joints.
  const jointWorld = {
    Hips: [0, 1, 0], Spine: [0, 1.4, 0],
    L_Arm: [0.35, 1.55, 0], L_Hand: [0.65, 1.55, 0],
    R_Arm: [-0.35, 1.55, 0], R_Hand: [-0.65, 1.55, 0],
  };
  const nodes = [
    { name: 'KittyMesh', mesh: 0, ...(skin ? { skin: 0 } : {}) },
    { name: 'Hips', translation: [0, 1, 0], children: [2] },
    { name: 'Spine', translation: [0, 0.4, 0], children: [3, 5] },
    { name: 'L_Arm', translation: [0.35, 0.15, 0], children: [4] },
    { name: 'L_Hand', translation: [0.3, 0, 0] },
    { name: 'R_Arm', translation: [-0.35, 0.15, 0], children: [6] },
    { name: 'R_Hand', translation: [-0.3, 0, 0] },
  ];
  const jointNodeOrder = ['Hips', 'Spine', 'L_Arm', 'L_Hand', 'R_Arm', 'R_Hand'];
  const skinDef = skin ? (() => {
    const ibm = new Float32Array(jointNodeOrder.flatMap(n => invTranslation(jointWorld[n])));
    const accIBM = addAccessor(ibm, 5126, 'MAT4');
    return [{ name: 'KittyRig', joints: [1, 2, 3, 4, 5, 6], skeleton: 1, inverseBindMatrices: accIBM }];
  })() : undefined;

  // --- material / texture ------------------------------------------------
  const images = [], textures = [], samplers = [];
  const material = { name: 'KittyFur', pbrMetallicRoughness: { baseColorFactor: [1, 0.7, 0.2, 1] } };
  if (texture) {
    const bvPng = addBV(PNG_1x1);
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

  const NODE = { Hips: 1, Spine: 2, L_Arm: 3, L_Hand: 4, R_Arm: 5, R_Hand: 6 };
  const trivialClip = name => addClip(name, [{
    node: NODE.Spine, path: 'rotation',
    times: [0, 0.5, 1], values: [rotX(0), rotX(2), rotX(0)],
  }]);

  const waveChannels = () => {
    const ch = [];
    const armWave = (degs, dur = 1.5) => ({
      node: NODE.L_Arm, path: 'rotation',
      times: degs.map((_, i) => (i / (degs.length - 1)) * dur),
      values: degs.map(rotZ),
    });
    switch (wave) {
      case 'good':
      case 'codeform': // geometry-side failure; the clip itself is a good wave
        ch.push(armWave([0, 60, 15, 60, 15, 60, 0]));
        break;
      case 'small':
        ch.push(armWave([0, 10, 2, 10, 2, 10, 0]));
        break;
      case 'oneOsc':
        ch.push(armWave([0, 60, 0]));
        break;
      case 'lurch':
        ch.push(armWave([0, 60, 15, 60, 15, 60, 0]));
        ch.push({
          node: NODE.Hips, path: 'translation',
          times: [0, 0.75, 1.5],
          values: [[0, 1, 0], [0.3, 1, 0], [0, 1, 0]], // 0.3 units ~ 19% of height
        });
        break;
      case 'spineRotate':
        ch.push(armWave([0, 60, 15, 60, 15, 60, 0]));
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
    if (clip === 'wave') addClip('wave', waveChannels());
    else trivialClip(name);
  }

  // --- assemble ----------------------------------------------------------
  const attributes = { POSITION: accPos, TEXCOORD_0: accUV };
  if (skin) { attributes.JOINTS_0 = accJoints; attributes.WEIGHTS_0 = accWeights; }
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
 * Tests.
 * ========================================================================== */

test('correct single-GLB fixture PASSES all five checks', () => {
  const report = validate({ files: { single: makeKitty() } });
  for (const c of report.checks) assert.equal(c.pass, true, `${c.id} should pass: ${c.measured}`);
  assert.equal(report.ok, true);
  assert.equal(report.layout, 'single');
  const sc = subChecks(report);
  assert.equal(check(report, 'waveArm').details.wavedArm, 'left');
  assert.ok(sc.armRotation.peakRestDeg >= 45 && sc.armRotation.peakRestDeg <= 90,
    `peak ${sc.armRotation.peakRestDeg}`);
  assert.ok(sc.oscillations.oscillations >= 2, `oscillations ${sc.oscillations.oscillations}`);
  assert.ok(sc.rootTranslation.rangeFrac <= 0.001, `root moved ${sc.rootTranslation.rangeFrac}`);
  assert.ok(sc.torsoCoDeformation.p95Frac <= 0.001, `torso p95 ${sc.torsoCoDeformation.p95Frac}`);
});

test('`run` is accepted in place of `walk`', () => {
  const report = validate({ files: { single: makeKitty({ walkName: 'run' }) } });
  assert.equal(check(report, 'clips').pass, true, check(report, 'clips').measured);
  assert.equal(report.ok, true);
});

test('clip names are matched EXACTLY (a `Wave` clip does not satisfy `wave`)', () => {
  const glb = makeKitty({ clips: ['idle', 'jump', 'walk'] });
  // add a wrongly-cased clip by rebuilding with renamed animation
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

test('FAILS wave (i): the arm barely moves (10 deg < 45 deg)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'small' }) } });
  const sc = subChecks(report);
  assert.equal(sc.armRotation.pass, false, sc.armRotation.measured);
  assert.ok(sc.armRotation.peakRestDeg < 15, `peak ${sc.armRotation.peakRestDeg}`);
  assert.equal(check(report, 'waveArm').pass, false);
  assert.equal(report.ok, false);
});

test('FAILS wave (ii): only one oscillation', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'oneOsc' }) } });
  const sc = subChecks(report);
  assert.equal(sc.armRotation.pass, true, 'the raise itself is big enough');
  assert.equal(sc.oscillations.pass, false, sc.oscillations.measured);
  assert.ok(sc.oscillations.oscillations < 2);
  assert.equal(report.ok, false);
});

test('FAILS wave (iii): hips translate (whole-body lurch)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'lurch' }) } });
  const sc = subChecks(report);
  assert.equal(sc.rootTranslation.pass, false, sc.rootTranslation.measured);
  assert.ok(sc.rootTranslation.rangeFrac > 0.05, `frac ${sc.rootTranslation.rangeFrac}`);
  assert.equal(report.ok, false);
});

test('FAILS wave (iv): the spine rotates with the wave', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'spineRotate' }) } });
  const sc = subChecks(report);
  assert.equal(sc.torsoRotation.pass, false, sc.torsoRotation.measured);
  assert.ok(sc.torsoRotation.maxDeg > 15, `deg ${sc.torsoRotation.maxDeg}`);
  assert.equal(report.ok, false);
});

test('FAILS wave (v): torso vertices co-weighted to the arm (co-deformation)', () => {
  const report = validate({ files: { single: makeKitty({ wave: 'codeform' }) } });
  const sc = subChecks(report);
  // the clip itself is a clean wave — only the skinning is fused:
  assert.equal(sc.armRotation.pass, true, sc.armRotation.measured);
  assert.equal(sc.oscillations.pass, true, sc.oscillations.measured);
  assert.equal(sc.rootTranslation.pass, true, sc.rootTranslation.measured);
  assert.equal(sc.torsoRotation.pass, true, sc.torsoRotation.measured);
  assert.equal(sc.torsoCoDeformation.pass, false, sc.torsoCoDeformation.measured);
  assert.ok(sc.torsoCoDeformation.p95Frac > 0.03, `p95 frac ${sc.torsoCoDeformation.p95Frac}`);
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
