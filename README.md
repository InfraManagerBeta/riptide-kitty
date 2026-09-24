# Riptide Kitty

A test project for the [Riptide Protocol](https://riptide-staging.vercel.app)
task market. Tasks post a spec from `specs/`; agents deliver rival
implementation PRs; the poster merges one.

## Tasks

- [`specs/001-kitty-animated.md`](specs/001-kitty-animated.md) — a rigged
  3D kitty (generated from the reference images) with four animations:
  idle, jump, walk/run, and a one-paw wave. The arms must be separate,
  articulated limbs so the wave reads cleanly.

## Reference

The character is defined by the four orthographic views in `reference/`
(front / back / left / right).

## Kitty asset: pipeline, validation, preview

### Generation pipeline (spec 001)

The kitty is **generated, not hand-modeled**, with Tripo3D. Only ONE
repository image went into Tripo: `reference/kitty-front.png`. Every other
view in the pipeline is **Tripo-derived**, not the repository's back/left/
right orthographic references:

1. **Multiview derivation** — Tripo3D `generate_multiview_image` regenerates
   a consistent 4-view turnaround from the single front reference.
2. **Arms-apart edit** — Tripo3D `edit_multiview_image` re-poses those
   derived views into an arms-up A-pose (so the arms rig as separate limbs,
   not fused to the torso). These edited, Tripo-derived views are saved
   under `assets/kitty/derived/`.
3. **Multiview-to-model** — Tripo3D `multiview_to_model` from the
   **Tripo-derived arms-apart views** (step 2), not from the four
   orthographic views in `reference/`.
4. **Auto-rig** — Tripo3D biped/humanoid rig.
5. **Animations** — preset retargets for `idle` / `jump` / `walk` (or `run`)
   plus an authored `wave` (one arm raised, at least two oscillations, body
   held stable).
6. **Single-GLB merge** — all four clips in `assets/kitty/kitty.glb`
   (spec-allowed fallback: four files `assets/kitty/kitty-<clip>.glb`).

The exact prompts, tool/model versions, and per-stage Tripo3D task IDs are
recorded in `assets/kitty/kitty.provenance.json` (committed alongside the
asset by the generation pipeline).

### Re-running validation

Requires Node >= 20; zero dependencies, no install step.

```sh
# validate the asset (assets/kitty/kitty.glb or the four-file layout);
# prints a PASS/FAIL table per check and exits non-zero on any failure
# (a MISSING asset is a failure — CI green means the asset passes):
node scripts/validate-kitty.mjs

# validate a specific file or directory:
node scripts/validate-kitty.mjs path/to/kitty.glb

# run the validator's self-test on synthetic in-memory GLB fixtures
# (includes one fixture per adversarial mutation from the round-1 review):
npm test
```

#### Validator checks and thresholds

Every threshold is a documented constant in `THRESHOLDS`, and every joint
name pattern in `JOINT_NAME_PATTERNS`, at the top of
`scripts/validate-kitty.mjs`. Model height *h* = Y extent of the POSITION
bounds. All animation measurements sample at 60 samples/s with correct
STEP/LINEAR/CUBICSPLINE interpolation.

| Check | Requirement | Threshold |
|---|---|---|
| a. skin | at least one skin | >= 1 |
| b. clips | `idle`, `jump`, `walk` (or `run`), `wave` by exact name | all present |
| c. triangles | rendered triangles: summed per node instance, all scenes, **multiplied by `EXT_mesh_gpu_instancing` instance counts**; strips/fans = count−2, points/lines = 0, morph targets add none | <= 20,000 |
| d. texture | a `baseColorTexture` on a material used by a skinned primitive whose image bytes are a **real PNG/JPEG** (magic bytes + parseable header, nonzero width×height — declared mimeType is ignored) | valid image |
| V1 arm rotation | some joint of the waving arm chain rotates vs the **wave clip's first frame** (never the bind pose); the side is chosen by in-clip motion | >= 45° |
| V1 raised hand | at the peak frame, the waving hand/wrist world height >= that arm's shoulder (or upper-arm) joint height + margin | +5% *h* |
| V1 oscillations | direction reversals of the dominant rotation component, with hysteresis (swings must span max(10°, 20% of range)) | >= 2 |
| V1 other arm | every joint of the non-waving arm chain stays put in-clip | <= 15° |
| V2 root translation | world-translation range of **every** root candidate: the top joint, every `root\|hips\|pelvis` joint, and the topmost joint with an animated translation channel | each <= 5% *h* |
| V2 torso rotation | max torso joint rotation vs the clip's first frame | <= 15° |
| V3 co-deformation | linear-blend-skinned displacement vs the clip's first frame, over **all** vertices not dominated by the waving arm chain (head, tail, legs, other arm included; shoulder-dominated skin counts as body) | <= 0.5% of them may move > 3% *h*, none > 8% *h* |
| V3 edge strain | max edge-length ratio vs the clip's first frame over triangles whose three vertices are all non-waving-arm | <= 1.5× |
| V4 weight leakage | vertices dominated by a head/neck joint carrying > 0.2 total arm-chain weight (either arm) | zero vertices |
| V5 loop seams | in `idle` and `walk`/`run`, every channel's end value equals its start value | <= 1% *h* translation, <= 2° rotation, <= 1% scale |
| V6 jump | at some frame, **every** foot joint (`foot\|toe\|ankle`) rises above its clip-start world height | >= 8% *h* |
| V7 limb integrity | in **every** clip: skinned geometry must ride the bones it wraps. Bone segments = parent→child world segments over the skin's joints (segments from the top joint and zero-length segments ignored). Per vertex, `d_bind` = bind-pose distance (raw `POSITION`, identity joint deltas) to the nearest bone segment **of its own weighted joints**; at each sampled frame, `d_t` = the same distance with each candidate bone carried rigidly by its joint's delta `G·IBM` — so geometry rigidly attached to any of its joints keeps `d_t = d_bind` exactly, which is what makes muzzles/ears/tail tips hanging off a **leaf joint** benign *in principle* (the joint's frame rotates with them; a bare segment cannot express that orientation), and no joint is ever exempted by name. Catches detached/doubled limbs (a second forearm lagging behind the real one) even when their dominant joint is the V3-exempt waving arm chain | per frame <= 0.2% of skinned vertices beyond `min(1.5·d_bind, d_bind + 3% h) + 3% h`; none beyond `min(2·d_bind, d_bind + 8% h) + 8% h`. The multiplicative slack (skin sliding scales with wrap radius) is **capped in absolute terms**: skin slide is bounded by tissue scale, and without the cap a limb-shaped mass parked 25% *h* from every bone would earn 12% *h* of free drift |

CI runs the same self-test and validator on every push and pull request
(`.github/workflows/validate-kitty.yml`). The validator (and therefore CI)
**fails when no asset is present** — a green check means the committed
asset passed every check above.

### Opening the viewer

The viewer is zero-build (three.js 0.160.1 from a pinned CDN). Serve the
**repository root** statically and open `/preview/`:

```sh
npx serve .
# or
python3 -m http.server
```

then browse to `http://localhost:3000/preview/` (serve) or
`http://localhost:8000/preview/` (http.server). Orbit with the mouse; the
buttons switch among the four clips (`idle` autoplays and loops; switches
crossfade). The page exposes `window.__kitty = { clips, play(name), current }`
for automated playtests. GitHub Pages serving the branch root works the same
way: `/preview/index.html` loads `/assets/kitty/…` by relative path.
