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
repository image went into Tripo: `reference/kitty-front.png`. The round-3
asset (`assets/kitty/kitty.glb`) came from a **single-view** generation —
multiview consistency failures were the root cause of the earlier fused /
doubled-lobe defects, so the mesh is now generated from one edited front
view that has no second view to disagree with:

1. **Front-view arms-apart edit** — Tripo3D `edit_multiview_image` re-poses
   the front reference into a clean, symmetric, shallow arms-apart pose
   (saved under `assets/kitty/derived/`). Round-1's derived turnaround was
   only re-used for reference; no new multiview generation.
2. **Mesh + texture** — Tripo3D `image_to_model` from that **single edited
   front view** (deliberately NOT `multiview_to_model`; see
   `generation_choice` in the provenance for the full rationale).
3. **Auto-rig** — Tripo3D biped rig (`animate_prerigcheck` + `animate_rig`).
4. **Scripted skin-weight post-processes** —
   `scripts/pipeline/clean_skin_weights.py` (zeroes stray arm-chain weight
   on non-arm-dominated vertices, then renormalizes) and
   `scripts/pipeline/fix_arm_shoulder_weights.py` (tightens the
   shoulder/clavicle-vs-limb split, run post-merge).
5. **Animations** — `idle` is the Tripo preset retarget used as-is; `walk`
   is the preset retarget made **in-place** by
   `scripts/pipeline/fix_walk_inplace.py` (hip drift removed so the cycle
   loops); `jump` is **authored** (`keyframe_jump.py` — the preset was a
   5-bounce loop that never cleared the required rise, so it was diagnosed
   and discarded); `wave` is **authored** (`keyframe_wave.py` — no Tripo
   preset exists; one arm, >= 2 oscillations, body held stable, sweeping in
   the model's frontal plane about the re-verified forward axis).
6. **Single-GLB merge** — all four clips in `assets/kitty/kitty.glb`
   (spec-allowed fallback: four files `assets/kitty/kitty-<clip>.glb`).

The exact prompts, tool/model versions, per-stage Tripo3D task IDs, the
per-clip provenance (preset vs authored), and the credit/cost accounting
are recorded in `assets/kitty/kitty.provenance.json` (committed alongside
the asset by the generation pipeline).

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
# (includes one fixture per adversarial mutation from the round-1 review
# and one per rigid-rebind / seam / boundary blind spot from the round-3
# probes — 39 tests):
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
| V3' co-deformation | linear-blend-skinned displacement vs the clip's first frame, over all vertices **outside the waving arm's GEOMETRIC limb volume, fixed at BIND**: the limb side of the plane through the upper-arm joint perpendicular to the upper-arm bone, within 12% *h* of the arm's bind bone chain. Membership keys off geometry, never off weights — re-pointing a torso vertex's weights at the arm (the spec's own named disqualifier: *"an asset whose arm is welded to the torso … does not pass"*) no longer removes it from the check | <= 0.5% of them may move > 3% *h*, none > 8% *h* |
| V4 weight leakage | vertices dominated by a head/neck joint carrying > 0.2 total arm-chain weight (either arm) | zero vertices |
| V4' reverse leakage | vertices **DOMINATED by an arm-chain joint (clavicle/shoulder included)** inside the geometric head region: bind height above the topmost neck joint AND within 45% *h* of the head joints' bind bone star. V4 only inspects head-dominated vertices, so a cheek region owned by `RightShoulder` never entered it | zero vertices |
| V5 loop seams | in `idle` and `walk`/`run`, every channel's end value equals its start value | <= 1% *h* translation, <= 2° rotation, <= 1% scale |
| V6 jump | at some frame, **every** foot joint (`foot\|toe\|ankle`) rises above its clip-start world height | >= 8% *h* |
| V7 limb integrity | in **every** clip: skinned geometry must ride the bones it wraps. Bone segments = parent→child world segments over the skin's joints (segments from the top joint and zero-length segments ignored). Per vertex, `d_bind` = bind-pose distance (raw `POSITION`, identity joint deltas) to the nearest bone segment **of its own weighted joints**; at each sampled frame, `d_t` = the same distance with each candidate bone carried rigidly by its joint's delta `G·IBM` — so geometry rigidly attached to any of its joints keeps `d_t = d_bind` exactly, which is what makes muzzles/ears/tail tips hanging off a **leaf joint** benign *in principle* (the joint's frame rotates with them; a bare segment cannot express that orientation), and no joint is ever exempted by name. Catches detached/doubled limbs (a second forearm lagging behind the real one) even when their dominant joint is the V3'-exempt waving arm chain | per frame <= 0.2% of skinned vertices beyond `min(1.5·d_bind, d_bind + 3% h) + 3% h`; none beyond `min(2·d_bind, d_bind + 8% h) + 8% h`. The multiplicative slack (skin sliding scales with wrap radius) is **capped in absolute terms**: skin slide is bounded by tissue scale, and without the cap a limb-shaped mass parked 25% *h* from every bone would earn 12% *h* of free drift |
| V7' mis-binding at bind | V7 is invariant **by construction** for any vertex bound 100% to one joint (`skinned = delta_k·v_bind`, so `d_t ≡ d_bind` at every frame) — a paw chunk rebound rigidly to the hips rides its wrong bone forever and V7 cannot see it. V7' judges the **binding itself, at bind**: a vertex is mis-bound when ALL its weighted joints' bone stars are > 10% *h* away while some OTHER bone is < 10% *h* away and **2.5× closer**. Legitimately-far skin (muzzle, ears, belly) is safe in principle: its own bone IS the nearest bone, so the dominance factor can never fire. A **hard offender** is any single influence >= 0.35 weight bound > 20% *h* from its bones under the same nearness/dominance test (catches small blended fragments, e.g. 0.55 Hand / 0.45 Spine2, whose dominant bone is nearby). Plus **limb-scale sanity**: every skin joint's scale — rest pose and every animated scale value in every clip — must be 1 per component | soft offenders <= 0.2% of skinned vertices; hard offenders = 0; scale within \|s−1\| <= 0.05 |
| V8 edge strain | **ALL edges** of every skinned primitive (no exemption for arm-touching triangles — that exemption dropped exactly the arm/torso boundary where the skin tears), in **every clip**, against **both** baselines: the BIND length (raw `POSITION`) and the clip's first frame. Edges wholly inside a geometric limb volume use looser caps (elbow/shoulder folds legitimately stretch compressed crease skin). A ratio offense requires the edge to also open > 0.2% *h* (noise floor for micro-edges); the absolute cap is independent of it | <= 0.5% of edges past 1.5× (2× limb-interior); none past 3× (4× limb-interior); none opening > 3% *h* in absolute length |
| V9 seam cracks | vertices co-located within 1e-5 (model units) at bind are one weld group (UV/texture seams duplicate vertices); in every clip, at every sampled frame, the max intra-group separation must stay tiny or the texture seam visibly cracks apart | <= 0.5% *h* per group, zero groups past it |

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
crossfade). GitHub Pages serving the branch root works the same way:
`/preview/index.html` loads `/assets/kitty/…` by relative path.

On load the camera **frames the whole character** (feet included) with an
exact projected-corner fit: the union of the skinned bounds across the
initial clip's loop is projected corner-by-corner into both the vertical
and horizontal fov, so nothing is cropped at any viewport aspect. The
default view is the rig's **front three-quarter**: the facing is derived
from the skeleton (up × the left→right shoulder axis, yawed 35° and
elevated ~18°), never hard-coded to an axis. The page exposes
`window.__kitty = { clips, play(name), current, framing() }` for automated
playtests — `framing()` returns the framing box corners in NDC under the
current camera (all `|x|,|y| <= 1` means fully in frame).
