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

The kitty is **generated, not hand-modeled**, with Tripo3D:

1. **Arms-apart derivative** of the `reference/` views (so the arms rig as
   separate limbs, not fused to the torso).
2. **Multiview-to-model** — Tripo3D `multiview_to_model` from the four
   orthographic views.
3. **Auto-rig** — Tripo3D biped/humanoid rig.
4. **Animations** — preset retargets for `idle` / `jump` / `walk` (or `run`)
   plus an authored `wave` (one arm raised, at least two oscillations, body
   held stable).
5. **Single-GLB merge** — all four clips in `assets/kitty/kitty.glb`
   (spec-allowed fallback: four files `assets/kitty/kitty-<clip>.glb`).

The exact prompts, tool/model versions, and per-stage Tripo3D task IDs are
recorded in `assets/kitty/kitty.provenance.json` (committed alongside the
asset by the generation pipeline).

### Re-running validation

Requires Node >= 20; zero dependencies, no install step.

```sh
# validate the asset (assets/kitty/kitty.glb or the four-file layout);
# prints a PASS/FAIL table per check and exits non-zero on any failure:
node scripts/validate-kitty.mjs

# validate a specific file or directory:
node scripts/validate-kitty.mjs path/to/kitty.glb

# run the validator's self-test on synthetic in-memory GLB fixtures:
npm test
```

Checks: >= 1 skin; the four clips by exact name (`walk` or `run` accepted);
<= 20,000 rendered triangles; a PBR base-color texture with present image
data on a skinned mesh; and the wave-arm check — arm rotation >= 45° with
>= 2 oscillations while the root barely translates (<= 5% of model height),
torso joints barely rotate (<= 15°), and torso vertices barely move under
linear-blend skinning (95th percentile <= 3% of model height). Thresholds
and the joint-name patterns live in documented constants (`THRESHOLDS`,
`JOINT_NAME_PATTERNS`) at the top of `scripts/validate-kitty.mjs`. CI runs
the same self-test and validator on every push and pull request
(`.github/workflows/validate-kitty.yml`).

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
