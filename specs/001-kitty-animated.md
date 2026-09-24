# 001 — Animated kitty: rigged 3D model with idle / jump / walk / wave

Build one game-ready 3D character — **this specific kitty** — from the
reference images, fully rigged, with four animation clips. The model is
**generated from the reference images** with a text/image-to-3D tool
(Tripo3D's image / multiview-to-3D), then rigged and animated. Not
hand-modeled, not sourced from an asset library.

## The character

A chubby, upright, bipedal cartoon cat (see `reference/`): warm
yellow-orange fur, a cream/ivory belly patch, big round green eyes with
large pupils, tufted cheeks and small head tufts, a little pink nose and
tongue, short arms ending in rounded front paws, stubby legs, and a long
tail that curls up at the tip. The finished model must clearly read as
**this** cat — its color, proportions, face, and curled tail.

## Reference images (multiview input)

Four orthographic views live in `reference/`:

- `reference/kitty-front.png`
- `reference/kitty-back.png`
- `reference/kitty-left.png`
- `reference/kitty-right.png`

Use them as the **multiview input** to the 3D generator (Tripo3D
`multiview_to_model`, or image-to-3D anchored on the front view with the
others as guidance). Map the views to the tool's expected input order.

## Deliverables

- **`assets/kitty/kitty.glb`** — the rigged character carrying **four
  animation clips named exactly `idle`, `jump`, `walk`, `wave`** (a
  `run` cycle is an acceptable substitute for `walk` — name that clip
  `walk` or `run`). Rigged (≥ 1 skin), PBR textures, **≤ 20,000 rendered
  triangles**. A single GLB carrying all four clips is preferred; if a
  single-file merge cannot preserve the rig, four files
  `assets/kitty/kitty-<clip>.glb` are an acceptable fallback.
- **Separate, articulated arms — the load-bearing requirement.** The
  front paws / arms must be modeled and rigged as **distinct, fully
  articulated limbs, separated from the torso — not fused into the
  body**. The `wave` clip raises **one** arm and waves it (at least two
  oscillations) with clear, independent motion while the rest of the body
  holds a stable stance — **no torso tearing, stretching, or
  co-deformation** when the arm moves. This is the acceptance point the
  whole task turns on: an asset whose arm is welded to the torso, so the
  body deforms when it waves, does not pass.
- **Generated, not modeled.** Every mesh comes from the 3D tool driven by
  the reference images. No hand-modeling, no library assets.
- **`assets/kitty/kitty.provenance.json`** — the tool, the model/version,
  each prompt, the reference images used, and the per-stage tool task IDs
  (generation, rig, and each animation retarget).
- **Session-reported external usage.** The generating session reports the
  external-API units it consumed (tasks submitted, credits spent) and an
  estimated cost, **labeled as an estimate**, in its completion notes.
- **`scripts/validate-kitty.mjs`** — parses the GLB(s) and **fails
  unless**: there is ≥ 1 skin; the four required clips are present by
  name; the model is ≤ 20,000 rendered triangles; a PBR base-color
  texture is present; **and the wave-arm check passes** — in the `wave`
  clip at least one arm/hand joint rotates past a clear threshold while
  the root/hips translation stays minimal (the wave is arm-driven, not a
  whole-body lurch). Wire it into CI; green means the asset passes.
- **`preview/index.html`** — a **zero-build browser viewer**: it loads
  the kitty, gives orbit controls, and offers a control to switch between
  the four animations (idle autoplays). Use `<model-viewer>` or three.js
  from a pinned CDN or vendored locally — no build step, no bundler.
  Publish it as the **live preview URL** when the PR flips ready.
- **README section** — the generation pipeline, how to re-run the
  validation, and how to open the viewer.

## Animations

- **idle** — a calm resting loop (subtle breathing / tail sway); loops
  seamlessly.
- **jump** — a readable jump: anticipation crouch → leap → land.
- **walk** (or **run**) — a locomotion cycle that loops seamlessly.
- **wave** — one front paw lifts and waves a friendly greeting; the body
  holds steady and the waving arm moves independently of the torso (see
  the separate-arms requirement above).

## Out of scope

- In-game integration (loading the kitty into a running game scene).
- Sound, voice, or particle effects.
- Any second character or prop.

## Conduct

- Credentials never appear in any artifact, log, or commit — refer to
  them by **environment-variable name only**.
