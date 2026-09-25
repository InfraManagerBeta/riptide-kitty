#!/usr/bin/env python3
"""Round-3 batch driver for assets/kitty/previews/*.png against the
regenerated (single-front-view, consistent arms-apart) kitty.glb.

Reuses generate_previews_round2.py's job list unchanged: the wave clip's
authored timeline (duration 3.2s, raise/hold/lower envelope) and the
authored jump clip's keyframe timeline are BOTH produced by the same,
unmodified scripts/pipeline/keyframe_wave.py / keyframe_jump.py this
round, so the same sample times (rest/mid1/mid2/peak for wave; crouch/
apex/land for jump) still land on the same clip phases. Re-verified by
sampling this round's own clips before reusing the times (see
kitty.provenance.json round-3 stage "5e. preview timing re-check").
"""
import subprocess

GLB = "assets/kitty/kitty.glb"
OUT = "assets/kitty/previews"
RENDER = ["python3", "scripts/pipeline/render_preview_textured.py"]

JOBS = [
    ("wave-rest-front.png", "wave", 0.00, 0, [], "wave rest, front (t=0.00)"),
    ("wave-rest-34.png", "wave", 0.00, 45, [], "wave rest, 3/4 (t=0.00)"),
    ("wave-mid1-front.png", "wave", 0.71, 0, [], "wave mid-phase 1, front (t=0.71)"),
    ("wave-mid1-34.png", "wave", 0.71, 45, [], "wave mid-phase 1, 3/4 (t=0.71)"),
    ("wave-mid2-front.png", "wave", 1.10, 0, [], "wave mid-phase 2, front (t=1.10)"),
    ("wave-mid2-34.png", "wave", 1.10, 45, [], "wave mid-phase 2, 3/4 (t=1.10)"),
    ("wave-armup-front.png", "wave", 2.30, 0, [], "wave peak, front (t=2.30)"),
    ("wave-armup-34.png", "wave", 2.30, 45, [], "wave peak, 3/4 (t=2.30)"),
    ("wave-shoulder-closeup-peak.png", "wave", 2.30, 45,
     ["--zoom", "1.6", "--focus-y-frac", "0.28", "0.60"],
     "right shoulder close-up at wave peak (t=2.30)"),
    ("face-idle-t0.png", "idle", 0.0, 0,
     ["--zoom", "1.8", "--focus-y-frac", "0.55", "0.92"],
     "face, idle (t=0.00)"),
    ("face-wave-peak.png", "wave", 2.30, 0,
     ["--zoom", "1.8", "--focus-y-frac", "0.55", "0.92"],
     "face, wave peak (t=2.30) -- no arm leak"),
    ("jump-crouch.png", "jump", 0.15, 45, [], "jump crouch (t=0.15)"),
    ("jump-apex.png", "jump", 0.55, 45, [], "jump apex (t=0.55), arms intact"),
    ("jump-land.png", "jump", 0.85, 45, [], "jump land (t=0.85)"),
    ("walk-midstride.png", "walk", 1.18, 45, [], "walk mid-stride (t=1.18), arms intact"),
]


def main():
    for out_name, anim, t, az, extra, title in JOBS:
        out_path = f"{OUT}/{out_name}"
        cmd = RENDER + [GLB, out_path, "--anim", anim, "--time", str(t),
                         "--azimuth", str(az), "--title", title] + extra
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
