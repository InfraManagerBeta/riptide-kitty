#!/usr/bin/env python3
"""One-off batch driver for this round's committed preview PNGs (see
kitty.provenance.json's "stage: visual proof" entry) -- regenerates every
existing assets/kitty/previews/*.png against the CURRENT (post arm-weight
fix) kitty.glb, using scripts/pipeline/render_preview_textured.py, plus
adds the two new views the round-2 work order specifically asks for
(a shoulder close-up at the wave peak, and a walk-clip frame) that did
not exist before this round.
"""
import subprocess
import sys

GLB = "assets/kitty/kitty.glb"
OUT = "assets/kitty/previews"
RENDER = ["python3", "scripts/pipeline/render_preview_textured.py"]

# (out_name, anim, time, azimuth, extra_args, title)
JOBS = [
    # wave: rest / two mid-wave phases / peak -- front and 3/4, per the
    # work order ("wave front + 3/4 views at rest, peak, and two
    # mid-wave phases"). Times: 0.00 rest (clip start); 0.71 / 1.10 are
    # the ForeArm oscillation's two extremes during the raised hold;
    # 2.30 is the validator's own measured peak hand-above-shoulder frame.
    ("wave-rest-front.png", "wave", 0.00, 0, [], "wave rest, front (t=0.00)"),
    ("wave-rest-34.png", "wave", 0.00, 45, [], "wave rest, 3/4 (t=0.00)"),
    ("wave-mid1-front.png", "wave", 0.71, 0, [], "wave mid-phase 1, front (t=0.71)"),
    ("wave-mid1-34.png", "wave", 0.71, 45, [], "wave mid-phase 1, 3/4 (t=0.71)"),
    ("wave-mid2-front.png", "wave", 1.10, 0, [], "wave mid-phase 2, front (t=1.10)"),
    ("wave-mid2-34.png", "wave", 1.10, 45, [], "wave mid-phase 2, 3/4 (t=1.10)"),
    ("wave-armup-front.png", "wave", 2.30, 0, [], "wave peak, front (t=2.30)"),
    ("wave-armup-34.png", "wave", 2.30, 45, [], "wave peak, 3/4 (t=2.30)"),

    # NEW this round: close-up of the right shoulder at the wave peak --
    # the work order's specific ask, to show the whole arm (upper arm
    # through paw) as one continuous, unstretched limb at the shoulder
    # seam instead of a torn "sleeve".
    ("wave-shoulder-closeup-peak.png", "wave", 2.30, 45,
     ["--zoom", "1.6", "--focus-y-frac", "0.28", "0.60"],
     "right shoulder close-up at wave peak (t=2.30)"),

    # face close-ups (round-1 provenance: checking no arm-chain weight
    # leak into head/neck skin -- V4). Regenerated against this round's
    # asset for freshness.
    ("face-idle-t0.png", "idle", 0.0, 0,
     ["--zoom", "1.8", "--focus-y-frac", "0.55", "0.92"],
     "face, idle (t=0.00)"),
    ("face-wave-peak.png", "wave", 2.30, 0,
     ["--zoom", "1.8", "--focus-y-frac", "0.55", "0.92"],
     "face, wave peak (t=2.30) -- no arm leak"),

    # jump: crouch -> apex -> land (times from this round's own foot-
    # height sampling: lowest ~0.15s, highest ~0.55s [also the provenance-
    # recorded 14.7%h rise frame], back down ~0.85s).
    ("jump-crouch.png", "jump", 0.15, 45, [], "jump crouch (t=0.15)"),
    ("jump-apex.png", "jump", 0.55, 45, [], "jump apex (t=0.55), arms intact"),
    ("jump-land.png", "jump", 0.85, 45, [], "jump land (t=0.85)"),

    # NEW this round: a walk-clip frame -- the work order's other new ask
    # ("one frame each of walk and jump showing the arms intact"); no
    # walk preview existed before this round.
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
