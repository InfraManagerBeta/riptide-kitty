#!/usr/bin/env python3
"""
scripts/pipeline/rebuild_round4_weights.py

FIX ROUND 4 (t1) orchestrator: runs the full skin-weight rebuild pipeline
(SCOPE item 1) against `assets/kitty/kitty.glb` in place (via a temp file,
only overwriting the output on success), in the order that matters:

  1. weld_seam_weights.py (PRE-PASS) -- start from a state where every
     UV-seam duplicate group already carries IDENTICAL joints/weights.
     Skipping this and welding only at the end was tried first and
     measured to be WORSE: this asset's pre-existing UV-seam duplicate
     groups (N1) already carried wildly different weights (one group's
     four co-located members ranged from 0.23 to 0.76 combined distal
     weight), each duplicate then diffuses toward ITS OWN UV island's
     local neighborhood in step 3, and averaging four already-diverged
     results back together at the very end reintroduced edge strain
     (measured: 1.37x before a final-only weld's average, 1.68-1.95x
     after it) instead of removing it.
  2. rebuild_face_weights.py -- geodesic head-vs-arm reclassification
     (N2) plus (inert, see that script) neck-band smoothing scaffolding.
  3. rebuild_arm_body_weights.py -- geodesic pool-preserving diffusion of
     the shoulder<->limb<->torso boundary (N1).
  4. weld_seam_weights.py (FINAL PASS) -- re-weld; pre-welding in step 1
     plus using weld-augmented adjacency during steps 2-3's diffusion
     (see those scripts) means this final pass only has to correct a
     SMALL residual drift, not the large original inconsistency, so it
     no longer undoes the diffusion's own result.

Each stage's own script remains independently runnable (and independently
documented) -- this orchestrator only fixes the ORDER, with no logic of
its own beyond chaining and writing per-stage reports next to the asset.

Usage:
    python3 rebuild_round4_weights.py <in.glb> <out.glb> [--report-dir DIR]
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args_list):
    cmd = [sys.executable, os.path.join(HERE, script)] + args_list
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_glb")
    ap.add_argument("output_glb")
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args()

    report_dir = args.report_dir
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    def rp(name):
        return ["--report", os.path.join(report_dir, name)] if report_dir else []

    with tempfile.TemporaryDirectory() as td:
        prewelded = os.path.join(td, "0_prewelded.glb")
        step1 = os.path.join(td, "1_face.glb")
        step2 = os.path.join(td, "2_arm_body.glb")
        step3 = os.path.join(td, "3_final.glb")

        run("weld_seam_weights.py", [args.input_glb, prewelded] + rp("weld_pre.json"))
        run("rebuild_face_weights.py", [prewelded, step1] + rp("face.json"))
        run("rebuild_arm_body_weights.py", [step1, step2] + rp("arm_body.json"))
        run("weld_seam_weights.py", [step2, step3] + rp("weld_final.json"))

        os.replace(step3, args.output_glb)
    print(f"\nwrote {args.output_glb}")


if __name__ == "__main__":
    main()
