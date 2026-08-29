#!/usr/bin/env python
"""Which T1 Jacobian columns are real, and which are artefacts of the extraction.

    python scripts/t1_structural_zeros_d6.py

WHY THIS EXISTS, AND WHY THE ANSWER IN THE NOTES IS SLIGHTLY WRONG
------------------------------------------------------------------
The project has said, since D4, that on the commercial solver's fixed-mesh path
"only t_fe reaches the deck -- L_g, N_ch and t_IL live in the mesh -- so three of
four T1 Jacobian columns are identically zero."

Three of four is not what is on disk, and the difference matters. Measured
against the banked Sentaurus cache at the cross-check design point:

    column        Id-Vg curve vs centre    seven FoMs
    t_fe          DIFFERS                  all move        <- real physics
    L_g           BYTE-IDENTICAL           vth_fwd moves   <- NOT physics
    log10_N_ch    BYTE-IDENTICAL           none move       <- exactly zero
    t_IL          BYTE-IDENTICAL           none move       <- exactly zero

So TWO columns are identically zero, and the third is worse than zero: it is
SPURIOUSLY NON-ZERO. The solver returns the same current at every gate voltage
whether L_g is 38.4 nm or 41.6 nm, and the threshold still moves -- because the
constant-current criterion is I_crit = 100 nA * W / L_g, so shrinking the gate
raises the bar the curve has to cross and reads out a different V_th on an
identical curve.

That is a real number and it is not a device sensitivity. Anyone reading the T1
Jacobian and finding the L_g column non-zero would conclude the mesh path
carries some gate-length physics. It does not. This makes the distinction from
the cache rather than from an argument, so the writeup can state it as a
measurement.

Nothing here calls a solver: every point is already in results/cache/sentaurus/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

os.environ.setdefault("DIFFSILICON_PROVENANCE_DISABLE", "1")

CROSSCHECK = _REPO / "results" / "runs" / "cross_check_sentaurus_devsim_d4.json"
OUT = _REPO / "results" / "runs" / "t1_structural_zeros_d6.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from diffsilicon.shared.cache import CacheStore, cache_key
    from diffsilicon.shared.contract import DIFFERENTIABLE_OUTPUTS, make_oracle_input
    from diffsilicon.shared.design import get_design

    banked = json.loads(CROSSCHECK.read_text(encoding="utf-8"))
    theta = np.asarray(banked["theta"], dtype=np.float64)
    alpha = float(banked["alpha"])
    spec = get_design(int(banked["d"]))
    store = CacheStore("sentaurus")

    def record(t):
        p = store.path_for(cache_key(make_oracle_input(t), "sentaurus"))
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    centre = record(theta)
    if centre is None:
        print("centre point is not in the Sentaurus cache; nothing to measure.")
        return 2
    base = np.asarray(centre["id_vg"], dtype=np.float64)

    print(f"design point {list(theta)}, alpha = {alpha}")
    print(f"centre: ss = {centre['ss']:.6f} mV/dec, "
          f"vth_fwd = {centre['vth_fwd']:.6f} V\n")
    print(f"{'column':<12s} {'side':>5s}  {'Id-Vg curve':<16s} {'FoMs that move'}")
    print("-" * 72)

    cols = []
    for i, name in enumerate(spec.names):
        sides, moved_any, curve_same = [], set(), True
        for sgn in (+1.0, -1.0):
            t = theta.copy()
            t[i] = min(1.0, max(0.0, t[i] + sgn * alpha))
            rec = record(t)
            if rec is None:
                print(f"{name:<12s} {sgn:>+5.0f}  NOT CACHED")
                sides.append({"sign": sgn, "cached": False})
                continue
            same = bool(np.array_equal(np.asarray(rec["id_vg"], dtype=np.float64), base))
            curve_same = curve_same and same
            moved = [k for k in DIFFERENTIABLE_OUTPUTS
                     if float(rec[k]) != float(centre[k])]
            moved_any.update(moved)
            print(f"{name:<12s} {sgn:>+5.0f}  "
                  f"{'BYTE-IDENTICAL' if same else 'differs':<16s} "
                  f"{', '.join(moved) if moved else 'none'}")
            sides.append({"sign": sgn, "cached": True, "curve_identical": same,
                          "foms_that_move": moved})

        # The classification, and it is the point of the script.
        if not curve_same:
            verdict = "real -- the solver returns a different device"
        elif not moved_any:
            verdict = "structurally zero -- identical curve, identical FoMs"
        else:
            verdict = ("SPURIOUS -- identical curve, but the extraction moves: "
                       "I_crit = 100 nA * W / L_g reads a different threshold "
                       "off the same current")
        cols.append({"column": name, "curve_identical_both_sides": curve_same,
                     "foms_that_move": sorted(moved_any), "verdict": verdict,
                     "sides": sides})
        print(f"{'':<12s} {'':>5s}  -> {verdict}\n")

    real = [c["column"] for c in cols if "real" in c["verdict"]]
    zero = [c["column"] for c in cols if c["verdict"].startswith("structurally")]
    spur = [c["column"] for c in cols if c["verdict"].startswith("SPURIOUS")]

    print("=" * 72)
    print(f"real physics        : {', '.join(real) or 'none'}")
    print(f"structurally zero   : {', '.join(zero) or 'none'}")
    print(f"spuriously non-zero : {', '.join(spur) or 'none'}")
    print("\nThe project's notes say 'three of four columns are identically "
          "zero'.\nMeasured, it is TWO -- and the third is worse than zero, "
          "because it is\nnon-zero for a reason that is not the device.")

    Path(args.out).write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "results/cache/sentaurus/ via "
                  "results/runs/cross_check_sentaurus_devsim_d4.json",
        "backend": "sentaurus (fixed-mesh path)",
        "theta": [float(v) for v in theta], "alpha": alpha,
        "solver_calls_made": 0,
        "columns": cols,
        "real_physics": real,
        "structurally_zero": zero,
        "spuriously_nonzero": spur,
        "correction": "The D4/D5 notes say three of four T1 columns are "
                      "identically zero. Measured: two are (log10_N_ch, t_IL). "
                      "The third, L_g, returns a byte-identical Id-Vg curve but "
                      "a different vth_fwd, because the constant-current "
                      "criterion I_crit = 100 nA * W / L_g moves with the gate "
                      "length. That column is non-zero for a reason that is not "
                      "a device sensitivity.",
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
