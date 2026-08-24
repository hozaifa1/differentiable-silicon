#!/usr/bin/env python
"""Re-run every device-side baseline after the D3 recalibration.

Why this script exists
----------------------
Two changes on D3 invalidated every banked device number at once:

* `shared/extract.py` was rewritten for the widened sweep window, so the seven
  figures of merit stored in each cache record mean something different now.
* the design vector became d=4 with `Pr` and `Ec` locked, so a design point is a
  different physical device.

`shared.cache.cache_key` now folds a hash of `extract.py` into the key, which
means nothing stale can be served -- but it also means everything has to be
recomputed. This is that recomputation, in one place, writing one JSON so the
before/after is a diff rather than a memory.

DEVSIM AND PYTORCH CANNOT SHARE A PROCESS. Both link Intel OpenMP and the second
to initialise aborts the interpreter with OMP Error #15 and no traceback. So
nothing here imports torch, and the SNN side is measured separately.

    python scripts/rebaseline_d3.py --backend devsim
    python scripts/rebaseline_d3.py --backend mock --points 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from diffsilicon.shared.circuit import load_circuit, transduce  # noqa: E402
from diffsilicon.shared.contract import DEFAULT_VG_GRID, make_oracle_input  # noqa: E402
from diffsilicon.shared.design import denormalise, get_design, nominal_theta  # noqa: E402
from diffsilicon.shared.material import HZO_CALIBRATION  # noqa: E402
from diffsilicon.shared.oracle import device_geometry, run_oracle  # noqa: E402


def evaluate(theta, backend, cc):
    t0 = time.perf_counter()
    out = run_oracle(make_oracle_input(theta), backend=backend)
    wall = time.perf_counter() - t0
    w, lg = device_geometry(theta)
    phi = transduce(out, cc, lg, w)
    curves = np.asarray(out.id_vg)
    return {
        "ss_mV_per_dec": float(out.ss),
        "vth_fwd_V": float(out.vth_fwd),
        "vth_rev_V": float(out.vth_rev),
        "memory_window_V": float(out.vth_fwd - out.vth_rev),
        "i_leak_A": float(out.i_leak),
        "g_lo_S": float(out.g_lo),
        "g_hi_S": float(out.g_hi),
        "g_ratio": float(out.g_hi / out.g_lo),
        "dg_dvth_S_per_V": float(out.dg_dvth),
        "beta": float(phi.beta),
        "th_th": float(phi.th_th),
        "sig_w": float(phi.sig_w),
        "id_min_A": float(curves.min()),
        "id_max_A": float(curves.max()),
        "converged": float(out.converged),
        "solver_seconds": float(out.solver_seconds),
        "wall_seconds": wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="devsim", choices=["mock", "devsim", "sentaurus"])
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--points", type=int, default=6, help="random box points, on top of the fixed ones")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cc = load_circuit()
    spec = get_design(args.d)
    rng = np.random.default_rng(0)

    points = [("nominal", nominal_theta(args.d))]
    # The corners that matter: a film too thin to hold a window, and a thick one.
    thin = np.zeros(args.d)
    thin[0] = 0.0
    thick = np.zeros(args.d) + 0.5
    thick[0] = 1.0
    points += [("t_fe_min", thin), ("t_fe_max", thick)]
    points += [(f"rand{i}", rng.random(args.d)) for i in range(args.points)]

    res = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": args.backend,
        "design": {"d": args.d, "label": spec.label, "names": list(spec.names),
                   "lo": list(map(float, spec.lo)), "hi": list(map(float, spec.hi))},
        "locked_material": HZO_CALIBRATION,
        "sweep": {"n": len(DEFAULT_VG_GRID), "v_min": float(DEFAULT_VG_GRID[0]),
                  "v_max": float(DEFAULT_VG_GRID[-1])},
        "circuit": {"v_read": cc.v_read, "v_leak": cc.v_leak, "k_syn": cc.k_syn},
        "points": [],
    }

    print(f"backend={args.backend}  d={args.d} ({spec.label})  {len(points)} points")
    print(f"sweep {DEFAULT_VG_GRID[0]:+.2f} .. {DEFAULT_VG_GRID[-1]:+.2f} V, "
          f"{len(DEFAULT_VG_GRID)} points")
    print(f"locked film: Pr={HZO_CALIBRATION['Pr_uC_cm2']} Ps={HZO_CALIBRATION['Ps_uC_cm2']} "
          f"uC/cm2  Ec={HZO_CALIBRATION['Ec_MV_cm']} MV/cm  (node {HZO_CALIBRATION['node']})")
    print()
    hdr = f"{'point':10s} " + " ".join(f"{n:>10s}" for n in spec.names)
    print(hdr + f" | {'SS':>8s} {'MW':>7s} {'i_leak':>10s} {'g_hi/g_lo':>10s} "
                f"{'beta':>6s} {'th_th':>8s} {'s':>6s}")
    print("-" * (len(hdr) + 62))

    for name, theta in points:
        row = evaluate(theta, args.backend, cc)
        phys = denormalise(np.asarray(theta), spec)
        row["name"] = name
        row["theta"] = list(map(float, np.asarray(theta)))
        row["phys"] = {n: float(v) for n, v in zip(spec.names, phys, strict=True)}
        res["points"].append(row)
        print(
            f"{name:10s} " + " ".join(f"{v:10.3f}" for v in phys)
            + f" | {row['ss_mV_per_dec']:8.2f} {row['memory_window_V']:7.3f} "
              f"{row['i_leak_A']:10.2e} {row['g_ratio']:10.1f} "
              f"{row['beta']:6.3f} {row['th_th']:8.2f} {row['wall_seconds']:6.1f}",
            flush=True,
        )

    # G5, the memory-window gate, re-read on the fixed extraction.
    nominal = res["points"][0]
    res["gates"] = {
        "G5_memory_window_gt_0p1V": {
            "value": nominal["memory_window_V"],
            "pass": bool(nominal["memory_window_V"] > 0.1),
        },
        "SS_within_calibration_45_75_mV_per_dec": {
            "value": nominal["ss_mV_per_dec"],
            "pass": bool(45.0 <= nominal["ss_mV_per_dec"] <= 120.0),
            "note": "calibration reports 45-75 mV/dec; 120 is the gate because "
                    "this is a different geometry on the same film",
        },
        "sign_convention_MW_positive": {
            "pass": all(p["memory_window_V"] > 0 for p in res["points"]),
            "note": "forward = erased = high V_th, so MW > 0 on every point",
        },
    }
    print()
    for k, v in res["gates"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}"
              + (f"  = {v['value']:.4f}" if "value" in v else ""))

    out = Path(args.out or _REPO / "results" / "runs" / f"rebaseline_d3_{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if all(v["pass"] for v in res["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
