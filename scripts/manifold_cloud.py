#!/usr/bin/env python
"""The reachable manifold: what the device can actually hand the network.

WHY THIS EXISTS -- it answers the single strongest objection to the project
-------------------------------------------------------------------------
The objection goes: "phi is only five scalars. The composition H(G(theta)) is a
map from four design numbers into R^5. So don't drag a TCAD solver through an
optimiser -- just optimise the five numbers freely, find the best phi*, and then
invert G once to find the device that gives it. Thirty solver calls, no adjoint,
no Tesseract."

That objection is correct IF phi* is reachable. The defence is that it is not:
the set of phi a real device can produce is a four-dimensional sheet embedded in
R^5, and a freely optimised phi* lands off it. But a defence nobody measured is
worth nothing, so this measures it.

WHAT IT DOES

1. Draws `--points` design vectors by Latin hypercube over the d=4 box, which
   covers the box far more evenly than independent uniform draws at this sample
   size.
2. Runs the OPEN solver at each and transduces to phi. That set of phi is the
   reachable manifold, sampled.
3. Reports its shape: how many dimensions it actually occupies (by principal
   components), and how tightly.
4. Saves everything so the figure can be drawn without re-running the solver.

The manifold is a property of the DEVICE PHYSICS and the circuit. It needs no
network, no training and no loss, which is why it is worth banking on its own:
it is the one expensive artefact that nothing downstream can invalidate.

    python scripts/manifold_cloud.py --points 200 --backend devsim

A note on what this samples. The design box is the four fabrication knobs on ONE
FIXED FILM. Before the D3 recalibration the box included the remanent
polarization and coercive field, and a free phi* would have looked far more
reachable then -- because "deposit a different film" was inside the search
space. It is not any more, which makes this argument stronger, not weaker.
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

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffsilicon.shared.circuit import load_circuit, transduce  # noqa: E402
from diffsilicon.shared.contract import make_oracle_input  # noqa: E402
from diffsilicon.shared.design import get_design  # noqa: E402
from diffsilicon.shared.oracle import run_oracle  # noqa: E402
from diffsilicon.snn.lif import PHI_KEYS  # noqa: E402

FOM_KEYS = ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth")


def latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """One sample per stratum per axis, then the axes shuffled independently.

    At 200 points in four dimensions an independent uniform draw leaves visible
    gaps and clumps, and the whole point here is to say what the reachable set
    looks like rather than where 200 coin flips landed.
    """
    rng = np.random.default_rng(seed)
    cut = np.arange(n)[:, None] + rng.random((n, d))
    out = cut / n
    for j in range(d):
        rng.shuffle(out[:, j])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--backend", default="devsim")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_REPO / "results" / "runs"
                                         / "manifold_cloud_d4.json"))
    args = ap.parse_args()

    import os
    os.environ["ORACLE_BACKEND"] = args.backend

    cc = load_circuit()
    spec = get_design(args.d)
    lo, hi = np.asarray(spec.lo), np.asarray(spec.hi)
    thetas = latin_hypercube(args.points, args.d, args.seed)

    print(f"backend={args.backend}  d={args.d} ({spec.label})  "
          f"{args.points} Latin-hypercube points")
    print(f"knobs: {', '.join(spec.names)}\n")

    rows, failed, refused = [], [], []
    t_start = time.time()
    for i, th in enumerate(thetas):
        try:
            out = run_oracle(make_oracle_input(th), args.backend)
        except RuntimeError as exc:
            failed.append({"i": i, "theta": th.tolist(),
                           "detail": str(exc).splitlines()[0][:200]})
            print(f"  [{i + 1:3d}/{args.points}] SOLVER FAILED", flush=True)
            continue
        if float(out.converged) <= 0.5:
            # Not a measurement -- see shared/extract.py. Recorded, not used.
            refused.append({"i": i, "theta": th.tolist()})
            print(f"  [{i + 1:3d}/{args.points}] REFUSED (not a measurement)",
                  flush=True)
            continue

        y = {k: float(getattr(out, k)) for k in FOM_KEYS}
        phys = lo + th * (hi - lo)
        lg = float(phys[spec.names.index("L_g")]) if "L_g" in spec.names else 40.0
        phi = transduce({k: jnp.asarray(v) for k, v in y.items()}, cc, lg)
        rows.append({
            "i": i,
            "theta": th.tolist(),
            "phys": {n: float(v) for n, v in zip(spec.names, phys, strict=True)},
            "y": y,
            "phi": {k: float(v) for k, v in zip(PHI_KEYS, phi, strict=True)},
            "solver_seconds": float(out.solver_seconds),
        })
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t_start
            rate = el / (i + 1)
            print(f"  [{i + 1:3d}/{args.points}] {len(rows)} usable, "
                  f"{len(failed)} failed, {len(refused)} refused  "
                  f"({el / 60:.1f} min, ~{rate * (args.points - i - 1) / 60:.0f} "
                  f"min left)", flush=True)

    if not rows:
        print("no usable points at all")
        return 1

    # --- the shape of the reachable set -------------------------------------
    # In standardised units, because the five coordinates are a decay (~0.6), two
    # conductances (~1e-6 and ~1e-4 S), a spike count (~5) and a noise fraction
    # (~0.1). A principal-component analysis of raw phi would describe the units.
    P = np.array([[r["phi"][k] for k in PHI_KEYS] for r in rows])
    mu, sd = P.mean(0), P.std(0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = (P - mu) / sd_safe
    _, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var = s**2 / max(len(Z) - 1, 1)
    frac = var / var.sum()

    print(f"\n{len(rows)} usable points of {args.points} "
          f"({len(failed)} solver failures, {len(refused)} refused)")
    print(f"wall clock {(time.time() - t_start) / 60:.1f} min\n")

    print("Spread of each of the five numbers the device can hand the network:")
    print(f"{'':10s} {'min':>12s} {'max':>12s} {'mean':>12s} {'sd':>12s}")
    for j, k in enumerate(PHI_KEYS):
        print(f"{k:10s} {P[:, j].min():12.4e} {P[:, j].max():12.4e} "
              f"{mu[j]:12.4e} {sd[j]:12.4e}")

    print("\nHow many dimensions the reachable set actually occupies:")
    cum = 0.0
    for j, f in enumerate(frac):
        cum += f
        print(f"  component {j + 1}: {f * 100:5.1f}% of the variance "
              f"(running total {cum * 100:5.1f}%)")
    n90 = int(np.searchsorted(np.cumsum(frac), 0.90) + 1)
    print(f"\n  -> {n90} of 5 directions carry 90% of it.")
    print("     Four knobs cannot fill five dimensions, and this is that fact "
          "measured\n     rather than asserted.")

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": args.backend, "d": args.d, "seed": args.seed,
        "requested": args.points, "usable": len(rows),
        "phi_keys": list(PHI_KEYS),
        "standardise": {"mean": mu.tolist(), "sd": sd.tolist()},
        "pca": {"components": vt.tolist(),
                "explained_variance_ratio": frac.tolist()},
        "points": rows, "solver_failed": failed, "refused": refused,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
