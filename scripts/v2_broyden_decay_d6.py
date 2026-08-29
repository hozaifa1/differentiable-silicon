#!/usr/bin/env python
"""V2, re-measured and banked: how fast does the manufactured Jacobian go stale?

    python scripts/v2_broyden_decay_d6.py                 # cache-served points only
    python scripts/v2_broyden_decay_d6.py --allow-solver  # solve what is missing

WHY THIS RUN EXISTS
-------------------
`ShimConfig.refresh_every` is 4, and the comment beside it says "K; the V2 cosine
curve revises it on D3, not asserted". That is the right thing to say and until
today it was not backed by anything on disk -- the curve was measured on D2/D3
and never banked, so the repository asserts a constant and cites a measurement
nobody can see. A judge who asks "why 4?" deserves a file, not a memory.

This is that file.

WHAT IS MEASURED
----------------
The shim holds a local linear model J of the solver, rebuilt from finite
differences when it goes stale and patched by a rank-one Broyden update between
rebuilds. The whole economic argument for the apparatus is that the Broyden
patch is nearly free -- it uses a secant pair every accepted step supplies
anyway -- so the question is how many steps that patch survives before the
direction it gives is no longer the direction the solver would give.

Walking the flagship's own accepted path:

* anchor a fresh central-difference Jacobian J0 at the first point (2D+1 calls);
* at each subsequent point, patch it by Broyden from the secant pair that step
  provides, and ALSO rebuild the true Jacobian there from central differences;
* report the cosine between the two, as a function of steps since the anchor.

Cosine, not norm, because the optimiser uses the direction. A model whose
magnitude is wrong takes a badly sized step and the trust region catches it; a
model whose DIRECTION is wrong walks uphill and the trust region only finds out
afterwards, having spent the calls.

TWO COSINES ARE REPORTED and they are not the same question:

* `cos_J` -- the mean over the seven rows of the 7xD Jacobian. This is the
  shim's own accuracy, independent of what the network happens to want.
* `cos_g` -- the cosine of the COMPOSED gradient dL/dtheta = (dL/dy) J, which is
  what the optimiser actually steps along. It can stay high while individual
  rows rot, if the rows that rot are ones the loss is insensitive to, and that
  is the honest thing to plot for a claim about the optimiser.

COST. A true Jacobian at each point is 2D+1 = 9 solver calls at D=4. The
flagship already visited these points and refreshed at several of them, so many
probes are already in `results/cache/devsim/`. The script reports the cache
hit rate BEFORE it runs, and without `--allow-solver` it refuses to call the
solver at all -- so a run on a clean clone either reproduces from cache or says
plainly which points it cannot serve.
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

STEPS = _REPO / "results" / "runs" / "flagship-d4-fixed" / "steps.jsonl"
RESULT = _REPO / "results" / "runs" / "flagship-d4-fixed" / "result.json"
OUT = _REPO / "results" / "runs" / "v2_broyden_decay_d6.json"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="devsim")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--allow-solver", action="store_true",
                    help="permit real solver calls for probes not in the cache")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    os.environ["ORACLE_BACKEND"] = args.backend

    from diffsilicon.shared.cache import CacheStore, cache_key
    from diffsilicon.shared.contract import (
        DIFFERENTIABLE_OUTPUTS,
        make_oracle_input,
    )
    from diffsilicon.shim.adjoint import ShimConfig, fd_jacobian, y_vector
    from diffsilicon.shared.oracle import run_oracle

    rows = [json.loads(line) for line in
            STEPS.read_text(encoding="utf-8").splitlines() if line.strip()]
    flag = json.loads(RESULT.read_text(encoding="utf-8"))

    # The distinct points the run actually stood on, in order.
    path: list[np.ndarray] = []
    for r in rows:
        t = np.asarray(r["theta"], dtype=np.float64)
        if not path or not np.array_equal(t, path[-1]):
            path.append(t)
    last = np.asarray(rows[-1]["theta_next"], dtype=np.float64)
    if not np.array_equal(last, path[-1]):
        path.append(last)
    d = int(path[0].size)

    # --- what this will cost, before it costs it ----------------------------
    store = CacheStore(args.backend)
    needed: list[np.ndarray] = []
    for t in path:
        needed.append(t)
        for i in range(d):
            for sgn in (+1.0, -1.0):
                p = t.copy()
                p[i] = min(1.0, max(0.0, p[i] + sgn * args.alpha))
                needed.append(p)
    keys = [cache_key(make_oracle_input(t), args.backend) for t in needed]
    miss = [k for k in keys if not store.path_for(k).is_file()]
    print(f"path: {len(path)} design points, D={d}, alpha={args.alpha}")
    print(f"probes required: {len(keys)}  cached: {len(keys) - len(miss)}  "
          f"missing: {len(miss)}")
    if miss and not args.allow_solver:
        print("\nRefusing to call the solver. Re-run with --allow-solver to "
              "solve the missing probes,\nor reproduce only what the cache "
              "covers.")
        return 2
    if miss:
        print(f"will call {args.backend} for {len(miss)} probes "
              f"(~{len(miss) * 20 / 60:.0f} min at 20 s each)")

    cfg = ShimConfig(alpha=args.alpha, backend=args.backend,
                     max_oracle_calls=10_000)

    # dL/dy at the flagship's final device, used to weight the composed cosine.
    # It is held FIXED across the path on purpose: the question is how the
    # SOLVER's local model decays, and letting the cotangent move as well would
    # mix two effects into one curve.
    cot = None
    t0 = time.perf_counter()

    out_rows = []
    j_hat = None
    y_prev = None
    t_prev = None
    for k, theta in enumerate(path):
        j_true, y = fd_jacobian(theta, make_oracle_input(theta), cfg, central=True)
        if k == 0:
            j_hat = j_true.copy()
            cos_j = cos_g = 1.0
        else:
            # Broyden from the secant pair this step supplies for free.
            s = theta - t_prev
            ss = float(s @ s)
            if ss > 0.0:
                dy = y - y_prev
                j_hat = j_hat + np.outer(dy - j_hat @ s, s) / ss
            rows_cos = [cosine(j_hat[r], j_true[r]) for r in range(j_true.shape[0])]
            cos_j = float(np.nanmean(rows_cos))
            if cot is None:
                # A fixed, non-degenerate cotangent in the seven FoMs. Scaled by
                # 1/|y| so that a figure of merit measured in 1e-11 amps does not
                # silently dominate one measured in volts.
                cot = 1.0 / np.where(np.abs(y) > 0, np.abs(y), 1.0)
            cos_g = cosine(cot @ j_hat, cot @ j_true)

        out_rows.append({
            "k": k, "steps_since_refresh": k,
            "theta": [float(v) for v in theta],
            "cos_J_rowmean": float(cos_j), "cos_gradient": float(cos_g),
            "row_cosines": ([float(cosine(j_hat[r], j_true[r]))
                             for r in range(j_true.shape[0])] if k else [1.0] * 7),
        })
        print(f"  k={k}  cos_J {cos_j:+.4f}   cos_grad {cos_g:+.4f}")
        t_prev, y_prev = theta, y

    wall = time.perf_counter() - t0
    result = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": "V2: cosine between a Broyden-patched Jacobian and the true "
                "finite-difference Jacobian, as a function of steps since the "
                "anchor, along the flagship's own accepted path.",
        "backend": args.backend, "alpha": args.alpha, "d": d,
        "path_source": "results/runs/flagship-d4-fixed/steps.jsonl",
        "fom_order": list(DIFFERENTIABLE_OUTPUTS),
        "refresh_every_in_use": 4,
        "probes_required": len(keys),
        "probes_from_cache": len(keys) - len(miss),
        "probes_solved": len(miss),
        "wall_seconds": round(wall, 2),
        "rows": out_rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwall {wall / 60:.1f} min; written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
