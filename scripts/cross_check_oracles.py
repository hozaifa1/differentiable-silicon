#!/usr/bin/env python
"""V4 in miniature: do two oracles agree about the PHYSICS at one design point?

    python scripts/cross_check_oracles.py --a mock --b devsim --d 3

Two solvers built from different constitutive models will never agree on
magnitudes -- an analytic soft-min MOSFET and a 2-D drift-diffusion solve with a
meshed Miller ferroelectric are not the same function. What they must agree on,
if the optimisation is being steered by physics rather than by numerics, is:

* the SIGN of d(FoM_j)/d(theta_i) for every one of the 7 x D entries, and
* the RANK ORDER of |d(FoM_j)/d(theta_i)| across the D design variables, i.e.
  which knob each figure of merit is most sensitive to.

This is the cheap early warning for V4 on D6. It costs 2D+1 calls per oracle --
at d=3 that is seven DEVSIM points, about four minutes -- and it is run before
the flagship rather than after, because a sign disagreement found on D6 is a
disagreement you no longer have time to explain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from diffsilicon.shared.contract import DIFFERENTIABLE_OUTPUTS, make_oracle_input  # noqa: E402
from diffsilicon.shared.design import get_design, nominal_theta  # noqa: E402
from diffsilicon.shim.adjoint import ShimConfig, fd_jacobian  # noqa: E402


def jacobian_for(backend: str, theta: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    os.environ["ORACLE_BACKEND"] = backend
    template = make_oracle_input(theta)
    cfg = ShimConfig(alpha=alpha, backend=backend, max_oracle_calls=10_000)
    return fd_jacobian(theta, template, cfg, central=True)


def _rank(v: np.ndarray) -> np.ndarray:
    """Rank of |v| within the row, 0 = largest. Ties broken by index, as np does."""
    order = np.argsort(-np.abs(v))
    r = np.empty_like(order)
    r[order] = np.arange(v.size)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="mock")
    ap.add_argument("--b", default="devsim")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.04)
    ap.add_argument("--out", default=str(_REPO / "results" / "runs" / "cross_check.json"))
    args = ap.parse_args()

    spec = get_design(args.d)
    theta = nominal_theta(args.d)

    ja, ya = jacobian_for(args.a, theta, args.alpha)
    jb, yb = jacobian_for(args.b, theta, args.alpha)

    # Compare in RELATIVE units: d log(FoM) / d theta_i, so that a solver whose
    # currents are 3x larger everywhere is not reported as disagreeing.
    scale_a = np.where(np.abs(ya) > 0, np.abs(ya), 1.0)[:, None]
    scale_b = np.where(np.abs(yb) > 0, np.abs(yb), 1.0)[:, None]
    ra, rb = ja / scale_a, jb / scale_b

    sign_ok = np.sign(ra) == np.sign(rb)
    rows = []
    for j, name in enumerate(DIFFERENTIABLE_OUTPUTS):
        rank_a, rank_b = _rank(ra[j]), _rank(rb[j])
        rows.append({
            "fom": name,
            f"y_{args.a}": float(ya[j]),
            f"y_{args.b}": float(yb[j]),
            "d_rel_a": [float(v) for v in ra[j]],
            "d_rel_b": [float(v) for v in rb[j]],
            "signs_agree": [bool(v) for v in sign_ok[j]],
            "rank_a": [int(v) for v in rank_a],
            "rank_b": [int(v) for v in rank_b],
            "rank_order_agrees": bool(np.array_equal(rank_a, rank_b)),
        })

    summary = {
        "oracle_a": args.a,
        "oracle_b": args.b,
        "d": args.d,
        "alpha": args.alpha,
        "theta": [float(v) for v in theta],
        "names": list(spec.names),
        "sign_agreement": float(np.mean(sign_ok)),
        "rank_order_agreement": float(np.mean([r["rank_order_agrees"] for r in rows])),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    w = max(len(n) for n in spec.names) + 2
    print(f"\n=== {args.a} vs {args.b}, d={args.d}, alpha={args.alpha} ===")
    print("FoM        " + "".join(f"{n:>{w}}" for n in spec.names) + "   ranks agree")
    for r in rows:
        marks = "".join(
            f"{('+' if s > 0 else '-' if s < 0 else '0') + ('' if ok else '!'):>{w}}"
            for s, ok in zip(np.sign(r["d_rel_a"]), r["signs_agree"], strict=True)
        )
        print(f"{r['fom']:<11}{marks}   {'yes' if r['rank_order_agrees'] else 'NO'}")
    print(f"\nsign agreement       {summary['sign_agreement'] * 100:.1f}% of {7 * args.d} entries")
    print(f"rank-order agreement {summary['rank_order_agreement'] * 100:.1f}% of 7 rows")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
