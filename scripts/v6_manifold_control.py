#!/usr/bin/env python
"""V6: is the freely-optimised phi* actually reachable by a device?

*** SUPERSEDED 2026-08-28 (D5). THIS SCRIPT'S FREE ARM DOES NOT OPTIMISE. ***
*** Use `scripts/v6_free_refit_d5.py`. Do not re-run this one for a result. ***

The bug, kept here rather than silently fixed because the trap is worth knowing:

    z0 = Z[rng.integers(len(Z))] if r else Z.mean(0)

`Z` is standardised by the same cloud, so `Z.mean(0)` is EXACTLY the zero vector.
SciPy's Nelder-Mead builds its initial simplex by perturbing each coordinate of
x0 by 5% -- except a coordinate that is exactly zero, which it perturbs by
`zdelt = 0.00025` instead. Every coordinate here is zero, so the simplex is
2.5e-4 across against `xatol = 1e-4`. It is converged before it starts and
returns its own starting point; the later restarts never beat it.

Measured: the phi* this wrote into `v6_manifold_control_d4.json` equals the cloud
MEAN to 1.7e-14. So that file records the loss at the average device, not a free
optimum, and "1.9 device-spacings off the sheet" is really "the centroid of a
curved cloud lies off the cloud".

Re-run properly the numbers move a long way -- projected 1.0258 -> 1.1128, phi*
1.9 -> 13.5 spacings off the sheet -- and the D5 gate flips from failing at 0.79%
to passing at 8.6%. See `docs/D5_FINDINGS.md` section 2.

ANY optimiser started at an exactly-zero vector is exposed to this. If a
derivative-free search returns its starting point, check the simplex before
believing the answer.


THE OBJECTION THIS ANSWERS, which is the strongest one anybody raises
--------------------------------------------------------------------
"phi is five scalars. So don't drag a TCAD solver through an optimiser. Optimise
the five numbers freely, find the best phi*, then invert the device model once to
find the device that produces it. Thirty solver calls, no adjoint, no Tesseract."

That is a good objection and it is fatal IF phi* is reachable. Four fabrication
knobs cannot fill five dimensions, so the set of phi a real device can produce is
a SHEET in R^5, and a freely optimised phi* generally lands off it. This measures
whether that is true here rather than asserting it.

THREE ARMS, and the comparison between them is the whole point:

  FREE       optimise phi directly, ignoring whether any device can make it.
             Zero solver calls. This is the objection's own proposal, and its
             loss is a LOWER BOUND that nothing physical can beat.
  PROJECTED  take that phi*, find the reachable device closest to it, and score
             THAT. This is the objection's proposal actually cashed out -- you
             cannot build phi*, so you build the nearest thing.
  JOINT      descend through the solver, which is what this project does.

If PROJECTED is close to JOINT, the objection stands and the pipeline is
unnecessary. If JOINT is clearly better, the pipeline is earning its cost: the
best REACHABLE device is not the projection of the best imaginary one.

The reachable set comes from `scripts/manifold_cloud.py`, which must have been
run first. Nothing here calls the solver -- the cloud already paid for that.
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

from diffsilicon.snn.lif import PHI_KEYS  # noqa: E402


def _load_cloud(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    P = np.array([[r["phi"][k] for k in PHI_KEYS] for r in d["points"]])
    return d, P


def _standardise(P):
    """Work in units of the reachable set's own spread.

    The five coordinates are a decay (~0.6), two conductances (~1e-6 and ~1e-4
    S), a spike count (~5) and a noise fraction (~0.1). Any distance computed on
    raw phi is a statement about units. Standardising by the cloud's own mean and
    spread makes "far from the manifold" mean "far compared with how much devices
    actually vary", which is the only scale with meaning here.
    """
    mu, sd = P.mean(0), P.std(0)
    return mu, np.where(sd > 0, sd, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=str(_REPO / "results" / "runs"
                                           / "manifold_cloud_d4.json"))
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_REPO / "results" / "runs"
                                         / "v6_manifold_control_d4.json"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("SNN_TRAIN_MODE", "frozen")
    os.environ.setdefault("SNN_VJP", "fd")

    from scipy.optimize import minimize
    from tesseract_core import Tesseract

    cloud_path = Path(args.cloud)
    if not cloud_path.is_file():
        print(f"no manifold cloud at {cloud_path}; run scripts/manifold_cloud.py first")
        return 1
    cloud, P = _load_cloud(cloud_path)
    mu, sd = _standardise(P)
    Z = (P - mu) / sd
    print(f"reachable set: {len(P)} devices from {cloud_path.name}\n")

    api = _REPO / "tesseracts" / "snn-lif-ecg" / "tesseract_api.py"
    snn = Tesseract.from_tesseract_api(str(api))

    n_eval = {"n": 0}

    with snn:
        def loss_of_phi(vec):
            """The network's loss at an ARBITRARY phi -- reachable or not."""
            n_eval["n"] += 1
            d = dict(zip(PHI_KEYS, (float(v) for v in vec), strict=True))
            # Conductances must stay positive and ordered, or the weight mapping
            # divides by a negative span. The free optimiser has no physics to
            # stop it, so the bound is imposed here and recorded.
            d["g_min"] = max(d["g_min"], 1e-12)
            d["g_max"] = max(d["g_max"], d["g_min"] * 1.000001)
            r = snn.apply({**d, "seed": 0, "batch": 16, "smooth_spikes": False})
            return float(r["loss"])

        def loss_of_z(z):
            return loss_of_phi(mu + z * sd)

        # --- ARM 1: FREE ----------------------------------------------------
        print("ARM 1 -- FREE. Optimising phi directly, ignoring physics.")
        rng = np.random.default_rng(args.seed)
        best = None
        t0 = time.time()
        for r in range(args.restarts):
            z0 = Z[rng.integers(len(Z))] if r else Z.mean(0)
            res = minimize(loss_of_z, z0, method="Nelder-Mead",
                           options={"maxiter": args.maxiter, "xatol": 1e-4,
                                    "fatol": 1e-6})
            if best is None or res.fun < best.fun:
                best = res
            print(f"  restart {r + 1}/{args.restarts}: loss {res.fun:.6f}"
                  f"{'   <- best' if best is res else ''}", flush=True)
        z_star = np.asarray(best.x)
        phi_star = mu + z_star * sd
        print(f"\n  best free loss : {best.fun:.6f}  "
              f"({n_eval['n']} network evaluations, {(time.time() - t0) / 60:.1f} min)")
        print("  phi*           : " + "  ".join(
            f"{k}={v:.6g}" for k, v in zip(PHI_KEYS, phi_star, strict=True)))

        # --- is it reachable? ------------------------------------------------
        dist = np.linalg.norm(Z - z_star, axis=1)
        j = int(np.argmin(dist))
        # How far a typical device sits from its own nearest neighbour, as the
        # yardstick for "far".
        from scipy.spatial.distance import cdist
        DD = cdist(Z, Z)
        np.fill_diagonal(DD, np.inf)
        typical = float(np.median(DD.min(axis=1)))
        print(f"\n  distance from phi* to the nearest REACHABLE device : "
              f"{dist[j]:.3f}")
        print(f"  typical distance between neighbouring devices      : "
              f"{typical:.3f}")
        print(f"  -> phi* sits {dist[j] / max(typical, 1e-12):.1f} typical "
              f"device-spacings off the reachable sheet.")

        # --- ARM 2: PROJECTED -------------------------------------------------
        print("\nARM 2 -- PROJECTED. The reachable device closest to phi*.")
        phi_proj = P[j]
        l_proj = loss_of_phi(phi_proj)
        pt = cloud["points"][j]
        print("  device  : " + "  ".join(
            f"{k}={v:.4g}" for k, v in pt["phys"].items()))
        print(f"  loss    : {l_proj:.6f}")

        # --- best reachable device, for reference -----------------------------
        print("\nFor reference: the best device in the whole sampled cloud.")
        losses = [loss_of_phi(P[i]) for i in range(len(P))]
        i_best = int(np.argmin(losses))
        print(f"  loss {losses[i_best]:.6f} at " + "  ".join(
            f"{k}={v:.4g}" for k, v in cloud["points"][i_best]["phys"].items()))

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cloud": cloud_path.name, "n_devices": len(P),
        "free": {"loss": float(best.fun),
                 "phi": {k: float(v) for k, v in zip(PHI_KEYS, phi_star, strict=True)},
                 "evaluations": n_eval["n"]},
        "reachability": {"distance_to_nearest": float(dist[j]),
                         "typical_neighbour_distance": typical,
                         "in_spacings": float(dist[j] / max(typical, 1e-12))},
        "projected": {"loss": float(l_proj), "index": j,
                      "phys": cloud["points"][j]["phys"],
                      "phi": {k: float(v) for k, v in zip(PHI_KEYS, phi_proj, strict=True)}},
        "best_in_cloud": {"loss": float(losses[i_best]), "index": i_best,
                          "phys": cloud["points"][i_best]["phys"]},
        "all_cloud_losses": [float(v) for v in losses],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\nCompare `projected` against the flagship's final loss. If the")
    print("flagship is clearly lower, descending through the solver bought")
    print("something that projecting a free optimum does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
