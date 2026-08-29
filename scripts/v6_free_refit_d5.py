#!/usr/bin/env python
"""V6, re-run: the FREE arm in the banked D4 file never actually optimised.

WHAT WENT WRONG, and why it matters more than it looks
------------------------------------------------------
`scripts/v6_manifold_control.py` optimises phi freely, in units of the reachable
cloud's own spread, and its first restart starts at the cloud's centroid:

    z0 = Z[rng.integers(len(Z))] if r else Z.mean(0)

Z is standardised by that same cloud, so `Z.mean(0)` is EXACTLY the zero vector.
SciPy's Nelder-Mead builds its initial simplex by perturbing each coordinate of
x0 by 5% -- except a coordinate that is exactly zero, which it perturbs by
`zdelt = 0.00025` instead. Every coordinate here is exactly zero, so the initial
simplex has a diameter of 2.5e-4 against a convergence tolerance `xatol` of 1e-4.
It is converged before it starts. The restart returned its own starting point,
and the five genuine restarts that followed never beat it.

Measured, on the banked file: phi* equals the cloud mean to 1.7e-14 in
standardised units. So `v6_manifold_control_d4.json` does not record a free
optimum. It records **the loss at the average device**, and

    "phi* sits 1.9 typical device-spacings off the reachable sheet"

is really "the centroid of a curved cloud is 1.9 spacings from the nearest point
of that cloud". That is true, and it is nearly content-free: the centroid of any
curved sheet lies off it. It is not the claim the section is making.

This matters because that claim is the answer to the single strongest objection
to this project -- "skip the solver, optimise the five numbers freely, then build
the nearest device" -- and Figure 1 is built on it. An objection answered with a
number that does not mean what it says is worse than an objection not answered.

WHAT THIS DOES INSTEAD
----------------------
Makes the FREE arm as strong as it can honestly be made, because a weak free arm
flatters this project. Free optimisation is the arm we WANT to be strong: if it
wins, the pipeline is unnecessary and we need to know that before a judge does.

  * Differential evolution over the standardised box, which is a global search
    and does not care where it starts;
  * then a Nelder-Mead polish from the winner, with an EXPLICIT non-degenerate
    initial simplex so the failure above cannot recur;
  * plus polishes from the best cloud device and from the centroid, so the old
    answer is still in the running and can only be beaten, never lost.

THE BOX. Free means free of the device, not free of arithmetic. phi is searched
over the cloud's own range padded by one standardised unit per coordinate -- so
anything a device can produce is inside, plus a generous margin outside. Letting
it run to infinity would just find the degenerate corner where the network stops
being a classifier, and reporting that as "the free optimum" would be a strawman
in our own favour. The padding is recorded in the output so the choice is
auditable.

g_min < g_max is imposed, as in the original: the weight mapping divides by their
span and the free optimiser has no physics to keep them ordered.

NO SOLVER CALLS. The reachable cloud already paid for those.

    python scripts/v6_free_refit_d5.py
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

import os  # noqa: E402

os.environ.setdefault("SNN_TRAIN_MODE", "frozen")
os.environ.setdefault("SNN_VJP", "fd")

from diffsilicon.snn.lif import PHI_KEYS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=str(_REPO / "results" / "runs"
                                           / "manifold_cloud_d4.json"))
    ap.add_argument("--pad", type=float, default=1.0,
                    help="box padding beyond the cloud's range, in standardised units")
    ap.add_argument("--maxiter", type=int, default=60, help="DE generations")
    ap.add_argument("--popsize", type=int, default=12, help="DE population multiplier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_REPO / "results" / "runs"
                                         / "v6_manifold_control_d5.json"))
    args = ap.parse_args()

    import torch
    from scipy.optimize import differential_evolution, minimize
    from scipy.spatial.distance import cdist
    from tesseract_core import Tesseract

    # One thread: this runs alongside the solver sweep, and the network is tiny
    # enough that thread contention costs more than the parallelism buys.
    torch.set_num_threads(1)

    cloud = json.loads(Path(args.cloud).read_text(encoding="utf-8"))
    P = np.array([[r["phi"][k] for k in PHI_KEYS] for r in cloud["points"]])
    mu, sd = P.mean(0), P.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (P - mu) / sd
    print(f"reachable set: {len(P)} devices from {Path(args.cloud).name}")

    lo = Z.min(0) - args.pad
    hi = Z.max(0) + args.pad
    print("search box, standardised:")
    for k, a, b in zip(PHI_KEYS, lo, hi, strict=True):
        print(f"  {k:7s} [{a:7.3f}, {b:7.3f}]   (cloud spans "
              f"[{Z[:, PHI_KEYS.index(k)].min():7.3f}, "
              f"{Z[:, PHI_KEYS.index(k)].max():7.3f}])")

    api = _REPO / "tesseracts" / "snn-lif-ecg" / "tesseract_api.py"
    snn = Tesseract.from_tesseract_api(str(api))
    n_eval = {"n": 0, "t0": time.time(), "best": np.inf}

    with snn:
        def loss_of_phi(vec) -> float:
            n_eval["n"] += 1
            d = dict(zip(PHI_KEYS, (float(v) for v in vec), strict=True))
            d["g_min"] = max(d["g_min"], 1e-12)
            d["g_max"] = max(d["g_max"], d["g_min"] * 1.000001)
            r = snn.apply({**d, "seed": 0, "batch": 16, "smooth_spikes": False})
            v = float(r["loss"])
            if v < n_eval["best"]:
                n_eval["best"] = v
            if n_eval["n"] % 100 == 0:
                el = time.time() - n_eval["t0"]
                print(f"    {n_eval['n']:6d} evals, best {n_eval['best']:.6f}, "
                      f"{el / 60:.1f} min ({el / n_eval['n']:.2f} s/eval)", flush=True)
            return v

        def loss_of_z(z) -> float:
            return loss_of_phi(mu + np.asarray(z) * sd)

        # --- global search --------------------------------------------------
        print("\nARM 1a -- differential evolution over the free box.")
        t0 = time.time()
        de = differential_evolution(
            loss_of_z, list(zip(lo, hi, strict=True)),
            maxiter=args.maxiter, popsize=args.popsize, tol=1e-8,
            mutation=(0.4, 1.0), recombination=0.8, init="sobol",
            polish=False, seed=args.seed, disp=False,
        )
        print(f"  DE best {de.fun:.6f} after {n_eval['n']} evals "
              f"({(time.time() - t0) / 60:.1f} min)")

        # --- local polish, from three different places ----------------------
        # An EXPLICIT initial simplex, one standardised unit across. This is the
        # line whose absence caused the original failure, so it is not a default.
        def polish(z0, label):
            z0 = np.asarray(z0, dtype=float)
            simplex = np.vstack([z0, *(z0 + np.eye(len(z0))[i] * 0.5
                                       for i in range(len(z0)))])
            r = minimize(loss_of_z, z0, method="Nelder-Mead",
                         options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-8,
                                  "initial_simplex": simplex})
            print(f"  polish from {label:16s} -> {r.fun:.6f}", flush=True)
            return r

        print("\nARM 1b -- Nelder-Mead polish, explicit non-degenerate simplex.")
        cands = [polish(de.x, "the DE winner")]
        cands.append(polish(np.zeros(len(PHI_KEYS)), "the cloud centroid"))
        losses_cloud = None
        best_dev = int(np.argmin([np.inf]))  # placeholder, replaced below

        # The best device in the cloud, scored first because it is also reported.
        print("\n  scoring all 192 reachable devices (needed for the other arms)",
              flush=True)
        losses_cloud = [loss_of_phi(P[i]) for i in range(len(P))]
        best_dev = int(np.argmin(losses_cloud))
        cands.append(polish(Z[best_dev], "the best device"))

        best = min(cands, key=lambda r: r.fun)
        z_star = np.asarray(best.x)
        phi_star = mu + z_star * sd
        print(f"\n  FREE optimum: {best.fun:.6f}  ({n_eval['n']} network evaluations)")
        print("  phi*        : " + "  ".join(
            f"{k}={v:.6g}" for k, v in zip(PHI_KEYS, phi_star, strict=True)))

        # --- reachability ---------------------------------------------------
        dist = np.linalg.norm(Z - z_star, axis=1)
        j = int(np.argmin(dist))
        DD = cdist(Z, Z)
        np.fill_diagonal(DD, np.inf)
        typical = float(np.median(DD.min(axis=1)))
        print(f"\n  distance from phi* to the nearest REACHABLE device : {dist[j]:.3f}")
        print(f"  typical distance between neighbouring devices      : {typical:.3f}")
        print(f"  -> {dist[j] / typical:.2f} typical device-spacings off the sheet")

        # --- projected ------------------------------------------------------
        print("\nARM 2 -- PROJECTED: the reachable device closest to phi*.")
        l_proj = losses_cloud[j]
        print("  device : " + "  ".join(
            f"{k}={v:.4g}" for k, v in cloud["points"][j]["phys"].items()))
        print(f"  loss   : {l_proj:.6f}")
        print(f"\nbest device in the cloud: {losses_cloud[best_dev]:.6f} at " +
              "  ".join(f"{k}={v:.4g}"
                        for k, v in cloud["points"][best_dev]["phys"].items()))

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "supersedes": "v6_manifold_control_d4.json",
        "why": ("the D4 free arm's first Nelder-Mead restart started at the "
                "standardised origin, where SciPy builds a 2.5e-4-wide initial "
                "simplex against xatol=1e-4, so it returned its own starting "
                "point; phi* in that file is the cloud MEAN to 1.7e-14"),
        "cloud": Path(args.cloud).name, "n_devices": len(P),
        "standardise": {"mean": mu.tolist(), "sd": sd.tolist()},
        "search_box_z": {"lo": lo.tolist(), "hi": hi.tolist(), "pad": args.pad},
        "free": {"loss": float(best.fun),
                 "phi": {k: float(v) for k, v in zip(PHI_KEYS, phi_star, strict=True)},
                 "z": z_star.tolist(),
                 "evaluations": n_eval["n"],
                 "de_loss": float(de.fun)},
        "reachability": {"distance_to_nearest": float(dist[j]),
                         "typical_neighbour_distance": typical,
                         "in_spacings": float(dist[j] / typical)},
        "projected": {"loss": float(l_proj), "index": j,
                      "phys": cloud["points"][j]["phys"],
                      "phi": {k: float(v)
                              for k, v in zip(PHI_KEYS, P[j], strict=True)}},
        "best_in_cloud": {"loss": float(losses_cloud[best_dev]), "index": best_dev,
                          "phys": cloud["points"][best_dev]["phys"]},
        "all_cloud_losses": [float(v) for v in losses_cloud],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
