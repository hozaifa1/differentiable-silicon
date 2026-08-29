#!/usr/bin/env python
"""The race: does descending through the solver beat not doing so?

THE CLAIM BEING TESTED. This project drags a TCAD solver through an optimiser
and manufactures a derivative for it. That is only worth doing if it finds a
better device, or the same device for fewer solver calls, than methods that do
not need a derivative at all. Every arm below gets the SAME solver-call budget
and the same starting corner, and the solver call is the unit of cost because it
is the only expensive thing here.

THE ARMS

  gradient      this project. Manufactured Jacobian, trust region.
  random        uniform random search. The honest floor.
  lhs           Latin hypercube. Random done properly -- it covers the box
                evenly instead of clumping, and at these budgets that is a real
                advantage, so it is the fairer floor.
  nelder_mead   derivative-free local search from the same corner. This is the
                arm that matters: it is what a sensible engineer does when the
                simulator has no adjoint, which is the situation this whole
                project exists to address.
  bayes         Gaussian-process expected improvement, warm-started from a small
                Latin-hypercube design. Included only if scikit-learn is
                present, which it is.

Bayesian optimisation is the headline baseline and it is deliberately given the
warm start; a cold-start GP on a four-dimensional box with a few dozen calls is
a strawman and beating it would prove nothing.

Every arm is scored the same way: the BEST loss it has found after N solver
calls. Ties in call count are broken by nothing -- the number is the number.
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
sys.path.insert(0, str(_REPO / "src"))


def _lhs(n, d, rng):
    cut = np.arange(n)[:, None] + rng.random((n, d))
    out = cut / n
    for j in range(d):
        rng.shuffle(out[:, j])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=64,
                    help="solver calls per arm; the flagship used 64")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--theta0", default="0.05,0.80,0.90,0.70")
    ap.add_argument("--backend", default="devsim")
    ap.add_argument("--arms", default="lhs,random,nelder_mead,bayes,gradient")
    ap.add_argument("--tag", default="",
                    help="prefix for the gradient arm's run directory; see below")
    ap.add_argument("--out", default=str(_REPO / "results" / "runs" / "race_d4.json"))
    args = ap.parse_args()

    os.environ["ORACLE_BACKEND"] = args.backend
    os.environ.setdefault("SNN_TRAIN_MODE", "frozen")
    os.environ.setdefault("SNN_VJP", "fd")
    os.environ["SHIM_MAX_ORACLE_CALLS"] = "100000"

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from tesseract_core import Tesseract

    from diffsilicon.pipeline import composed_loss
    from diffsilicon.shared.circuit import load_circuit

    theta0 = np.array([float(v) for v in args.theta0.split(",")])
    D = theta0.size
    cc = load_circuit()
    api = _REPO / "tesseracts" / "{}" / "tesseract_api.py"
    shim_t = Tesseract.from_tesseract_api(str(api).format("adjoint-shim"))
    snn_t = Tesseract.from_tesseract_api(str(api).format("snn-lif-ecg"))

    print(f"budget {args.budget} solver calls per arm, {args.seeds} seeds, "
          f"backend {args.backend}")
    print(f"start  {theta0}\n")

    results = []

    with shim_t, snn_t:
        def raw_loss(th):
            return float(composed_loss(shim_t, snn_t, jnp.asarray(th), cc,
                                       seed=0, batch=16, smooth_spikes=False))

        class Budget(Exception):
            pass

        def make_scored(budget):
            """A loss that counts its own solver calls and stops at the cap.

            A design point the solver cannot handle is scored as the worst seen
            rather than crashing the arm -- every arm meets those points and
            none of them should win or lose on how it handles a crash.
            """
            state = {"n": 0, "best": np.inf, "best_th": None, "trace": []}

            def f(th):
                if state["n"] >= budget:
                    raise Budget()
                th = np.clip(np.asarray(th, dtype=float), 0.0, 1.0)
                state["n"] += 1
                try:
                    v = raw_loss(th)
                except Exception:  # noqa: BLE001 -- see docstring
                    v = state["best"] if np.isfinite(state["best"]) else 10.0
                if v < state["best"]:
                    state["best"], state["best_th"] = v, th.copy()
                state["trace"].append(float(state["best"]))
                return v

            return f, state

        for seed in range(args.seeds):
            for arm in args.arms.split(","):
                f, st = make_scored(args.budget)
                rng = np.random.default_rng(1000 * seed + 7)
                t0 = time.time()
                try:
                    if arm == "random":
                        f(theta0)
                        while True:
                            f(rng.random(D))

                    elif arm == "lhs":
                        f(theta0)
                        for th in _lhs(args.budget, D, rng):
                            f(th)

                    elif arm == "nelder_mead":
                        minimize(f, theta0, method="Nelder-Mead",
                                 options={"maxfev": args.budget * 10,
                                          "xatol": 1e-3, "fatol": 1e-5})

                    elif arm == "bayes":
                        from scipy.stats import norm
                        from sklearn.gaussian_process import GaussianProcessRegressor
                        from sklearn.gaussian_process.kernels import (
                            RBF,
                            ConstantKernel,
                            WhiteKernel,
                        )

                        # Warm start: a small even design, then expected
                        # improvement. A cold GP with a handful of points would
                        # be a strawman.
                        n_init = max(8, args.budget // 4)
                        X = [theta0.copy()]
                        Y = [f(theta0)]
                        for th in _lhs(n_init, D, rng):
                            X.append(np.clip(th, 0, 1))
                            Y.append(f(X[-1]))
                        while True:
                            gp = GaussianProcessRegressor(
                                kernel=ConstantKernel(1.0) * RBF([0.3] * D)
                                + WhiteKernel(1e-4),
                                normalize_y=True, n_restarts_optimizer=2,
                                random_state=seed)
                            gp.fit(np.array(X), np.array(Y))
                            cand = rng.random((2048, D))
                            mu, sd = gp.predict(cand, return_std=True)
                            sd = np.maximum(sd, 1e-9)
                            z = (min(Y) - mu - 0.01) / sd
                            ei = (min(Y) - mu - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
                            nxt = cand[int(np.argmax(ei))]
                            X.append(nxt)
                            Y.append(f(nxt))

                    elif arm == "gradient":
                        # The project's own pipeline, called at the same budget
                        # and from the same corner.
                        #
                        # THE SHIM REGISTRY MUST BE CLEARED FIRST. `shim_for`
                        # caches one shim per input template at module scope, so
                        # a second run in the same process inherits the first
                        # one's Jacobian AND its call counter -- measured: seed 1
                        # stopped after 41 calls at 1.025376 instead of 64 calls
                        # at 1.017666, purely from that carry-over. Every other
                        # arm starts clean, so this one must too.
                        import diffsilicon.shim.adjoint as _adj
                        _adj._REGISTRY.clear()

                        from diffsilicon.optimise import FlagshipConfig, run_flagship
                        cfg = FlagshipConfig(
                            backend=args.backend, d=D,
                            max_oracle_calls=args.budget, max_steps=40,
                            theta0=args.theta0,
                            tag=f"race-gradient-seed{seed}",
                            # The gradient arm has no search randomness: the
                            # start is fixed and the batch seed is fixed, so all
                            # seeds give the same answer. That is a property
                            # worth reporting, not a bug -- the other arms are
                            # being averaged over their own randomness and this
                            # one has none to average.
                            seed=0,
                            # ONE DIRECTORY PER (budget, seed), not one for the
                            # whole race. `run_flagship` APPENDS to steps.jsonl
                            # and OVERWRITES result.json, so a shared directory
                            # silently interleaves every gradient run in the
                            # sweep into one step log and leaves only the last
                            # result.json standing. Harmless while the race was
                            # a single budget; not harmless across five.
                            out_dir=str(
                                _REPO / "results" / "runs" / "race"
                                / f"{args.tag}b{args.budget}-seed{seed}"
                            ),
                        )
                        r = run_flagship(cfg)
                        st["best"] = float(r["objective_final"])
                        st["n"] = int(r["oracle_calls"])
                        st["best_th"] = list(r["theta_final"])
                        st["trace"] = []
                    else:
                        raise ValueError(f"unknown arm {arm!r}")
                except Budget:
                    pass
                except Exception as exc:  # noqa: BLE001
                    print(f"  seed {seed} {arm:12s} FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)

                dt = time.time() - t0
                results.append({"seed": seed, "arm": arm, "best": float(st["best"]),
                                "calls": int(st["n"]),
                                "theta": (None if st["best_th"] is None
                                          else [float(v) for v in st["best_th"]]),
                                "trace": st["trace"], "seconds": round(dt, 1)})
                print(f"  seed {seed} {arm:12s} best {st['best']:.6f} "
                      f"in {st['n']:3d} calls  ({dt / 60:.1f} min)", flush=True)
                Path(args.out).write_text(json.dumps(results, indent=1),
                                          encoding="utf-8")

    print("\n\nSUMMARY -- best loss found within the budget, median over seeds")
    print(f"{'arm':14s} {'median':>10s} {'best':>10s} {'worst':>10s}")
    print("-" * 48)
    for arm in args.arms.split(","):
        v = sorted(r["best"] for r in results if r["arm"] == arm)
        if v:
            print(f"{arm:14s} {v[len(v) // 2]:10.6f} {v[0]:10.6f} {v[-1]:10.6f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
