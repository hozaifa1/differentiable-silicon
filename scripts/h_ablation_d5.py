#!/usr/bin/env python
"""V7, the H-ablation: is the transducer's Jacobian carrying physics, or just numbers?

THE QUESTION, and why it is not the same as V2
----------------------------------------------
V2 checks that the composed gradient agrees with a finite difference of the
composed loss. That proves the chain rule is IMPLEMENTED. It does not prove the
middle link means anything: a chain rule assembled from three consistent but
physically arbitrary maps would pass V2 exactly as well.

So replace the middle link with noise and see what breaks.

    J(theta) = F( H( G(theta) ) )
                   ^
                   H : R^7 -> R^5, the DPI transducer. Seven figures of merit a
                       solver measured, turned into the five numbers a spiking
                       network runs on. Closed-form circuit algebra.

J_H = dH/dy is the 5x7 Jacobian of that map. This script replaces it, in the
BACKWARD pass only, with a norm-matched random matrix, and reruns the same
trust-region descent from the same poor corner on the same budget.

WHAT COUNTS AS THE RESULT. Not the loss. A descent direction drawn from a
scrambled-but-self-consistent map can still walk downhill -- four knobs in a box
with a trust region will find *something*, and if the ablated run's loss also
falls, that is expected and it is the honest thing to report. The result is
whether the RECOVERED DEVICE is physically sensible:

    subthreshold slope  SS  should go DOWN (a steeper switch is a better switch)
    memory window       MW  should go UP   (the two conductance states separate)

If a random J_H recovers a device with SS driven up and MW driven down while the
true J_H drives them the right way, then the transducer is steering by physics.
If a random J_H recovers the same physics, it is not, and this project's central
claim is decoration.

THE FORWARD PASS IS NEVER ABLATED. Every loss, every figure of merit and every
phi reported here still comes from the solver and from the exact transducer, at
the design point it is attributed to. Only dH/dy is replaced, and only where the
optimiser reads a direction from it. That keeps the comparison honest: the two
runs are scored by the identical yardstick and differ only in where they walked.

TWO WAYS TO MATCH THE NORM, and the second is the one that matters
------------------------------------------------------------------
`frobenius`  R = G * ||J_H||_F / ||G||_F, G ~ N(0,1). The literal reading of the
             spec. It is also the WEAK ablation, and saying so matters: the rows
             of J_H live on wildly different scales -- dbeta/di_leak against
             dg_min/dg_lo = 1 -- so a single global rescaling puts almost all of
             a random matrix's mass into whichever row was largest and starves
             the rest. Beating that is close to free.

`rowwise`    each row of R is a random direction in R^7 with the LENGTH of the
             corresponding row of J_H. Every phi component therefore responds
             exactly as strongly as it really does, and only the question of
             WHICH figures of merit it responds to is scrambled. This is the
             ablation that is hard to survive, so it is the one to lead with.

A CONTROL arm runs the identical code path with R = J_H itself. If the control
does not reproduce the flagship exactly, the harness is changing the answer and
nothing else here can be trusted.

    python scripts/h_ablation_d5.py
    python scripts/h_ablation_d5.py --modes rowwise --seeds 3
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

os.environ.setdefault("SNN_TRAIN_MODE", "frozen")
os.environ.setdefault("SNN_VJP", "fd")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffsilicon.shared.contract import DIFFERENTIABLE_OUTPUTS  # noqa: E402
from diffsilicon.snn.lif import PHI_KEYS  # noqa: E402

FOM_KEYS = DIFFERENTIABLE_OUTPUTS  # ss, vth_fwd, vth_rev, i_leak, g_lo, g_hi, dg_dvth


def random_matched(JH: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """A random 5x7 matrix matched to `JH` in the requested norm."""
    if mode == "control":
        return JH.copy()
    if mode == "frobenius":
        G = rng.standard_normal(JH.shape)
        n = np.linalg.norm(G)
        return G * (np.linalg.norm(JH) / n if n > 0 else 0.0)
    if mode == "rowwise":
        R = np.zeros_like(JH)
        for i in range(JH.shape[0]):
            g = rng.standard_normal(JH.shape[1])
            gn = np.linalg.norm(g)
            R[i] = g / gn * np.linalg.norm(JH[i]) if gn > 0 else 0.0
        return R
    raise ValueError(f"unknown mode {mode!r}")


def make_ablated_transduce(mode: str, seed: int, stats: dict):
    """A drop-in replacement for `pipeline.transduce_jax` with a scrambled dH/dy.

    Built with `jax.custom_vjp` so that the FORWARD value is the exact
    transducer and only the cotangent is replaced. The direct theta -> phi path
    (the Pelgrom area term uses L_g and W without going through the solver) keeps
    its TRUE derivative: J_H is defined as dH/dy, and ablating anything else
    would be a different experiment.
    """
    from diffsilicon.pipeline import transduce_jax as _true

    rng = np.random.default_rng(seed)

    def _exact(y_vec, theta, cfg):
        y = {k: y_vec[i] for i, k in enumerate(FOM_KEYS)}
        phi = _true(y, theta, cfg)
        return jnp.stack([phi[k] for k in PHI_KEYS])

    def build(cfg):
        @jax.custom_vjp
        def h(y_vec, theta):
            return _exact(y_vec, theta, cfg)

        def h_fwd(y_vec, theta):
            phi = _exact(y_vec, theta, cfg)
            # The true 5x7 dH/dy, and the true 5xD direct dH/dtheta. Both are
            # closed-form algebra on five scalars; differentiating them costs
            # nothing and no solver call.
            JH = np.asarray(jax.jacobian(lambda v: _exact(v, theta, cfg))(y_vec),
                            dtype=np.float64)
            Jt = jax.jacobian(lambda t: _exact(y_vec, t, cfg))(theta)
            R = random_matched(JH, mode, rng)
            stats["n"] = stats.get("n", 0) + 1
            stats["JH_fro"] = float(np.linalg.norm(JH))
            stats["R_fro"] = float(np.linalg.norm(R))
            # How far the scrambled map points from the true one, as an angle.
            # Recorded so the ablation's strength is a measurement, not a label.
            a, b = JH.ravel(), np.asarray(R).ravel()
            den = np.linalg.norm(a) * np.linalg.norm(b)
            stats.setdefault("cosines", []).append(
                float(a @ b / den) if den > 0 else 0.0)
            return phi, (jnp.asarray(R), Jt)

        def h_bwd(res, ct):
            R, Jt = res
            return (R.T @ ct, Jt.T @ ct)

        h.defvjp(h_fwd, h_bwd)
        return h

    cache: dict = {}

    def transduce_ablated(y: dict, theta, cfg):
        if id(cfg) not in cache:
            cache[id(cfg)] = build(cfg)
        h = cache[id(cfg)]
        y_vec = jnp.stack([y[k] for k in FOM_KEYS])
        phi_vec = h(y_vec, theta)
        return {k: phi_vec[i] for i, k in enumerate(PHI_KEYS)}

    return transduce_ablated


def _physics(fom: dict) -> dict:
    """The two numbers the claim is about, plus the conductance ratio."""
    return {
        "ss_mV_per_dec": fom["ss"],
        "memory_window_V": fom["vth_fwd"] - fom["vth_rev"],
        "g_ratio": (fom["g_hi"] / fom["g_lo"]) if fom["g_lo"] > 0 else float("nan"),
    }


def run_one(mode: str, seed: int, args) -> dict:
    """One flagship descent with dH/dy replaced according to `mode`."""
    import diffsilicon.pipeline as pipeline
    import diffsilicon.shim.adjoint as adj
    from diffsilicon.optimise import FlagshipConfig, run_flagship

    # THE REGISTRY MUST BE CLEARED. `shim_for` caches one shim per input
    # template at module scope, so run number two would inherit run number one's
    # Jacobian and its call counter. This is the trap that cost D4 an hour on
    # the race harness; every multi-run driver in this repo has to do it.
    adj._REGISTRY.clear()

    stats: dict = {}
    original = pipeline.transduce_jax
    pipeline.transduce_jax = make_ablated_transduce(mode, seed, stats)
    tag = f"h-ablation-{mode}-seed{seed}"
    t0 = time.time()
    try:
        cfg = FlagshipConfig(
            backend=args.backend, d=4,
            max_oracle_calls=args.budget, max_steps=40,
            theta0=args.theta0, seed=0, batch=16, tag=tag,
            out_dir=str(_REPO / "results" / "runs" / "h_ablation" / tag),
        )
        res = run_flagship(cfg)
        err = None
    except Exception as exc:  # noqa: BLE001 -- a failed arm is a result, not a crash
        res, err = None, f"{type(exc).__name__}: {exc}"
        print(f"    FAILED: {err[:200]}", flush=True)
    finally:
        pipeline.transduce_jax = original

    row = {
        "mode": mode, "seed": seed, "tag": tag,
        "seconds": round(time.time() - t0, 1),
        "error": err,
        "jacobian_rebuilds": stats.get("n", 0),
        "JH_frobenius": stats.get("JH_fro"),
        "R_frobenius": stats.get("R_fro"),
        "cosine_R_vs_JH": (float(np.mean(stats["cosines"]))
                           if stats.get("cosines") else None),
    }
    if res is not None:
        row.update({
            "loss_initial": res["objective_initial"],
            "loss_final": res["objective_final"],
            "ce_initial": res["ce_initial"], "ce_final": res["ce_final"],
            "accuracy_initial": res["accuracy_initial"],
            "accuracy_final": res["accuracy_final"],
            "oracle_calls": res["oracle_calls"],
            "steps": res["steps"], "accepted": res["accepted"],
            "rejected": res["rejected"],
            "theta_final_phys": res["theta_final_phys"],
            "physics_initial": _physics(res["fom_initial"]),
            "physics_final": _physics(res["fom_final"]),
            "phi_final": res["phi_final"],
        })
        pi, pf = row["physics_initial"], row["physics_final"]
        row["d_ss"] = pf["ss_mV_per_dec"] - pi["ss_mV_per_dec"]
        row["d_mw"] = pf["memory_window_V"] - pi["memory_window_V"]
        print(f"    loss {row['loss_initial']:.6f} -> {row['loss_final']:.6f}   "
              f"acc {row['accuracy_initial']:.3f} -> {row['accuracy_final']:.3f}   "
              f"SS {pi['ss_mV_per_dec']:.1f} -> {pf['ss_mV_per_dec']:.1f}   "
              f"MW {pi['memory_window_V']:.3f} -> {pf['memory_window_V']:.3f}   "
              f"({row['oracle_calls']} calls, {row['seconds'] / 60:.1f} min)",
              flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="control,rowwise,frobenius")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--backend", default="devsim")
    ap.add_argument("--theta0", default="0.05,0.80,0.90,0.70")
    ap.add_argument("--out", default=str(_REPO / "results" / "runs"
                                         / "h_ablation_d4.json"))
    args = ap.parse_args()

    os.environ["ORACLE_BACKEND"] = args.backend
    modes = args.modes.split(",")

    print("V7 H-ABLATION -- replace J_H = dH/dy with a norm-matched random matrix")
    print(f"  modes   {modes}")
    print(f"  seeds   {args.seeds} (the control is deterministic; 1 seed)")
    print(f"  budget  {args.budget} solver calls, backend {args.backend}")
    print(f"  start   {args.theta0}")
    print("\nThe result is the RECOVERED DEVICE, not the loss. Watch SS and MW.\n",
          flush=True)

    rows = []
    t_all = time.time()
    out = Path(args.out)
    for mode in modes:
        n = 1 if mode == "control" else args.seeds
        for seed in range(n):
            print(f"  [{mode} seed {seed}] {time.strftime('%H:%M:%S')}", flush=True)
            rows.append(run_one(mode, seed, args))
            out.write_text(json.dumps({
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "backend": args.backend, "budget": args.budget,
                "theta0": args.theta0, "snn_train_mode": "frozen",
                "modes": modes, "seeds": args.seeds,
                "wall_seconds": round(time.time() - t_all, 1),
                "runs": rows,
            }, indent=1), encoding="utf-8")

    # --- the table the claim is read off ------------------------------------
    print(f"\n{'=' * 96}")
    print("RECOVERED DEVICE PHYSICS. SS should FALL and MW should RISE if the "
          "gradient is steering by physics.")
    print(f"{'=' * 96}")
    print(f"{'mode':11s} {'seed':>4s} {'loss ->':>20s} {'acc':>12s} "
          f"{'SS ->':>18s} {'MW ->':>18s} {'calls':>6s}")
    print("-" * 96)
    for r in rows:
        if r.get("error"):
            print(f"{r['mode']:11s} {r['seed']:4d}  FAILED: {r['error'][:60]}")
            continue
        pi, pf = r["physics_initial"], r["physics_final"]
        print(f"{r['mode']:11s} {r['seed']:4d} "
              f"{r['loss_initial']:9.6f}->{r['loss_final']:9.6f} "
              f"{r['accuracy_initial']:5.3f}->{r['accuracy_final']:5.3f} "
              f"{pi['ss_mV_per_dec']:8.2f}->{pf['ss_mV_per_dec']:8.2f} "
              f"{pi['memory_window_V']:8.3f}->{pf['memory_window_V']:8.3f} "
              f"{r['oracle_calls']:6d}")

    ok = [r for r in rows if not r.get("error")]
    for mode in modes:
        v = [r for r in ok if r["mode"] == mode]
        if not v:
            continue
        print(f"\n{mode}: median dSS {np.median([r['d_ss'] for r in v]):+.2f} mV/dec, "
              f"median dMW {np.median([r['d_mw'] for r in v]):+.4f} V, "
              f"median final loss {np.median([r['loss_final'] for r in v]):.6f}")
        if mode != "control" and v[0].get("cosine_R_vs_JH") is not None:
            print(f"  mean cosine between the random R and the true J_H: "
                  f"{np.mean([r['cosine_R_vs_JH'] for r in v]):+.4f}")

    print(f"\ntotal {(time.time() - t_all) / 60:.1f} min")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
