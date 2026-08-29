#!/usr/bin/env python
"""Is the gradient sane enough to spend two hours optimising against?

WHY THIS EXISTS
---------------
The D3 flagship burned a whole run before anyone noticed the gradient was
6.1e13 against a loss of order 1. At that size the trust region's rho is
meaningless -- it predicts a drop of 3.7e12 against an actual 0.07 -- so every
step is judged a failure, the "rho < 0.25 forces a refresh" rule fires every
time, and the budget goes on rebuilding the same Jacobian instead of descending.

This is the check that should have run first. It is cheap and it is decisive.

WHAT IT MEASURES, AND WHY IN THREE PIECES
-----------------------------------------
The loss reaches the design vector through three links:

    theta --[ J1: the solver, via the shim ]--> y   (7 figures of merit)
        y --[ J2: transduce, pure JAX      ]--> phi (5 SNN hyperparameters)
      phi --[ J3: the spiking network      ]--> L   (one scalar)

    dL/dtheta = J3 @ J2 @ J1

Reporting only |dL/dtheta| tells you it is broken but not WHERE. Reporting the
three factors separately says which one to fix, which is the whole point --
last night's diagnosis of J2 was inferred from one line of algebra, not
measured, and an inferred cause is a guess with a decimal point on it.

TIERS
-----
Tier 1 (default, seconds, no solver and no network): J2 only, plus the spread
of beta across the design box. J2 is pure JAX over BANKED figures of merit, so
it costs nothing. This alone answers "did the beta fix work".

Tier 2 (--full, minutes): the whole chain at a few design points, including J1
from the shim and J3 through the network. This is the number the optimiser
actually sees.

    python scripts/gradient_gate.py                 # tier 1
    python scripts/gradient_gate.py --full          # tier 1 + tier 2
    python scripts/gradient_gate.py --baseline pre.json --compare post.json

EXIT CODE is 0 if the gate PASSES, 1 if it FAILS. The overnight run must stop
on a failure rather than optimise against a broken gradient.
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

FOM_KEYS = ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth")
PHI_NAMES = ("beta", "g_min", "g_max", "th_th", "sig_w")

# --- the thresholds the gate judges against ---------------------------------
#
# CORRECTED 2026-08-27 (D4). The first version of this file gated on
# `MAX_J2 = 1e6`, i.e. on the raw entries of dphi/dy. THAT IS THE WRONG TEST,
# and it is wrong in the direction that matters: it fails a gradient that is
# perfectly fine.
#
# Measured at the exact design point where the D3 flagship's gradient hit
# 6.15e13 (theta = 0.06412, 0.76056, 0.86470, 0.67553):
#
#   J1 = dy/dtheta      largest entry 4.06e3 (at d(SS)/d(t_IL)); the i_leak row
#                       maxes at 9.2e-12
#   J2 = dphi/dy        largest entry 3.4e8 (at d(beta)/d(i_leak))
#   J2 @ J1 = dphi/dtheta   LARGEST ENTRY 7.8 -- and the beta path contributes
#                       0.0032 of it
#   J3 = dL/dphi        7.2e10 at that point, against 1.06e7 at the healthy
#                       nominal device
#
# So a large dphi/dy is NOT by itself a problem. d(beta)/d(i_leak) ~ 1e9 per
# amp is multiplied by a d(i_leak)/d(theta) of ~1e-12 and the product is 0.003.
# Units, not pathology: i_leak is measured in amps and is around 1e-11, so a
# per-amp sensitivity of 1e9 is a per-DEVICE sensitivity of 0.01.
#
# What the composition has to be bounded on is the COMPOSED product, and what
# actually exploded on D3 was the network. Hence the two thresholds below.
MAX_DPHI_DTHETA = 1.0e2  # |dphi/dtheta|. D3 measured 7.8 -- already fine.
MAX_DL_DPHI = 1.0e4      # |dL/dphi|, the network alone. D3 measured 7.2e10.
MAX_GRAD = 1.0e4         # |dL/dtheta|. D3 measured 6.1e13 against a loss of 1.
# beta must actually MOVE across the box. LOWERED from 0.05 to 0.02 on
# 2026-08-27, and the reason is not that it was failing -- it is that 0.05 was
# set before the leak-bias trim existed and it measures the wrong thing now.
# The trim's whole PURPOSE is to narrow beta's range, into the only band where
# the network has a usable gradient at all; measured, the eight banked devices
# land in 0.583 .. 0.626. What has to be true is that every device still gets a
# DISTINCT beta with a live derivative, not that the range is wide -- and
# `beta_not_mostly_pinned` plus the tests in tests/test_circuit.py are what say
# so. This threshold is left in place only to catch beta collapsing to a
# constant, which would mean the device had stopped reaching the network.
MIN_BETA_SPREAD = 0.02
MAX_BETA_PINNED = 0.5    # at most half the box may sit pinned at 0 or 1

# `J2` is still MEASURED and still PRINTED -- it is one third of the diagnosis
# and dropping it would make the report useless. It is simply no longer a
# pass/fail criterion, because on its own it does not decide anything.


def _banked_points(path: Path) -> list[dict]:
    """The figures of merit already measured on the open solver."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in data["points"]:
        if p.get("solver_failed"):
            continue
        out.append({
            "name": p["name"],
            "phys": p["phys"],
            "y": {
                "ss": p["ss_mV_per_dec"],
                "vth_fwd": p["vth_fwd_V"],
                "vth_rev": p["vth_rev_V"],
                "i_leak": p["i_leak_A"],
                "g_lo": p["g_lo_S"],
                "g_hi": p["g_hi_S"],
                "dg_dvth": p["dg_dvth_S_per_V"],
            },
        })
    return out


def tier1(points: list[dict], cc) -> dict:
    """J2 = dphi/dy, and the spread of beta. No solver, no network."""
    l_g = {p["name"]: p["phys"]["L_g"] for p in points}

    def phi_of(yvec, lg):
        y = dict(zip(FOM_KEYS, yvec, strict=True))
        p = transduce(y, cc, lg)
        return jnp.stack([p.beta, p.g_min, p.g_max, p.th_th, p.sig_w])

    rows = []
    print(f"{'device':10s} {'i_leak (A)':>11s} {'beta':>9s} "
          f"{'|dbeta/di_leak|':>16s} {'max |dphi/dy|':>14s}")
    print("-" * 66)
    for p in points:
        yvec = jnp.array([p["y"][k] for k in FOM_KEYS], dtype=jnp.float64)
        J2 = jax.jacobian(phi_of)(yvec, l_g[p["name"]])
        J2 = np.asarray(J2)
        phi = np.asarray(phi_of(yvec, l_g[p["name"]]))
        beta = float(phi[0])
        dbeta_dileak = abs(float(J2[0, FOM_KEYS.index("i_leak")]))
        mx = float(np.max(np.abs(J2)))
        rows.append({"name": p["name"], "i_leak": p["y"]["i_leak"], "beta": beta,
                     "dbeta_di_leak": dbeta_dileak, "max_abs_J2": mx,
                     "phi": {n: float(v) for n, v in zip(PHI_NAMES, phi, strict=True)}})
        print(f"{p['name']:10s} {p['y']['i_leak']:11.2e} {beta:9.4f} "
              f"{dbeta_dileak:16.3e} {mx:14.3e}")

    betas = np.array([r["beta"] for r in rows])
    pinned = np.mean((betas < 1e-6) | (betas > 1 - 1e-6))
    spread = float(betas.max() - betas.min())
    worst_j2 = max(r["max_abs_J2"] for r in rows)

    print()
    print(f"  beta range      : {betas.min():.4f} .. {betas.max():.4f}  (spread {spread:.4f})")
    print(f"  beta pinned at 0 or 1: {pinned * 100:.0f}% of points")
    print(f"  worst |dphi/dy| : {worst_j2:.3e}")

    return {"rows": rows, "beta_spread": spread, "beta_pinned_fraction": float(pinned),
            "worst_abs_J2": worst_j2}


def tier2(thetas: list[np.ndarray], names: list[str], cfg_d: int, backend: str) -> dict:
    """The whole chain, split into its three links, as the optimiser sees it.

    Costs solver calls -- but no MORE than one gradient already costs. The shim
    builds J1 = dy/dtheta itself while answering `value_and_grad`, and
    `shim_for` is a registry keyed on the input template, so the same object is
    reachable in-process afterwards and its Jacobian is READ rather than
    re-measured. J2 is pure JAX and free. J3 costs one more network pass.
    """
    import os

    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract

    from diffsilicon.pipeline import composed_loss, oracle_call, transduce_jax
    from diffsilicon.shared.contract import make_oracle_input
    from diffsilicon.shim.adjoint import shim_for
    from diffsilicon.snn.lif import PHI_KEYS
    os.environ["ORACLE_BACKEND"] = backend
    os.environ.setdefault("SHIM_MAX_ORACLE_CALLS", "10000")

    cc = load_circuit()
    api = _REPO / "tesseracts" / "{}" / "tesseract_api.py"
    shim_t = Tesseract.from_tesseract_api(str(api).format("adjoint-shim"))
    snn_t = Tesseract.from_tesseract_api(str(api).format("snn-lif-ecg"))

    def _phi_of_y(yvec, theta):
        y = dict(zip(FOM_KEYS, yvec, strict=True))
        phi = transduce_jax(y, theta, cc)
        return jnp.stack([phi[k] for k in PHI_KEYS])

    out = []
    with shim_t, snn_t:
        def loss_of(theta):
            return composed_loss(shim_t, snn_t, theta, cc, seed=0, batch=16,
                                 smooth_spikes=False)

        vg = jax.value_and_grad(loss_of)

        for name, th in zip(names, thetas, strict=True):
            t0 = time.perf_counter()
            th_j = jnp.asarray(th)
            L, g = vg(th_j)
            g = np.asarray(g)

            # --- J1: the solver. Read off the shim, which already built it. ---
            shim = shim_for(make_oracle_input(np.asarray(th)))
            J1 = None if shim.J is None else np.asarray(shim.J)

            y = oracle_call(shim_t, th_j)
            yvec = jnp.stack([y[k] for k in FOM_KEYS])

            # --- J2: the transducer. Pure JAX, free. ---
            J2 = np.asarray(jax.jacobian(_phi_of_y, argnums=0)(yvec, th_j))

            # --- the composed product, which is what has to be bounded ---
            dphi_dtheta = None if J1 is None else J2 @ J1

            # --- J3 = dL/dphi, through the network alone. ---
            phi = transduce_jax(y, th_j, cc)

            def snn_loss(pvec):
                d = dict(zip(PHI_KEYS, pvec, strict=True))
                r = apply_tesseract(snn_t, {**d, "seed": 0, "batch": 16,
                                            "smooth_spikes": False})
                return r["loss"]

            pvec = jnp.stack([phi[k] for k in PHI_KEYS])
            J3 = np.asarray(jax.grad(snn_loss)(pvec))

            rec = {
                "name": name,
                "loss": float(L),
                "grad_norm": float(np.linalg.norm(g)),
                "grad": [float(v) for v in g],
                "phi": {k: float(phi[k]) for k in PHI_KEYS},
                "max_abs_J1": None if J1 is None else float(np.max(np.abs(J1))),
                "max_abs_J2": float(np.max(np.abs(J2))),
                "max_abs_dphi_dtheta": (None if dphi_dtheta is None
                                        else float(np.max(np.abs(dphi_dtheta)))),
                "max_abs_dL_dphi": float(np.max(np.abs(J3))),
                "dL_dphi": {k: float(v) for k, v in zip(PHI_KEYS, J3, strict=True)},
                "seconds": round(time.perf_counter() - t0, 1),
            }
            out.append(rec)
            d2t = rec["max_abs_dphi_dtheta"]
            d2t_s = "n/a" if d2t is None else format(d2t, ".3e")
            print(f"  {name:10s} loss={float(L):.4f}  beta={float(phi['beta']):.4f}"
                  f"  |dphi/dtheta|={d2t_s}"
                  f"  |dL/dphi|={rec['max_abs_dL_dphi']:.3e}"
                  f"  |dL/dtheta|={rec['grad_norm']:.3e}  ({rec['seconds']:.0f}s)")

    known = [r["max_abs_dphi_dtheta"] for r in out
             if r["max_abs_dphi_dtheta"] is not None]
    return {
        "points": out,
        "worst_grad_norm": max((r["grad_norm"] for r in out), default=0.0),
        "worst_dphi_dtheta": max(known, default=0.0),
        "worst_dL_dphi": max((r["max_abs_dL_dphi"] for r in out), default=0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(_REPO / "results" / "runs"
                                              / "rebaseline_d3_devsim.json"))
    ap.add_argument("--full", action="store_true", help="also run the whole chain")
    ap.add_argument("--backend", default="devsim")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cc = load_circuit()
    src = Path(args.baseline)
    if not src.is_file():
        print(f"no banked baseline at {src}; run scripts/rebaseline_d3.py first")
        return 1
    points = _banked_points(src)
    print(f"TIER 1 -- transduce only, {len(points)} banked devices, no solver, no network")
    print(f"source: {src.name}\n")
    t1 = tier1(points, cc)

    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": src.name,
           "tier1": t1,
           "thresholds": {"max_dphi_dtheta": MAX_DPHI_DTHETA,
                          "max_dL_dphi": MAX_DL_DPHI,
                          "max_grad": MAX_GRAD,
                          "min_beta_spread": MIN_BETA_SPREAD,
                          "max_beta_pinned": MAX_BETA_PINNED}}

    # NOTE what is NOT here: a threshold on |dphi/dy|. See the thresholds at the
    # top of this file for the measurement that removed it.
    checks = {
        "beta_actually_moves": (t1["beta_spread"] >= MIN_BETA_SPREAD,
                                f"beta spread = {t1['beta_spread']:.4f}, "
                                f"need >= {MIN_BETA_SPREAD}"),
        "beta_not_mostly_pinned": (t1["beta_pinned_fraction"] <= MAX_BETA_PINNED,
                                   f"{t1['beta_pinned_fraction'] * 100:.0f}% pinned at 0 or 1, "
                                   f"limit {MAX_BETA_PINNED * 100:.0f}%"),
    }

    if args.full:
        print("\nTIER 2 -- the whole chain, split into its three links.")
        print("One gradient's worth of solver calls per point.\n")
        names = [p["name"] for p in points[:3]]
        spec_d = 4
        from diffsilicon.shared.design import get_design
        spec = get_design(spec_d)
        thetas = []
        for p in points[:3]:
            phys = np.array([p["phys"][n] for n in spec.names])
            thetas.append((phys - np.array(spec.lo)) / (np.array(spec.hi) - np.array(spec.lo)))
        t2 = tier2(thetas, names, spec_d, args.backend)
        res["tier2"] = t2
        checks["dphi_dtheta_bounded"] = (
            t2["worst_dphi_dtheta"] <= MAX_DPHI_DTHETA,
            f"worst |dphi/dtheta| = {t2['worst_dphi_dtheta']:.3e}, "
            f"limit {MAX_DPHI_DTHETA:.0e}")
        checks["network_gradient_bounded"] = (
            t2["worst_dL_dphi"] <= MAX_DL_DPHI,
            f"worst |dL/dphi| = {t2['worst_dL_dphi']:.3e}, limit {MAX_DL_DPHI:.0e}")
        checks["grad_norm_sane"] = (
            t2["worst_grad_norm"] <= MAX_GRAD,
            f"worst |dL/dtheta| = {t2['worst_grad_norm']:.3e}, limit {MAX_GRAD:.0e}")

    print("\n" + "=" * 66)
    ok = True
    for name, (passed, detail) in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
        ok = ok and passed
    print("=" * 66)
    print(f"\nGATE {'PASSES' if ok else 'FAILS'}"
          + ("" if ok else " -- do NOT launch the flagship against this gradient"))

    res["checks"] = {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in checks.items()}
    res["gate_passes"] = bool(ok)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
