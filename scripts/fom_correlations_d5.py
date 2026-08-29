#!/usr/bin/env python
"""Does any single figure of merit predict how well the device classifies?

WHY THIS QUESTION DECIDES WHETHER THE PROJECT IS NEEDED. If one scalar -- the
memory window, say, or the subthreshold slope -- ordered the devices by
performance, then the right thing to do is maximise that scalar and skip the
whole apparatus. No adjoint, no shim, no network in the loop. So this is not a
supporting measurement; it is the premise.

D3 answered it over EIGHT design points and one seed, which is not enough to
carry the claim, and D4 then invalidated even that: the two circuit trims and the
switch to a frozen network changed every loss number in the project. Pre-trim and
post-trim numbers cannot share a table, and the D6 checklist says so explicitly.

This re-measures it on the CURRENT objective over the 192-device manifold cloud,
whose losses were scored by `scripts/v6_free_refit_d5.py` under the frozen
network. Zero solver calls: both halves are already banked.

Reported for each figure of merit:

  Pearson r   linear correlation with the balanced cross-entropy.
  Spearman    rank correlation, which is the honest one here -- the question is
              whether the scalar ORDERS the devices, and several of these span
              decades, where a linear fit is a statement about outliers.
  R^2 alone   how much of the loss variation that one number explains by itself.

and then, because "no single one works" invites the obvious follow-up, what a
linear model on ALL SEVEN together explains -- the fair upper bound on any
one-shot "just predict the loss from the figures of merit" shortcut.

    python scripts/fom_correlations_d5.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

FOM_KEYS = ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth")
PHI_KEYS = ("beta", "g_min", "g_max", "th_th", "sig_w")
LABEL = {
    "ss": "subthreshold slope", "vth_fwd": "forward threshold",
    "vth_rev": "reverse threshold", "i_leak": "leakage current",
    "g_lo": "low conductance", "g_hi": "high conductance",
    "dg_dvth": "conductance slope", "mw": "MEMORY WINDOW",
    "g_ratio": "conductance ratio",
    "beta": "membrane decay", "g_min": "min conductance",
    "g_max": "max conductance", "th_th": "firing threshold",
    "sig_w": "weight noise",
}
# Quantities that live over decades. Ranking is unaffected; the linear
# correlation of a decades-wide quantity is otherwise a statement about its
# largest few points.
LOGGED = {"i_leak", "g_lo", "g_hi", "dg_dvth", "g_min", "g_max", "g_ratio"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=str(_REPO / "results" / "runs"
                                           / "manifold_cloud_d4.json"))
    ap.add_argument("--losses", default=str(_REPO / "results" / "runs"
                                            / "v6_manifold_control_d5.json"))
    ap.add_argument("--out", default=str(_REPO / "results" / "runs"
                                         / "fom_correlations_d5.json"))
    args = ap.parse_args()

    from scipy.stats import pearsonr, spearmanr

    cloud = json.loads(Path(args.cloud).read_text(encoding="utf-8"))
    ctrl = json.loads(Path(args.losses).read_text(encoding="utf-8"))
    L = np.asarray(ctrl["all_cloud_losses"], dtype=float)
    pts = cloud["points"]
    if len(pts) != len(L):
        raise SystemExit(f"{len(pts)} devices but {len(L)} losses")

    cols: dict[str, np.ndarray] = {}
    for k in FOM_KEYS:
        cols[k] = np.array([p["y"][k] for p in pts], dtype=float)
    cols["mw"] = cols["vth_fwd"] - cols["vth_rev"]
    cols["g_ratio"] = cols["g_hi"] / np.where(cols["g_lo"] > 0, cols["g_lo"], np.nan)
    for k in PHI_KEYS:
        cols[k] = np.array([p["phi"][k] for p in pts], dtype=float)

    order = [*FOM_KEYS, "mw", "g_ratio", *PHI_KEYS]
    print(f"{len(L)} devices, balanced cross-entropy under the FROZEN network")
    print(f"loss spans {L.min():.4f} to {L.max():.4f}\n")
    print(f"{'quantity':22s} {'Pearson r':>10s} {'R^2':>7s} {'Spearman':>10s} "
          f"{'log?':>5s}")
    print("-" * 60)

    rows = []
    for k in order:
        v = cols[k].copy()
        logged = k in LOGGED
        if logged:
            v = np.log10(np.where(v > 0, v, np.nan))
        ok = np.isfinite(v) & np.isfinite(L)
        r = float(pearsonr(v[ok], L[ok])[0])
        rho = float(spearmanr(v[ok], L[ok])[0])
        rows.append({"quantity": k, "label": LABEL[k], "pearson_r": r,
                     "r2": r * r, "spearman": rho, "log10": logged,
                     "n": int(ok.sum())})
        sep = "  <- device -> network" if k == "beta" else ""
        print(f"{LABEL[k]:22s} {r:+10.3f} {r * r:7.3f} {rho:+10.3f} "
              f"{'log10' if logged else '':>5s}{sep}")

    best = max(rows, key=lambda d: abs(d["pearson_r"]))
    print(f"\nBest single predictor: {best['label']} "
          f"(r = {best['pearson_r']:+.3f}, R^2 = {best['r2']:.3f}) -- it explains "
          f"{best['r2'] * 100:.0f}% of the variation on its own.")

    # --- the fair upper bound: all seven together ---------------------------
    X = np.column_stack([
        np.log10(np.abs(cols[k]) + 1e-300) if k in LOGGED else cols[k]
        for k in FOM_KEYS])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    resid = L - A @ coef
    r2_all = 1.0 - float(resid @ resid) / float(((L - L.mean()) ** 2).sum())
    print(f"\nAll seven figures of merit in one linear model: R^2 = {r2_all:.3f}.")
    print("That is the ceiling on any 'skip the network, predict the loss from")
    print("the device numbers' shortcut, and it is fitted on the same 192 points")
    print("it is scored on -- so it is generous to the shortcut, not to us.")

    payload = {
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "n_devices": int(len(L)), "objective": "balanced CE, SNN_TRAIN_MODE=frozen",
        "cloud": Path(args.cloud).name, "losses_from": Path(args.losses).name,
        "loss_min": float(L.min()), "loss_max": float(L.max()),
        "correlations": rows,
        "best_single": best,
        "r2_all_seven_foms_linear": r2_all,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
