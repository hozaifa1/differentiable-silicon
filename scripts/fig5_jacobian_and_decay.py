#!/usr/bin/env python
"""Figure 5: the manufactured adjoint -- what it contains, and how long it lasts.

    python scripts/fig5_jacobian_and_decay.py

Figures 1-4 are about the result. This one is about the mechanism, and it is the
figure to have ready when a judge asks the obvious question: *you say the
commercial solver has no adjoint, so what exactly are you differentiating?*

(a) THE JACOBIAN ITSELF. The 7xD matrix the shim holds, at the flagship's
    starting corner, measured by central differences from the solver -- 2D+1 = 9
    calls, all of them already in `results/cache/devsim/`. Shown in RELATIVE
    units, d log(FoM) / d theta, because the seven figures of merit are measured
    in volts, amps, siemens and mV/dec and a raw heatmap of those would be a
    picture of the unit choice rather than of the physics.

    Read it as: which fabrication knob moves which device property, and by how
    much, at the point where the descent starts.

(b) HOW LONG IT LASTS. The cosine between the Broyden-patched Jacobian and the
    true one, against steps since the ground-truth anchor, from
    `scripts/v2_broyden_decay_d6.py`. This is the curve behind
    `ShimConfig.refresh_every = 4` -- the whole economic argument for the
    apparatus is that the rank-one patch is free and the refresh is not, so the
    only question that matters is how many free steps you get before the
    direction stops being the solver's direction.

    Two curves, because they are two questions: the mean over the seven Jacobian
    rows (the shim's own accuracy) and the cosine of the COMPOSED gradient (what
    the optimiser actually steps along). The second can stay high while the first
    decays, if the rows that decay are ones the loss does not care about.

    AND THE CURVE IS NOT MONOTONE, WHICH IS THE INTERESTING PART. It decays
    through k=5 and then RECOVERS. The grey bars say why: the model goes stale
    with DISTANCE TRAVELLED, not with steps taken. The flagship's trust region
    grows to 0.33 by k=4-5, which is where the cosine is worst, then collapses to
    0.08 and 0.02 -- and a rank-one patch over a step that short barely has to be
    right. Over these nine points, distance from the anchor predicts the row
    cosine (r = -0.79) about twice as well as step count does (r = -0.36 against
    the composed gradient).

    That is a justification for the design rather than an embarrassment for it.
    `AdjointShim.jacobian` refreshes when `steps_since_refresh >= refresh_every`
    OR when `||theta - theta_anchor|| > radius`, and this says the second test is
    the one carrying the load. K = 4 is a proxy for a distance budget, and it is
    a proxy that holds only while the step size holds.

Both panels come from banked measurements and neither calls a solver.
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

os.environ.setdefault("ORACLE_BACKEND", "replay")
os.environ.setdefault("ORACLE_REPLAY_SOURCE", "devsim")
os.environ.setdefault("DIFFSILICON_PROVENANCE_DISABLE", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

from diffsilicon.shared.contract import DIFFERENTIABLE_OUTPUTS, make_oracle_input  # noqa: E402
from diffsilicon.shared.design import get_design  # noqa: E402
from diffsilicon.shim.adjoint import ShimConfig, fd_jacobian  # noqa: E402

RESULT = _REPO / "results" / "runs" / "flagship-d4-fixed" / "result.json"
DECAY = _REPO / "results" / "runs" / "v2_broyden_decay_d6.json"
OUT = _REPO / "docs" / "figures" / "fig5_jacobian_and_decay.png"

FOM_LABEL = {
    "ss": "subthreshold slope",
    "vth_fwd": "forward threshold",
    "vth_rev": "reverse threshold",
    "i_leak": "leakage current",
    "g_lo": "low conductance",
    "g_hi": "high conductance",
    "dg_dvth": "conductance slope",
}
KNOB_LABEL = {"t_fe": "$t_{fe}$", "L_g": "$L_g$",
              "log10_N_ch": "$\\log_{10}N_{ch}$", "t_IL": "$t_{IL}$"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    flag = json.loads(RESULT.read_text(encoding="utf-8"))
    theta = np.asarray(flag["theta_initial"], dtype=np.float64)
    spec = get_design(int(theta.size))

    cfg = ShimConfig(alpha=args.alpha, backend="replay", max_oracle_calls=10_000)
    j, y = fd_jacobian(theta, make_oracle_input(theta), cfg, central=True)
    # Relative: d log(FoM) / d theta. The seven FoMs are in four different units.
    scale = np.where(np.abs(y) > 0, np.abs(y), 1.0)[:, None]
    rel = j / scale

    print(f"Jacobian at the flagship start, theta = {list(theta)}")
    print(f"{'FoM':<20s}" + "".join(f"{KNOB_LABEL[n].strip('$'):>14s}"
                                    for n in spec.names))
    for i, name in enumerate(DIFFERENTIABLE_OUTPUTS):
        print(f"{FOM_LABEL[name]:<20s}" +
              "".join(f"{rel[i, k]:>14.4f}" for k in range(len(spec.names))))

    decay = None
    if DECAY.is_file():
        decay = json.loads(DECAY.read_text(encoding="utf-8"))
        print(f"\ndecay curve: {len(decay['rows'])} points from "
              f"{DECAY.name} ({decay['probes_attempted']} probes attempted, "
              f"{decay['probes_from_cache']} of {decay['probes_required']} "
              f"from cache, {len(decay['probes_refused'])} salvaged)")
    else:
        print(f"\n{DECAY.name} not found -- panel (b) will say so rather than "
              f"be omitted silently.")

    # --- figure ---------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.8, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "figure.dpi": 130, "savefig.dpi": 300, "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(11.4, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.30,
                          left=0.115, right=0.975, bottom=0.235, top=0.845)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) the heatmap ---------------------------------------------------------
    lim = float(np.nanmax(np.abs(rel)))
    # SYMMETRIC LOG, not linear. Two rows -- leakage current and low conductance
    # -- respond about eleven times more strongly in relative terms than any
    # other, because both are exponential in the surface potential. On a linear
    # scale those two rows saturate the colormap and every other entry reads as
    # white, i.e. the figure would say "only two things happen", which is the
    # opposite of what it is for. linthresh = 0.5 keeps the O(1) entries
    # coloured while the exponential rows stay visibly larger.
    norm = SymLogNorm(linthresh=0.5, vmin=-lim, vmax=lim, base=10)
    im = axA.imshow(rel, cmap="RdBu_r", norm=norm, aspect="auto")
    axA.set_xticks(range(len(spec.names)))
    axA.set_xticklabels([KNOB_LABEL[n] for n in spec.names], fontsize=10)
    axA.set_yticks(range(len(DIFFERENTIABLE_OUTPUTS)))
    axA.set_yticklabels([FOM_LABEL[n] for n in DIFFERENTIABLE_OUTPUTS], fontsize=8.4)
    for i in range(rel.shape[0]):
        for k in range(rel.shape[1]):
            v = rel[i, k]
            axA.text(k, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.4,
                     color="white" if abs(v) > 3.0 else "0.15")
    axA.set_title("(a)  the manufactured Jacobian, $d\\log(\\mathrm{FoM})/d\\theta$",
                  loc="left", fontsize=9.6)
    axA.set_xlabel("fabrication knob (normalised design box)")
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.6)

    # (b) the decay -----------------------------------------------------------
    if decay is not None:
        k = np.array([r["steps_since_refresh"] for r in decay["rows"]])
        cj = np.array([r["cos_J_rowmean"] for r in decay["rows"]])
        cg = np.array([r["cos_gradient"] for r in decay["rows"]])
        th = np.array([r["theta"] for r in decay["rows"]], dtype=np.float64)
        step = np.concatenate([[0.0], np.linalg.norm(np.diff(th, axis=0), axis=1)])

        # The step size first and behind everything: it is the explanation, not
        # the result, so it should read as background.
        axS = axB.twinx()
        axS.bar(k, step, width=0.62, color="0.86", edgecolor="0.72",
                linewidth=0.5, zorder=1)
        axS.set_ylabel("trust-region step $\\|\\Delta\\theta\\|$", fontsize=8.4,
                       color="0.45")
        axS.tick_params(axis="y", labelsize=7.5, colors="0.45")
        axS.set_ylim(0, float(step.max()) * 3.1)
        axS.spines["right"].set_color("0.7")

        kk = int(decay.get("refresh_every_in_use", 4))
        axB.axvspan(-0.4, kk, color="#eaf3ea", zorder=0)
        axB.plot(k, cj, "-o", ms=5.5, lw=1.8, color="#2166ac", zorder=6,
                 label="mean over the seven Jacobian rows")
        axB.plot(k, cg, "-s", ms=5.0, lw=1.8, color="#1b7837", zorder=7,
                 label="the composed gradient $dL/d\\theta$")
        axB.axvline(kk, color="#b2182b", ls=(0, (4, 2)), lw=1.2, zorder=5)
        axB.set_ylim(0.0, 1.06)
        axB.text(kk - 0.10, 1.035, f"refresh_every = {kk}", fontsize=7.8,
                 color="#b2182b", va="top", ha="right", zorder=8)
        # The two numbers the constant rests on.
        i4 = int(np.where(k == kk)[0][0]) if kk in k else None
        i5 = int(np.where(k == kk + 1)[0][0]) if (kk + 1) in k else None
        if i4 is not None and i5 is not None:
            axB.annotate(f"{cg[i4]:.2f}", (k[i4], cg[i4]), textcoords="offset points",
                         xytext=(-6, -13), fontsize=7.8, color="#1b7837", ha="right")
            axB.annotate(f"{cg[i5]:.2f}", (k[i5], cg[i5]), textcoords="offset points",
                         xytext=(3, -12), fontsize=7.8, color="#1b7837", ha="left")
        axB.set_xlabel("steps since the ground-truth anchor")
        axB.set_ylabel("cosine against the true Jacobian")
        axB.set_xlim(-0.4, k.max() + 0.4)
        axB.set_xticks(k)
        axB.set_zorder(axS.get_zorder() + 1)
        axB.patch.set_visible(False)
        axB.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=7.6)
        axB.grid(axis="y", lw=0.4, color="0.9", zorder=2)
    else:
        axB.text(0.5, 0.5, f"{DECAY.name}\nnot present -- run\n"
                           "scripts/v2_broyden_decay_d6.py",
                 ha="center", va="center", fontsize=9, color="0.45",
                 transform=axB.transAxes)
        axB.set_xticks([])
        axB.set_yticks([])
    axB.set_title("(b)  the free patch goes stale with DISTANCE, not with steps",
                  loc="left", fontsize=9.6)

    fig.text(0.115, 0.075,
             "(a) is 2D+1 = 9 central-difference probes of DEVSIM at the "
             "flagship's starting corner, all served from results/cache/devsim/. "
             "Relative units, because the seven\nfigures of merit are measured in "
             "volts, amps, siemens and mV/dec. (b) is measured along the "
             "flagship's own accepted path: anchor a true Jacobian, patch it by "
             "Broyden from the\nsecant pair each step supplies free, and rebuild "
             "the true one alongside. The grey bars are the trust-region step "
             "length, and they are why the curve recovers after k=6 -- a "
             "rank-one patch over a\nstep of 0.02 barely has to be right. "
             "Panel (a) calls no solver; panel (b) cost 11 DEVSIM probes on top "
             "of the 70 already cached.",
             fontsize=7.4, color="0.3", va="center", linespacing=1.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}\nwrote {pdf}")

    bank = _REPO / "results" / "runs" / "fig5_jacobian_and_decay.json"
    bank.write_text(json.dumps({
        "source": "results/runs/flagship-d4-fixed/result.json + "
                  "results/runs/v2_broyden_decay_d6.json",
        "backend": "replay(devsim)", "solver_calls_made": 0,
        "theta": [float(v) for v in theta], "alpha": args.alpha,
        "fom_order": list(DIFFERENTIABLE_OUTPUTS),
        "knob_order": list(spec.names),
        "jacobian_relative": [[float(v) for v in row] for row in rel],
        "y_at_theta": [float(v) for v in y],
        "decay_present": decay is not None,
    }, indent=2), encoding="utf-8")
    print(f"wrote {bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
