#!/usr/bin/env python
"""Figure 2: the sample-efficiency curve, and the budget below which we lose.

WHY THIS IS THE HEADLINE FIGURE. D4 measured that joint descent beats the
projected baseline by less than the 5% the gate asked for, and the instruction
attached to that gate was to move the headline to sample-efficiency. This is that
headline: how well each method does as a function of the only cost that matters,
the number of times it is allowed to call the solver.

WHAT IT MUST NOT DO is show only the budget where we win. A derivative has to be
paid for before it can be used -- the anchor Jacobian costs 2D+1 = 9 calls on
this four-knob problem, before a single step is taken -- so at a small budget
gradient descent is spending most of its money on a direction it barely gets to
use. That is real, it is not a defect, and the figure has to show it. Which is
why the x-axis starts at 12 calls, where this project comes LAST but one.

The three things to read off it:

  1. Gradient descent is the ONLY arm whose score keeps improving as the budget
     grows. Uniform random search is flat to six decimal places from 12 calls to
     48 -- the best point in its first twelve draws is still the best after
     forty-eight. Extra budget buys the derivative-free arms almost nothing on
     this objective; it buys the gradient arm a great deal.

  2. It has no spread. Gradient descent and Nelder-Mead start from a fixed corner
     and follow a deterministic rule, so all three seeds give the same answer to
     six decimal places. The bands on the other arms are them being averaged over
     their own luck. That is a property worth reporting, not a defect.

  3. The crossovers, which happen at different budgets against different
     baselines, and should be quoted that way rather than as one number.

    python scripts/fig2_budget_crossover.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "gradient": dict(color="#1b7837", lw=2.6, marker="o", ms=7, zorder=6,
                     label="gradient descent through the solver (this project)"),
    "bayes": dict(color="#762a83", lw=1.7, marker="s", ms=5.5, zorder=5,
                  label="Bayesian optimisation, warm-started"),
    "lhs": dict(color="#2166ac", lw=1.7, marker="^", ms=5.5, zorder=4,
                label="Latin hypercube"),
    "random": dict(color="#e08214", lw=1.7, marker="v", ms=5.5, zorder=3,
                   label="random search"),
    "nelder_mead": dict(color="#b2182b", lw=1.7, marker="D", ms=5.0, zorder=2,
                        label="Nelder-Mead (derivative-free local search)"),
}
ORDER = ("gradient", "bayes", "lhs", "random", "nelder_mead")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(_REPO / "results" / "runs"
                                           / "race_crossover_sweep.json"))
    ap.add_argument("--out", default=str(_REPO / "docs" / "figures"
                                         / "fig2_budget_crossover.png"))
    args = ap.parse_args()

    d = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    runs = d["runs"]
    budgets = sorted({r["budget"] for r in runs})
    arms = [a for a in ORDER if any(r["arm"] == a for r in runs)]

    stat: dict = {}
    for a in arms:
        med, lo, hi = [], [], []
        for b in budgets:
            v = sorted(r["best"] for r in runs if r["arm"] == a and r["budget"] == b)
            med.append(v[len(v) // 2])
            lo.append(v[0])
            hi.append(v[-1])
        stat[a] = (np.array(med), np.array(lo), np.array(hi))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.linewidth": 0.9, "axes.labelsize": 10.5, "axes.titlesize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.8,
        "figure.dpi": 130, "savefig.dpi": 300, "axes.axisbelow": True,
    })
    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.62, 1.0],
                                                "wspace": 0.24})

    def draw(ax, ylim=None, zoom=False):
        for a in arms:
            med, lo, hi = stat[a]
            st = dict(STYLE[a])
            lab = st.pop("label")
            if np.any(hi - lo > 1e-9):
                ax.fill_between(budgets, lo, hi, color=st["color"], alpha=0.13,
                                lw=0, zorder=st["zorder"] - 1)
            ax.plot(budgets, med, label=None if zoom else lab, **st)
        ax.set_xticks(budgets)
        ax.set_xlabel("solver calls allowed (the only cost that matters)")
        ax.grid(lw=0.45, color="0.9")
        if ylim:
            ax.set_ylim(*ylim)

    draw(ax)
    ax.set_ylabel("best balanced cross-entropy found  (median of 3 seeds)")
    ax.set_title("(a) every arm, same start, same budget", loc="left",
                 fontweight="bold")
    # The nine calls that buy the derivative, which is why we lose at 12.
    ax.axvspan(budgets[0] - 1.2, 9, color="#fdecea", zorder=0)
    ax.annotate("the anchor Jacobian costs\n$2D{+}1 = 9$ calls before the\n"
                "first step is taken, so at 12\ncalls this project is 4th of 5",
                xy=(12.15, stat["gradient"][0][0] + 0.004), xytext=(15.5, 1.148),
                fontsize=8.6, color="#8c2d24", ha="left",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#8c2d24",
                                connectionstyle="arc3,rad=0.18"))
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.005, 1.005))

    # A zoom, because everything except Nelder-Mead at 12 lives in a band 0.02
    # wide and the crossovers are invisible on the full scale.
    lo_y = min(stat[a][1].min() for a in arms) - 0.0015
    hi_y = 1.0605
    draw(axz, ylim=(lo_y, hi_y), zoom=True)
    axz.set_title("(b) the same curves, zoomed to where the crossings are",
                  loc="left", fontweight="bold")
    axz.set_ylabel("best balanced cross-entropy")
    # Say so when a point is off the top of the zoom, rather than letting the
    # line appear to start at the second budget.
    off = [a for a in arms if stat[a][0][0] > hi_y]
    for n, a in enumerate(off):
        axz.text(budgets[0] + 0.35, hi_y - 0.0013 - n * 0.0040,
                 f"$\\uparrow$ {stat[a][0][0]:.4f}  ({a.replace('_', '-')})",
                 fontsize=7.6, color=STYLE[a]["color"], ha="left", va="top",
                 fontweight="bold")

    # Mark, on the zoom, where gradient descent passes each baseline.
    g = stat["gradient"][0]
    notes = []
    for a in arms:
        if a == "gradient":
            continue
        m = stat[a][0]
        for i in range(len(budgets) - 1):
            if g[i] > m[i] and g[i + 1] <= m[i + 1]:
                x = budgets[i] + (budgets[i + 1] - budgets[i]) * 0.5
                axz.axvspan(budgets[i], budgets[i + 1], color=STYLE[a]["color"],
                            alpha=0.055, lw=0, zorder=0)
                notes.append((x, a, budgets[i], budgets[i + 1]))
    txt = "gradient descent overtakes\n" + "\n".join(
        f"  {a.replace('_', '-')}: between {b0} and {b1} calls"
        for _, a, b0, b1 in sorted(notes, key=lambda t: t[2]))
    axz.text(0.035, 0.035, txt, transform=axz.transAxes, fontsize=8.3,
             va="bottom", ha="left", color="0.2",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.7))

    fig.suptitle(
        "Sample efficiency: gradient descent through the solver is last at a "
        "small budget and first at a large one",
        fontsize=11.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005,
             "192-device design box, d=4, same poor starting corner, "
             "SNN_TRAIN_MODE=frozen, 3 seeds. Bands are min..max over seeds; "
             "gradient descent and Nelder-Mead have none because both are "
             "deterministic.",
             ha="center", fontsize=8.2, color="0.35")
    fig.subplots_adjust(top=0.875, bottom=0.145, left=0.068, right=0.988)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}\nwrote {out.with_suffix('.pdf')}")

    print("\nmedian best loss")
    print(f"{'arm':14s}" + "".join(f"{b:>11d}" for b in budgets))
    for a in arms:
        print(f"{a:14s}" + "".join(f"{v:11.6f}" for v in stat[a][0]))
    print("\ncrossings: " + "; ".join(
        f"{a} between {b0} and {b1}" for _, a, b0, b1 in
        sorted(notes, key=lambda t: t[2])))
    flat = stat["random"][0]
    print(f"\nrandom search from {budgets[0]} to {budgets[-2]} calls: "
          f"{flat[0]:.6f} -> {flat[-2]:.6f}  (change {flat[-2] - flat[0]:+.2e})")
    print(f"gradient over the same range: {g[0]:.6f} -> {g[-2]:.6f}  "
          f"(change {g[-2] - g[0]:+.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
