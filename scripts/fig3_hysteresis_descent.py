#!/usr/bin/env python
"""Figure 3: the device the optimiser is actually moving.

    python scripts/fig3_hysteresis_descent.py

Figures 1 and 2 are arguments about method -- what a device can be, and how many
solver calls each strategy needs. Neither of them shows a device. This one does,
and it is the figure that makes the project legible to somebody who reads
transistors rather than optimisers: the Id-Vg hysteresis loop of the FeFET at
every accepted step of the flagship descent, from the poor starting corner to
the design the loss picked.

The three panels answer the three questions in order:

(a) THE LOOP OPENS. Eight accepted steps, drawn from the same double sweep the
    solver returned at each one -- forward branch (erased, high V_th) and reverse
    branch (programmed, low V_th). The gap between the two branches at the read
    current IS the memory window, and the memory window is what sets the
    separation between the two conductance states the synapse stores. It widens
    0.415 V -> 0.576 V.

(b) WHAT THAT COSTS AND WHAT IT BUYS. The memory window against the subthreshold
    slope, step by step. They move together and UPWARD, which reads as a defect
    to anyone who designs transistors for switching: SS degrades from 71 to 97
    mV/dec. It is not a defect. A thicker HZO film and a thinner interlayer buy
    window at the cost of electrostatic control, and this network needs window
    more than it needs slope. The optimiser found that trade without being told
    it existed, and it is the same trade the V7 ablation could not find.

(c) THE FOUR KNOBS. Where each fabrication parameter went, in its own physical
    units, normalised to the design box so four different quantities share one
    axis. Channel doping is included because it is the knob a scrambled Jacobian
    never finds at all.

Every curve here was returned by DEVSIM at the design point it is attributed to,
and is read back out of `results/cache/devsim/`. No solver call, and nothing is
interpolated -- if a design point were missing from the cache this script would
fail rather than draw a plausible curve.
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
from matplotlib.lines import Line2D  # noqa: E402

from diffsilicon.shared.contract import DEFAULT_VG_GRID, make_oracle_input  # noqa: E402
from diffsilicon.shared.design import get_design  # noqa: E402
from diffsilicon.shared.oracle import extraction_config, run_oracle  # noqa: E402

STEPS = _REPO / "results" / "runs" / "flagship-d4-fixed" / "steps.jsonl"
RESULT = _REPO / "results" / "runs" / "flagship-d4-fixed" / "result.json"
OUT = _REPO / "docs" / "figures" / "fig3_hysteresis_descent.png"

# The read voltage the conductances are sampled at, and the constant-current
# threshold criterion, both come from the frozen circuit -- they are not chosen
# for the picture.
KNOB_LABEL = {
    "t_fe": r"$t_{fe}$  ferroelectric HZO thickness",
    "L_g": r"$L_g$  gate length",
    "log10_N_ch": r"$\log_{10} N_{ch}$  channel doping",
    "t_IL": r"$t_{IL}$  interfacial layer thickness",
}
KNOB_UNIT = {"t_fe": "nm", "L_g": "nm", "log10_N_ch": "cm$^{-3}$", "t_IL": "nm"}
KNOB_COLOR = {"t_fe": "#1b7837", "L_g": "#2166ac",
              "log10_N_ch": "#762a83", "t_IL": "#e08214"}


def accepted_path(rows: list[dict]) -> list[dict]:
    """The distinct design points the run actually stood on, in order.

    A rejected step leaves theta where it was, so the raw log repeats a point
    once per rejection. Drawing those repeats would put eight identical curves on
    top of each other and make the descent look longer than it was.
    """
    out: list[dict] = []
    for r in rows:
        if not out or r["theta"] != out[-1]["theta"]:
            out.append(r)
    # The final theta_next of the last row is where the run ENDED, and it is not
    # the theta of any row. Without it the figure stops one step short of the
    # answer the project reports.
    last = rows[-1]
    if last["theta_next"] != out[-1]["theta"]:
        out.append({"step": last["step"] + 1, "theta": last["theta_next"],
                    "loss": last.get("loss_next", last["loss"]),
                    "oracle_calls": last["oracle_calls"], "final": True})
    return out


def curves_at(theta) -> tuple[np.ndarray, np.ndarray, dict]:
    out = run_oracle(make_oracle_input(np.asarray(theta, dtype=np.float64)),
                     backend="replay")
    id_vg = np.asarray(out.id_vg, dtype=np.float64)
    cfg = extraction_config(np.asarray(theta, dtype=np.float64))
    foms = {"ss": float(out.ss), "vth_fwd": float(out.vth_fwd),
            "vth_rev": float(out.vth_rev), "g_lo": float(out.g_lo),
            "g_hi": float(out.g_hi), "i_leak": float(out.i_leak),
            # The constant-current criterion the two thresholds are DEFINED by.
            # It depends on L_g, so it moves as the optimiser moves the gate --
            # which is why panel (a) marks it per curve rather than drawing one
            # line and implying it is fixed.
            "i_crit": float(cfg.i_crit_per_wl * cfg.w_dev_nm / cfg.l_g_nm)}
    return id_vg[0], id_vg[1], foms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = [json.loads(line) for line in STEPS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    path = accepted_path(rows)
    spec = get_design(4)
    vg = np.asarray(DEFAULT_VG_GRID, dtype=np.float64)

    pts = []
    for i, r in enumerate(path):
        fwd, rev, foms = curves_at(r["theta"])
        phys = dict(zip(spec.names,
                        spec.lo + np.asarray(r["theta"]) * (spec.hi - spec.lo),
                        strict=True))
        pts.append({"i": i, "step": r["step"], "theta": r["theta"], "phys": phys,
                    "fwd": fwd, "rev": rev, "loss": r["loss"],
                    "calls": r.get("oracle_calls", 0),
                    "mw": foms["vth_fwd"] - foms["vth_rev"], **foms})

    n = len(pts)
    print(f"{n} distinct design points on the accepted path, "
          f"all served from results/cache/devsim/ with no solver call")
    for p in pts:
        print(f"  step {p['step']:2d}  loss {p['loss']:.6f}  "
              f"MW {p['mw']:.4f} V  SS {p['ss']:6.2f} mV/dec  "
              f"t_fe {p['phys']['t_fe']:.3f} nm")

    # --- figure ---------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.8, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "figure.dpi": 130, "savefig.dpi": 300, "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(12.6, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.0, 1.0],
                          wspace=0.30, left=0.055, right=0.985,
                          bottom=0.225, top=0.845)
    axA, axB, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    cmap = plt.get_cmap("viridis")
    cols = [cmap(0.06 + 0.86 * i / max(n - 1, 1)) for i in range(n)]

    # (a) the loops -----------------------------------------------------------
    #
    # The y-axis is clipped at 1e-13 A on purpose. Below that the DEVSIM curves
    # are sitting on the extraction's 1e-16 floor and what is plotted is the
    # solver's own numerical noise, not a current; drawn in full it takes two
    # thirds of the axis and hides the switching region the figure is about.
    for p, c in zip(pts, cols, strict=True):
        first_last = p["i"] in (0, n - 1)
        lw = 2.2 if first_last else 1.0
        a = 1.0 if first_last else 0.5
        axA.semilogy(vg, np.abs(p["fwd"]), color=c, lw=lw, alpha=a, zorder=4 + p["i"])
        axA.semilogy(vg, np.abs(p["rev"]), color=c, lw=lw, alpha=a,
                     ls=(0, (3, 1.6)), zorder=4 + p["i"])
        # Where the threshold is read: the constant-current criterion, per curve.
        # It scales as W/L_g, so it moves when the optimiser moves the gate.
        axA.plot([p["vth_fwd"], p["vth_rev"]], [p["i_crit"]] * 2, ls="none",
                 marker="|", ms=7, mew=1.5 if first_last else 0.8,
                 color=c, alpha=a, zorder=20 + p["i"])

    axA.set_ylim(1e-13, 6e-5)
    axA.set_xlim(-1.7, 1.5)

    # The memory window, bracketed where it is actually read -- between the two
    # thresholds. Start below the curves, final above, so the two cannot collide.
    for p, tag, y, col in ((pts[0], "start", 3.0e-12, "#3b3b6d"),
                           (pts[-1], "final", 1.4e-6, "#0d3d1f")):
        axA.annotate("", xy=(p["vth_rev"], y), xytext=(p["vth_fwd"], y),
                     arrowprops=dict(arrowstyle="<|-|>", lw=1.2, color=col,
                                     shrinkA=0, shrinkB=0), zorder=30)
        # 'start' is set to the RIGHT of its own bracket rather than centred on
        # it, because centred it lands on the legend.
        axA.text(p["vth_fwd"] + 0.07 if tag == "start"
                 else 0.5 * (p["vth_fwd"] + p["vth_rev"]),
                 y if tag == "start" else y * 2.6,
                 (f"start: {p['mw']:.3f} V" if tag == "start"
                  else f"final: memory window {p['mw']:.3f} V"),
                 ha="left" if tag == "start" else "center",
                 va="center", fontsize=7.8, color=col, zorder=31,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                           alpha=0.85))

    axA.set_xlabel("gate voltage $V_G$  (V)")
    axA.set_ylabel("drain current $|I_D|$  (A)")
    axA.set_title("(a)  the loop opens along the descent", loc="left", fontsize=9.6)
    axA.grid(lw=0.4, color="0.9", zorder=0)
    axA.legend(handles=[
        Line2D([], [], color="0.35", lw=1.6, label="forward sweep (erased, high $V_{th}$)"),
        Line2D([], [], color="0.35", lw=1.6, ls=(0, (3, 1.6)),
               label="reverse sweep (programmed, low $V_{th}$)"),
        Line2D([], [], color="0.35", ls="none", marker="|", ms=7, mew=1.5,
               label="$V_{th}$ at the constant-current criterion"),
    ], loc="lower left", frameon=True, framealpha=0.93, handlelength=2.2,
        fontsize=7.3, borderpad=0.5)

    # (b) the trade -----------------------------------------------------------
    ss = np.array([p["ss"] for p in pts])
    mw = np.array([p["mw"] for p in pts])
    axB.plot(ss, mw, "-", color="0.55", lw=1.0, zorder=2)
    axB.scatter(ss, mw, c=cols, s=64, edgecolors="white", linewidths=0.8, zorder=4)
    for p, x, y in zip(pts, ss, mw, strict=True):
        if p["i"] in (0, n - 1):
            axB.annotate(f"{'start' if p['i'] == 0 else 'final'}\n"
                         f"loss {p['loss']:.4f}",
                         (x, y), textcoords="offset points",
                         xytext=(30, 4) if p["i"] == 0 else (-6, -32),
                         ha="left" if p["i"] == 0 else "center", fontsize=7.8,
                         color="#3b3b6d" if p["i"] == 0 else "#0d3d1f",
                         arrowprops=dict(arrowstyle="-", lw=0.7, color="0.6",
                                         shrinkA=1, shrinkB=4))
    axB.margins(x=0.14, y=0.17)
    axB.set_xlabel("subthreshold slope  (mV/dec)   $\\longrightarrow$ worse")
    axB.set_ylabel("memory window  (V)   $\\longrightarrow$ better")
    axB.set_title("(b)  slope traded for window, on purpose", loc="left", fontsize=9.6)
    axB.grid(lw=0.4, color="0.9", zorder=0)

    # (c) the knobs -----------------------------------------------------------
    x = np.arange(n)
    ends = []
    for name in spec.names:
        i = spec.names.index(name)
        v = np.array([p["theta"][i] for p in pts])
        axC.plot(x, v, "-o", ms=4.2, lw=1.5, color=KNOB_COLOR[name],
                 label=KNOB_LABEL[name], zorder=4)
        ends.append((float(v[-1]), name, float(spec.lo[i]), float(spec.hi[i]),
                     float(v[0])))

    # Two knobs finish within 0.005 of each other in normalised units, so their
    # end labels have to be pushed apart or they overprint. Sorted by height,
    # then spaced by a fixed minimum, with a leader line back to the real point.
    ends.sort()
    ys: list[float] = []
    for k, e in enumerate(ends):
        ys.append(e[0] if k == 0 else max(e[0], ys[-1] + 0.085))
    for (yv, name, lo, hi, v0), ylab in zip(ends, ys, strict=True):
        axC.annotate(f"{lo + v0 * (hi - lo):.2f} $\\to$ {lo + yv * (hi - lo):.2f} "
                     f"{KNOB_UNIT[name]}",
                     xy=(x[-1], yv), xytext=(x[-1] + 0.45, ylab),
                     textcoords="data", fontsize=7.2, color=KNOB_COLOR[name],
                     va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6,
                                     color=KNOB_COLOR[name], shrinkA=2, shrinkB=1))
    axC.set_xticks(x)
    axC.set_xticklabels([str(p["calls"]) for p in pts], fontsize=7.6)
    axC.set_xlabel("solver calls spent")
    axC.set_ylabel("position in the design box  (0 = min, 1 = max)")
    axC.set_ylim(-0.07, 1.12)
    axC.set_xlim(-0.4, n - 1 + 3.4)
    axC.set_title("(c)  where the four fabrication knobs went", loc="left", fontsize=9.6)
    axC.grid(lw=0.4, color="0.9", zorder=0)
    axC.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=7.2,
               borderpad=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=pts[-1]["calls"]))
    cax = fig.add_axes([0.058, 0.068, 0.25, 0.028])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("solver calls spent", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.6)

    fig.text(0.345, 0.050,
             "Every curve is a DEVSIM double sweep at the design point it is drawn "
             "for, read back from results/cache/devsim/; no solver call, nothing "
             "interpolated. Flagship run "
             f"{result['tag']}: {result['oracle_calls']} calls, loss "
             f"{result['objective_initial']:.4f} $\\to$ "
             f"{result['objective_final']:.4f}, accuracy "
             f"{result['accuracy_initial']:.3f} $\\to$ "
             f"{result['accuracy_final']:.3f}.",
             fontsize=7.6, color="0.3", va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}\nwrote {pdf}")

    bank = _REPO / "results" / "runs" / "fig3_hysteresis_descent.json"
    bank.write_text(json.dumps({
        "source": "results/runs/flagship-d4-fixed/steps.jsonl",
        "backend": "replay(devsim)",
        "solver_calls_made": 0,
        "points": [{k: p[k] for k in
                    ("step", "theta", "loss", "calls", "mw", "ss", "vth_fwd",
                     "vth_rev", "g_lo", "g_hi", "i_leak")} | {"phys": p["phys"]}
                   for p in pts],
    }, indent=2, default=float), encoding="utf-8")
    print(f"wrote {bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
