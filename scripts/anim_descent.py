#!/usr/bin/env python
"""The descent, as footage: the FeFET's hysteresis loop opening step by step.

    uv run python scripts/anim_descent.py

Figure 3 draws all eight accepted steps at once, which is the right static
picture and the wrong moving one. On a screen recording nobody reads eight
overlaid curves. They watch one curve open.

Every frame here is a design point DEVSIM actually solved, read back out of
`results/cache/devsim/`. Nothing between two steps is interpolated: the loop
jumps because the optimiser jumped, and the frame count per step is a hold, not
a tween. That is the same rule the rest of the project runs on, applied to a GIF.

Writes:
    docs/figures/anim_descent.gif      loops, drop straight into a video timeline
    docs/figures/anim_frames/*.png     the same frames at 1920x1080, for editing
    docs/figures/anim_descent.mp4      only if ffmpeg is on PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "fig3_hysteresis_descent", Path(__file__).resolve().parent / "fig3_hysteresis_descent.py"
)
f3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f3)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from diffsilicon.shared.contract import DEFAULT_VG_GRID  # noqa: E402
from diffsilicon.shared.design import get_design  # noqa: E402

OUT_DIR = _REPO / "docs" / "figures"

# Dark, because this sits between terminal recordings in the video and a white
# figure flashing up between two dark shots is the thing that makes an edit look
# assembled rather than made.
INK = "#e8e8ea"
BG = "#0e0f13"
PANEL = "#15171d"
GRID = "#262a33"
HOT = "#ffb347"
COOL = "#4fc3f7"
KNOB_COLOR = {"t_fe": "#4ade80", "L_g": "#60a5fa",
              "log10_N_ch": "#c084fc", "t_IL": "#fbbf24"}
KNOB_LABEL = {"t_fe": "ferroelectric thickness", "L_g": "gate length",
              "log10_N_ch": "channel doping", "t_IL": "interlayer"}
KNOB_UNIT = {"t_fe": "nm", "L_g": "nm", "log10_N_ch": "", "t_IL": "nm"}

HOLD = 14        # frames per design point
TAIL = 26        # extra frames on the final device
FPS = 12


def load_points():
    rows = [
        json.loads(line)
        for line in f3.STEPS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path = f3.accepted_path(rows)
    spec = get_design(4)
    pts = []
    for i, r in enumerate(path):
        fwd, rev, foms = f3.curves_at(r["theta"])
        phys = dict(
            zip(spec.names, spec.lo + np.asarray(r["theta"]) * (spec.hi - spec.lo), strict=True)
        )
        pts.append(
            {"i": i, "step": r["step"], "theta": list(map(float, r["theta"])), "phys": phys,
             "fwd": fwd, "rev": rev, "loss": float(r["loss"]),
             "calls": int(r.get("oracle_calls", 0)),
             "mw": foms["vth_fwd"] - foms["vth_rev"], **foms}
        )
    return pts, spec


def build(pts, spec, out_dir: Path, save_frames: bool):
    vg = np.asarray(DEFAULT_VG_GRID, dtype=np.float64)
    n = len(pts)
    names = list(spec.names)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": PANEL, "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.22, hspace=0.34,
                          left=0.065, right=0.975, bottom=0.085, top=0.845)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    title = fig.text(0.065, 0.945, "", fontsize=25, fontweight="bold", color=INK, ha="left")
    sub = fig.text(0.065, 0.900, "", fontsize=14, color="#9aa3b2", ha="left")
    stamp = fig.text(0.975, 0.945, "", fontsize=14, color="#9aa3b2", ha="right")

    # (A) the loop -------------------------------------------------------------
    axA.set_yscale("log")
    # The solver sweeps -3.5 V to +1.5 V and the left two volts are flat floor on
    # every one of these devices. Figure 3 draws the whole sweep; this is footage,
    # so it opens on the part that moves.
    axA.set_xlim(-1.5, float(vg.max()))
    axA.set_ylim(1e-13, 3e-4)
    axA.set_xlabel("gate voltage  $V_g$  (V)", fontsize=13)
    axA.set_ylabel("drain current  $I_d$  (A)", fontsize=13)
    axA.grid(True, which="major", color=GRID, lw=0.7)
    axA.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.5)
    axA.set_title("the device the loss is moving", fontsize=15, color=INK, pad=10)

    ghosts = []
    for _ in range(n):
        gf, = axA.plot([], [], lw=1.1, color=COOL, alpha=0.16)
        gr, = axA.plot([], [], lw=1.1, color=HOT, alpha=0.16)
        ghosts.append((gf, gr))
    liveF, = axA.plot([], [], lw=2.8, color=COOL, label="erased  (high $V_{th}$)")
    liveR, = axA.plot([], [], lw=2.8, color=HOT, label="programmed  (low $V_{th}$)")
    critline = axA.axhline(1e-9, color="#8b93a3", lw=1.0, ls=(0, (4, 3)), alpha=0.0)
    mwspan = axA.axvspan(0, 0, color="#ffffff", alpha=0.0)
    mwtext = axA.text(0.0, 0.0, "", fontsize=13, color=INK, ha="center", va="bottom")
    leg = axA.legend(loc="upper left", frameon=True, fontsize=11)
    leg.get_frame().set_facecolor(PANEL)
    leg.get_frame().set_edgecolor(GRID)

    # (B) the loss -------------------------------------------------------------
    losses = [p["loss"] for p in pts]
    calls = [p["calls"] for p in pts]
    axB.set_xlim(-2, max(calls) + 4)
    pad = 0.06 * (max(losses) - min(losses))
    axB.set_ylim(min(losses) - pad, max(losses) + pad)
    axB.set_xlabel("solver calls", fontsize=12)
    axB.set_ylabel("balanced cross-entropy", fontsize=12)
    axB.grid(True, color=GRID, lw=0.7)
    axB.set_title("what it costs", fontsize=13, color=INK, pad=8)
    lossline, = axB.plot([], [], lw=2.2, color="#4ade80", marker="o", ms=6,
                         mfc="#4ade80", mec=BG)

    # (C) the four knobs -------------------------------------------------------
    axC.set_xlim(0, 1)
    axC.set_ylim(-0.6, len(names) - 0.4)
    axC.set_yticks(range(len(names)))
    axC.set_yticklabels([KNOB_LABEL[k] for k in names], fontsize=11)
    axC.invert_yaxis()
    axC.set_xlabel("position in the fabrication box", fontsize=12)
    axC.grid(True, axis="x", color=GRID, lw=0.7)
    axC.set_title("where the four knobs went", fontsize=13, color=INK, pad=8)
    bars = axC.barh(range(len(names)), [0] * len(names), height=0.5,
                    color=[KNOB_COLOR[k] for k in names])
    knobtexts = [axC.text(0.012, y, "", va="center", fontsize=11, color=BG,
                          fontweight="bold") for y in range(len(names))]

    def draw(idx: int):
        p = pts[idx]
        for j in range(n):
            gf, gr = ghosts[j]
            if j < idx:
                gf.set_data(vg, np.clip(pts[j]["fwd"], 1e-16, None))
                gr.set_data(vg, np.clip(pts[j]["rev"], 1e-16, None))
            else:
                gf.set_data([], [])
                gr.set_data([], [])
        liveF.set_data(vg, np.clip(p["fwd"], 1e-16, None))
        liveR.set_data(vg, np.clip(p["rev"], 1e-16, None))

        critline.set_ydata([p["i_crit"], p["i_crit"]])
        critline.set_alpha(0.65)
        lo, hi = sorted((p["vth_rev"], p["vth_fwd"]))
        # axvspan returns a Rectangle on current matplotlib, not a Polygon, so
        # this moves it by x/width rather than by rewriting its vertices.
        mwspan.set_x(lo)
        mwspan.set_width(hi - lo)
        mwspan.set_alpha(0.10)
        mwtext.set_position(((lo + hi) / 2.0, p["i_crit"] * 2.2))
        mwtext.set_text(f"memory window  {p['mw']:.3f} V")

        lossline.set_data(calls[: idx + 1], losses[: idx + 1])

        for b, t, k in zip(bars, knobtexts, names, strict=True):
            b.set_width(max(p["theta"][names.index(k)], 0.004))
            unit = KNOB_UNIT[k]
            t.set_text(f"{p['phys'][k]:.2f} {unit}".strip())

        title.set_text(f"balanced cross-entropy   {p['loss']:.4f}")
        sub.set_text(
            f"memory window {p['mw']:.3f} V     subthreshold slope {p['ss']:.1f} mV/dec"
        )
        last = "  final device" if idx == n - 1 else ""
        stamp.set_text(f"accepted step {idx} of {n - 1}     {p['calls']} solver calls{last}")

    order = []
    for i in range(n):
        order += [i] * (HOLD + (TAIL if i == n - 1 else 0))

    def update(fr):
        draw(order[fr])
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(order), interval=1000 // FPS)

    gif = out_dir / "anim_descent.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=FPS))
    print(f"wrote {gif}  ({len(order)} frames, {len(order) / FPS:.1f} s)")

    try:
        mp4 = out_dir / "anim_descent.mp4"
        anim.save(mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=6000))
        print(f"wrote {mp4}")
    except Exception as exc:  # ffmpeg is not a dependency of this repository
        print(f"no mp4: {type(exc).__name__}. The gif and the frames are enough for an editor.")

    if save_frames:
        fdir = out_dir / "anim_frames"
        fdir.mkdir(parents=True, exist_ok=True)
        for k, idx in enumerate(order):
            draw(idx)
            fig.savefig(fdir / f"frame_{k:04d}.png", dpi=120)
        print(f"wrote {len(order)} frames to {fdir}")

    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--frames", action="store_true",
                    help="also write every frame as a 1920x1080 png")
    args = ap.parse_args()

    pts, spec = load_points()
    print(f"{len(pts)} accepted design points, every curve replayed from results/cache/devsim/")
    for p in pts:
        print(f"  step {p['step']:2d}  loss {p['loss']:.6f}  MW {p['mw']:.4f} V  "
              f"SS {p['ss']:6.2f} mV/dec")
    build(pts, spec, Path(args.out_dir), args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
