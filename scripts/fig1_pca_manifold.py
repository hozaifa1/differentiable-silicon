#!/usr/bin/env python
"""Figure 1: what a device can hand the network, and where the free optimum isn't.

THE ARGUMENT THIS FIGURE HAS TO CARRY. The strongest objection to this project is
that phi is only five scalars, so you should skip the solver: optimise those five
freely, then find the device that makes them. The answer is that four fabrication
knobs cannot fill five dimensions -- what a real device can produce is a thin
sheet in R^5 -- and the freely optimised phi* lands off that sheet.

That is three claims and the figure makes each one visible:

  (a) the sheet is THIN. Two principal directions carry 90% of the variation
      across 192 devices sampled evenly over the design box. Four knobs, about
      two effective dimensions. This panel is the measurement, not a picture of
      it.

  (b) phi* IS OFF IT. The 192 devices and the freely optimised phi* in the plane
      those two directions span. The inset repeats the projection against PC3
      instead of PC2, because a single 2-D view of a 5-D set can always be
      accused of being the one projection that made the point.

  (c) AND HERE IS WHY IT CANNOT BE BUILT. Two of the five coordinates are
      outside the range 192 devices spanning the whole design box can reach --
      not near the edge, outside. This panel is what turns "far away in a
      standardised metric" into a statement a device engineer can check.

The mechanism is worth stating in words because it is not subtle: phi* asks for
g_max/g_min = 1.03. Real devices in this box span 2.2 to 6.5e7, median 290. The
free optimum wants a ferroelectric memory whose two stored states conduct almost
identically -- which is not a hard device to fabricate, it is not a memory at
all. That is the objection's proposal, priced.

Coordinates are STANDARDISED by the cloud's own mean and spread before the
principal components are taken. The five entries of phi are a decay of order 0.6,
two conductances of order 1e-6 and 1e-4 siemens, a spike count of order 5 and a
noise fraction of order 0.1; a PCA of raw phi would be a description of the
units. Distances are then quoted in TYPICAL DEVICE SPACINGS -- the median
distance from a device to its own nearest neighbour -- so "far" means "far
compared with how much devices actually differ".

    python scripts/fig1_pca_manifold.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PHI_KEYS = ("beta", "g_min", "g_max", "th_th", "sig_w")
PRETTY = {"beta": r"$\beta$", "g_min": r"$g_{\min}$", "g_max": r"$g_{\max}$",
          "th_th": r"$\vartheta$", "sig_w": r"$\sigma_w$"}

# One accent for the free optimum, one for the device this project found, and
# grey for everything that is context. Colour-blind safe.
C_FREE = "#d1362f"
C_JOINT = "#1b7837"
C_PROJ = "#e08214"
C_BEST = "#2166ac"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=str(_REPO / "results" / "runs"
                                           / "manifold_cloud_d4.json"))
    ap.add_argument("--control", default="",
                    help="v6 control json; defaults to the D5 refit if it exists")
    ap.add_argument("--flagship", default=str(_REPO / "results" / "runs"
                                              / "flagship-d4-fixed" / "result.json"))
    ap.add_argument("--out", default=str(_REPO / "docs" / "figures"
                                         / "fig1_pca_manifold.png"))
    args = ap.parse_args()

    ctrl_path = Path(args.control) if args.control else None
    if ctrl_path is None:
        d5 = _REPO / "results" / "runs" / "v6_manifold_control_d5.json"
        d4 = _REPO / "results" / "runs" / "v6_manifold_control_d4.json"
        ctrl_path = d5 if d5.is_file() else d4
    cloud = _load(Path(args.cloud))
    ctrl = _load(ctrl_path)
    print(f"cloud   {Path(args.cloud).name}  ({cloud['usable']} usable devices)")
    print(f"control {ctrl_path.name}")

    # --- the reachable set, standardised, then its principal axes -----------
    P = np.array([[r["phi"][k] for k in PHI_KEYS] for r in cloud["points"]])
    mu, sd = P.mean(0), P.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (P - mu) / sd
    _, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    frac = (s**2) / (s**2).sum()
    S = (Z - Z.mean(0)) @ vt.T
    cum2 = frac[0] + frac[1]
    print("explained variance: " + "  ".join(f"PC{i+1} {f*100:.1f}%"
                                              for i, f in enumerate(frac)))
    print(f"PC1+PC2 = {cum2*100:.1f}%")

    def to_scores(phi_vec):
        return ((np.asarray(phi_vec) - mu) / sd - Z.mean(0)) @ vt.T

    phi_star = np.array([ctrl["free"]["phi"][k] for k in PHI_KEYS])
    s_star = to_scores(phi_star)
    j_near = int(ctrl["projected"]["index"])
    i_best = int(ctrl["best_in_cloud"]["index"])
    spacings = float(ctrl["reachability"]["in_spacings"])
    l_free = float(ctrl["free"]["loss"])
    l_proj = float(ctrl["projected"]["loss"])
    l_best = float(ctrl["best_in_cloud"]["loss"])
    losses = np.array(ctrl["all_cloud_losses"], dtype=float)

    flag = _load(Path(args.flagship)) if Path(args.flagship).is_file() else None
    s_joint = l_joint = None
    if flag is not None:
        s_joint = to_scores([flag["phi_final"][k] for k in PHI_KEYS])
        l_joint = float(flag["objective_final"])
        print(f"flagship final loss {l_joint:.6f} in {flag['oracle_calls']} calls")

    # --- figure --------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.8, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "figure.dpi": 130, "savefig.dpi": 300, "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(12.0, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.70, 1.34, 1.02],
                          wspace=0.32, left=0.055, right=0.975,
                          bottom=0.235, top=0.855)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # (a) how many dimensions the reachable set actually occupies
    idx = np.arange(1, 6)
    ax0.bar(idx, frac * 100, color="#9ecae1", edgecolor="#3182bd", linewidth=0.8,
            width=0.62, zorder=3)
    ax0.plot(idx, np.cumsum(frac) * 100, "o-", color="#08519c", ms=3.4, lw=1.2,
             zorder=4, label="cumulative")
    ax0.axhline(90, color=C_FREE, ls=(0, (4, 2)), lw=1.0, zorder=2)
    ax0.text(5.35, 90, "90%", color=C_FREE, fontsize=8, va="center", ha="right")
    ax0.set_xticks(idx)
    ax0.set_xlabel("principal direction")
    ax0.set_ylabel("variance explained (%)")
    ax0.set_ylim(0, 104)
    ax0.set_title("(a) the sheet is thin", loc="left", fontweight="bold")
    ax0.annotate(f"PC1+PC2 = {cum2 * 100:.1f}%\n4 knobs, ~2 dimensions",
                 xy=(2, cum2 * 100), xytext=(2.55, 55),
                 fontsize=8, color="#08519c",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#08519c"))
    ax0.grid(axis="y", lw=0.4, color="0.88", zorder=0)
    ax0.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.38))

    # (b) the reachable sheet, and phi* off it
    def panel(ax, jy, small=False):
        sc = ax.scatter(S[:, 0], S[:, jy], c=losses, cmap="viridis_r",
                        s=8 if small else 17,
                        linewidths=0.25, edgecolors="white", zorder=3)
        ax.plot([s_star[0], S[j_near, 0]], [s_star[jy], S[j_near, jy]],
                color=C_FREE, lw=1.0, ls=(0, (3, 2)), zorder=4)
        ax.scatter([S[j_near, 0]], [S[j_near, jy]], s=40 if small else 95,
                   facecolors="none", edgecolors=C_PROJ,
                   linewidths=1.2 if small else 1.7, zorder=5)
        if not small:
            ax.scatter([S[i_best, 0]], [S[i_best, jy]], s=52, marker="s",
                       facecolors="none", edgecolors=C_BEST, linewidths=1.5,
                       zorder=5)
        if s_joint is not None:
            ax.scatter([s_joint[0]], [s_joint[jy]], s=40 if small else 95,
                       marker="D", facecolors=C_JOINT, edgecolors="white",
                       linewidths=0.7, zorder=6)
        ax.scatter([s_star[0]], [s_star[jy]], s=90 if small else 210, marker="*",
                   color=C_FREE, edgecolors="white", linewidths=0.7, zorder=7)
        ax.grid(lw=0.4, color="0.9", zorder=0)
        return sc

    sc = panel(ax1, 1)
    ax1.set_xlabel(f"PC1  ({frac[0] * 100:.1f}% of variance)")
    ax1.set_ylabel(f"PC2  ({frac[1] * 100:.1f}%)")
    ax1.set_title(r"(b) $\phi^*$ is not on the sheet", loc="left",
                  fontweight="bold")
    ax1.text(0.025, 0.028,
             f"$\\phi^*$ is {spacings:.1f} typical\n"
             f"device-spacings from the\n"
             f"nearest buildable device,\n"
             f"measured in full 5-D",
             transform=ax1.transAxes, fontsize=7.8, va="bottom", ha="left",
             color="0.25",
             bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="0.82", lw=0.6))

    # The same picture against PC3, so that no one can say the projection was
    # chosen to make the point.
    axi = ax1.inset_axes([0.632, 0.045, 0.352, 0.375])
    panel(axi, 2, small=True)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_title("vs PC3", fontsize=7.5, pad=2.0, color="0.3")
    for sp in axi.spines.values():
        sp.set_color("0.7")

    # (c) why it cannot be built: two coordinates out of range
    lo_d, hi_d = P.min(0), P.max(0)
    phi_j = (np.array([flag["phi_final"][k] for k in PHI_KEYS])
             if flag is not None else None)
    ypos = np.arange(len(PHI_KEYS))[::-1]
    out_flags = []
    for i, y in enumerate(ypos):
        # Everything scaled into the device range, so all five share one axis:
        # 0 is the smallest device, 1 the largest.
        span = hi_d[i] - lo_d[i]
        span = span if span > 0 else 1.0
        f_star = (phi_star[i] - lo_d[i]) / span
        outside = phi_star[i] < lo_d[i] or phi_star[i] > hi_d[i]
        out_flags.append(outside)
        ax2.plot([0, 1], [y, y], lw=6.5, color="#cfe3f3",
                 solid_capstyle="butt", zorder=2)
        ax2.plot([0, 1], [y, y], lw=1.0, color="#3182bd", zorder=3)
        if phi_j is not None:
            ax2.scatter([(phi_j[i] - lo_d[i]) / span], [y], marker="D", s=42,
                        color=C_JOINT, edgecolors="white", linewidths=0.7,
                        zorder=5)
        ax2.scatter([np.clip(f_star, -0.42, 1.42)], [y], marker="*", s=205,
                    color=C_FREE if outside else "#b0b0b0",
                    edgecolors="white", linewidths=0.7, zorder=6)
        if outside:
            ax2.annotate("", xy=(np.clip(f_star, -0.42, 1.42), y),
                         xytext=(1.0 if f_star > 1 else 0.0, y),
                         arrowprops=dict(arrowstyle="-|>", color=C_FREE, lw=1.1))
    ax2.axvspan(0, 1, color="#f2f8fd", zorder=0)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([PRETTY[k] for k in PHI_KEYS], fontsize=11)
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(-2.05, len(PHI_KEYS) - 0.45)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["smallest\ndevice", "largest\ndevice"], fontsize=8)
    ax2.set_xlabel("position within the reachable range")
    ax2.set_title(f"(c) {sum(out_flags)} of 5 coordinates are out of reach",
                  loc="left", fontweight="bold")
    ax2.grid(axis="x", lw=0.4, color="0.9", zorder=1)
    ratio = float(phi_star[2] / phi_star[1]) if phi_star[1] > 0 else float("nan")
    dev_ratio = P[:, 2] / np.where(P[:, 1] > 0, P[:, 1], np.nan)
    ax2.text(0.5, -1.42,
             f"$\\phi^*$ wants $g_{{\\max}}/g_{{\\min}} = {ratio:.2f}$; real devices\n"
             f"span {np.nanmin(dev_ratio):.1f} to {np.nanmax(dev_ratio):.0e} "
             f"(median {np.nanmedian(dev_ratio):.0f}). A memory whose\n"
             f"two states conduct alike is not a memory.",
             fontsize=7.8, ha="center", va="center", color="0.25",
             bbox=dict(boxstyle="round,pad=0.36", fc="white", ec="0.82", lw=0.6))
    print("  out-of-range coordinates: "
          f"{[k for k, o in zip(PHI_KEYS, out_flags, strict=True) if o]}")
    print(f"  g_max/g_min at phi* = {ratio:.3f}; devices "
          f"{np.nanmin(dev_ratio):.3g}..{np.nanmax(dev_ratio):.3g} "
          f"(median {np.nanmedian(dev_ratio):.4g})")

    cax = fig.add_axes([0.988, 0.235, 0.0082, 0.62])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("balanced cross-entropy", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    handles = [
        Line2D([], [], ls="none", marker="o", mfc="#4c9a76", mec="white", ms=5.5,
               label=f"{len(P)} reachable devices"),
        Line2D([], [], ls="none", marker="*", color=C_FREE, ms=13,
               label=f"free optimum $\\phi^*$  ({l_free:.4f}, 0 solver calls)"),
        Line2D([], [], ls="none", marker="o", mfc="none", mec=C_PROJ, mew=1.7,
               ms=8, label=f"nearest buildable device  ({l_proj:.4f})"),
        Line2D([], [], ls="none", marker="s", mfc="none", mec=C_BEST, mew=1.5,
               ms=6.5, label=f"best of the {len(P)} sampled  ({l_best:.4f})"),
    ]
    if s_joint is not None:
        handles.append(
            Line2D([], [], ls="none", marker="D", mfc=C_JOINT, mec="white", ms=7,
                   label=f"descent through the solver  ({l_joint:.4f}, "
                         f"{flag['oracle_calls']} calls)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.002), handletextpad=0.45,
               columnspacing=1.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}")
    print(f"wrote {pdf}")

    # The numbers the caption has to quote, printed so they are never retyped.
    print("\ncaption numbers")
    print(f"  devices                 {len(P)}")
    print(f"  PC1, PC2                {frac[0]*100:.1f}%, {frac[1]*100:.1f}%  "
          f"(sum {cum2*100:.1f}%)")
    print(f"  phi* off the sheet      {spacings:.2f} typical device-spacings")
    print(f"  free / projected        {l_free:.6f} / {l_proj:.6f}")
    print(f"  best of {len(P)}             {l_best:.6f}")
    if l_joint is not None:
        print(f"  joint descent           {l_joint:.6f} in "
              f"{flag['oracle_calls']} solver calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
